import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import torchio as tio
import os
import glob

PREPROCESSING_TRANSORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400),
                         out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

PREPROCESSING_MASK_TRANSORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1), flip_probability=0.5),
])

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import os
import torch

def save_sphere_visualization(sphere_tensor, save_dir="generated_spheres", base_name="sphere"):
    """
    Sphere visualisation
    """
    os.makedirs(save_dir, exist_ok=True)

    if isinstance(sphere_tensor, torch.Tensor):
        sphere_np = sphere_tensor.squeeze().detach().cpu().numpy()
    else:
        sphere_np = sphere_tensor.squeeze()
    nii_path = os.path.join(save_dir, f"{base_name}.nii.gz")
    nib.save(nib.Nifti1Image(sphere_np.astype(np.uint8), affine=np.eye(4)), nii_path)

    z_slice = np.argmax(sphere_np.sum(axis=(1, 2)))
    plt.imshow(sphere_np[z_slice], cmap="gray", vmin=0, vmax=1)
    plt.title(f"slice={z_slice}, voxels={int(sphere_np.sum())}")
    plt.show()

    img_path = os.path.join(save_dir, f"{base_name}_slice.png")
    plt.savefig(img_path, bbox_inches="tight", dpi=150)
    plt.close()

    print(f"[Saved] {nii_path} and {img_path} | Voxels={int(sphere_np.sum())}")

class LIDCMASKInDataset(Dataset):
    def __init__(self, root_dir, test_txt_path, augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_path
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS

    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(self.root_dir, './**/*.nii.gz'), recursive=True)
        test_file_names = set()
        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip() 
                test_file_names.add(test_file_name)
        filtered_file_names = [
            f for f in all_file_names
            if os.path.basename(f)[:-7] not in test_file_names  
        ]
        return sorted(filtered_file_names)

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]
        img = tio.ScalarImage(path)
        mask_path = path.replace("/Image/", "/Mask/")
        filename = mask_path.split('/')[-1]
        new_filename = filename.replace("Vol", "Mask")
        print("NEW: ", new_filename)
        mask_path = mask_path.replace(filename, new_filename)
        print(mask_path)
        mask = tio.LabelMap(mask_path)  
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)
        affine = img.affine
        mask = mask.data
        img = img.data
        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)
            print(img[mask > 0])
        sphere_list = []
        for c in range(mask.shape[0]):  # для каждого канала
            sphere_mask = self.create_random_spheres_mask(mask[c].shape)
            save_sphere_visualization(sphere_mask, save_dir="viz_spheres", base_name=f"sphere_{index}_{c}")
            sphere_list.append(sphere_mask)

        sphere = torch.stack(sphere_list, dim=0)
        sphere = sphere.float() * 2 - 1
        return {
            'GT': img,
            'GT_name': filename,
            'gt_keep_mask': mask,
            'affine': affine,
            'sphere': sphere,
        }

    @staticmethod
    def min_enclosing_sphere(mask):
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            return (0, 0, 0), 0
        points = indices.numpy()
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        print("RADIUS: ", radius)
        return center.astype(int), int(radius)

    @staticmethod
    def create_sphere_mask(shape, center, radius):
        Z, X, Y = np.mgrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = (Y - center[2]) ** 2 + (X - center[1]) ** 2 + (Z - center[0]) ** 2
        mask = dist_from_center <= radius ** 2
        return torch.tensor(mask, dtype=torch.uint8)

    @staticmethod
    def create_random_spheres_mask(shape,
                                  num_spheres_range=(3, 9),
                                  radius_range=(3, 9),
                                  edge_margin=3,
                                  z_focus=(0.3, 0.8),
                                  anisotropy_range=(0.8, 1.3),
                                  max_tries=20):
        """
        Generation of spheres for inference.
        """
        Z, X, Y = shape
        mask = np.zeros((Z, X, Y), dtype=np.uint8)
        num_spheres = np.random.randint(num_spheres_range[0], num_spheres_range[1] + 1)

        z_min = int(Z * z_focus[0])
        z_max = int(Z * z_focus[1])

        for _ in range(num_spheres):
            r = np.random.randint(radius_range[0], radius_range[1] + 1)


            cz = np.random.randint(max(r, z_min), min(Z - r, z_max))
            cx = np.random.randint(edge_margin, X - edge_margin)
            cy = np.random.randint(edge_margin, Y - edge_margin)

            rz, rx, ry = r * np.random.uniform(*anisotropy_range, 3)

            zz, xx, yy = np.ogrid[:Z, :X, :Y]
            lesion = ((zz - cz) / rz)**2 + ((xx - cx) / rx)**2 + ((yy - cy) / ry)**2 <= 1

            mask |= lesion

        total_voxels = int(mask.sum())
        print(f"Generated {num_spheres} lesions, total voxels ≈ {total_voxels}")
        return torch.tensor(mask, dtype=torch.uint8)



