# 直接去学习mask即可
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import SimpleITK as sitk
import torchio as tio
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from skimage import measure
from skimage.measure import label, regionprops
import nibabel as nib


PREPROCESSING_TRANSFORMS = tio.Compose([
    tio.Clamp(out_min=-1000, out_max=400),
    tio.RescaleIntensity(in_min_max=(-1000, 400), out_min_max=(-1.0, 1.0)),
    tio.CropOrPad(target_shape=(32, 64, 64))
])

PREPROCESSING_MASK_TRANSFORMS = tio.Compose([
    tio.CropOrPad(target_shape=(32, 64, 64))
])

TRAIN_TRANSFORMS = tio.Compose([
    tio.RandomFlip(axes=(1,), flip_probability=0.5),
])

class LIDCMASKDataset(Dataset):
    def __init__(self, root_dir, text_txt_path,augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = text_txt_path
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSFORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSFORMS

    def train_transform(self, image, label, p):
        train_transforms = tio.Compose([
            tio.RandomFlip(axes=(1,), flip_probability=p),
        ])
        image = train_transforms(image)
        label = train_transforms(label)
        return image, label

    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(self.root_dir, './*.nii.gz'), recursive=True)
        test_file_names = set()

        with open(self.remove_test_path, 'r') as file:
            for line in file:
                test_file_name = line.strip()
                test_file_names.add(test_file_name)

        filtered_file_names = [
            f for f in all_file_names
            if os.path.basename(f)[:-7] not in test_file_names
        ]
        for f in all_file_names:
            print(os.path.basename(f)[:-7])
            break

        print("FILTERED: ", len(filtered_file_names))
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, index):
        path = self.file_names[index]

        img = tio.ScalarImage(path)
        mask_path = path.replace("/Image/", "/Mask/")
        filename = mask_path.split('/')[-1]
        new_filename = filename.replace("Vol", "Mask")
        mask_path = mask_path.replace(filename, new_filename)
        mask = tio.LabelMap(mask_path)

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        p = np.random.choice([0, 1])
        img, mask = self.train_transform(img, mask, p)

        img = img.data
        mask = mask.data  # (1, D, H, W)

        hist = torch.histc(img[mask > 0], bins=16, min=-1, max=1) / mask.sum()
        if torch.sum(hist) == 0 or torch.isnan(hist).any():
            print(index, mask.sum(), "----", hist)


        mask_np = mask[0].cpu().numpy()

        labeled = label(mask_np)
        props = regionprops(labeled)

        sphere_full = np.zeros_like(mask_np, dtype=bool)


        for region in props:
            region_mask = np.zeros_like(mask_np, dtype=bool)
            region_mask[tuple(region.coords.T)] = True

            center, radius = self.min_enclosing_sphere(torch.from_numpy(region_mask))
            sphere_mask = self.create_sphere_mask(mask_np.shape, center, radius)

            sphere_full |= sphere_mask.numpy().astype(bool) 

        sphere = torch.from_numpy(sphere_full.astype(np.uint8)).unsqueeze(0)
        sphere = sphere * 2 - 1
        mask = mask * 2 - 1

        return {
            'data': img,
            'label': mask,
            'hist': hist,
            'sphere': sphere
        }

    @staticmethod
    def create_sphere_mask(shape, center, radius):
        Z, X, Y = np.mgrid[:shape[0], :shape[1], :shape[2]]
        dist_from_center = (Y - center[2])**2 + (X - center[1])**2 + (Z - center[0])**2
        mask = dist_from_center <= radius**2
        return torch.tensor(mask, dtype=torch.uint8)

    @staticmethod
    def min_enclosing_sphere(mask):
        indices = torch.nonzero(mask)
        if len(indices) == 0:
            return (0, 0, 0), 0

        points = indices.numpy()
        center = points.mean(axis=0)
        radius = np.max(np.linalg.norm(points - center, axis=1))
        
        
        return center.astype(int), int(radius)


def save_sphere_visualization(sphere_tensor, save_dir="generated_spheres", base_name="sphere"):

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




