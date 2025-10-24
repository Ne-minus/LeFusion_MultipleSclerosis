import numpy as np
import torch
from torch.utils.data.dataset import Dataset
import os
import glob
import torchio as tio
import matplotlib.pyplot as plt


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

class MSOriginalDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS

    def train_transform(self, image, label, p):
        TRAIN_TRANSFORMS = tio.Compose([
            tio.RandomFlip(axes=(1), flip_probability=p),
        ])
        image = TRAIN_TRANSFORMS(image)
        label = TRAIN_TRANSFORMS(label)
        return image, label

    def get_file_names(self):
        all_file_names = glob.glob(os.path.join(self.root_dir, './**/*.nii.gz'), recursive=True)

        test_file_names = set()
        if self.remove_test_path and os.path.exists(self.remove_test_path):
            with open(self.remove_test_path, 'r') as file:
                for line in file:
                    test_file_name = line.strip()  
                    test_file_names.add(test_file_name)

        filtered_file_names = [
            f for f in all_file_names
            if os.path.basename(f)[:-7] not in test_file_names 
        ]
        return filtered_file_names

    def __len__(self):
        return len(self.file_names)

    @staticmethod
    def create_mask(shape):
        return torch.zeros(shape, dtype=torch.uint8)

    @staticmethod 
    def project_to_2d(mask):
        projection = torch.max(mask, dim=0)[0]
        return projection.numpy()

    def __getitem__(self, index):
        path = self.file_names[index] 

        img = tio.ScalarImage(path)
 
        # Assuming mask files are in a 'masks' subdirectory
        mask_path = path.replace("/images", "/masks")
        # Or if masks are in the same directory with different naming
        # mask_path = path.replace(".nii.gz", "_mask.nii.gz")

        mask = tio.LabelMap(mask_path) 

        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        if self.augmentation:
            p = np.random.random()
            img, mask = self.train_transform(img, mask, p)

        # Calculate histogram features for conditioning (single lesion type)
        hist = self.calculate_histogram(img.data, mask.data)

        return {
            'data': img.data,
            'label': mask.data,
            'hist': hist,
            'GT_name': os.path.basename(path)
        }

    def calculate_histogram(self, image_tensor, mask_tensor):
        """Calculate histogram features for conditioning"""
        # Convert to numpy and flatten
        img_np = image_tensor.squeeze().numpy()
        mask_np = mask_tensor.squeeze().numpy()
        
        # Only consider lesion regions (mask > 0)
        lesion_pixels = img_np[mask_np > 0]
        
        if len(lesion_pixels) > 0:
            # Calculate histogram
            hist, bins = np.histogram(lesion_pixels, bins=16, range=(-1, 1))
            hist = hist.astype(np.float32)
            
            # Normalize histogram
            hist = hist / (hist.sum() + 1e-8)
        else:
            # No lesions, use zero histogram
            hist = np.zeros(16, dtype=np.float32)
        
        return torch.from_numpy(hist)
