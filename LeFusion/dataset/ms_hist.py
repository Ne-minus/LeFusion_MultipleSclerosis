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

class MSDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.file_names = self.get_file_names()
        self.augmentation = augmentation
        self.preprocessing_img = PREPROCESSING_TRANSORMS
        self.preprocessing_mask = PREPROCESSING_MASK_TRANSORMS
        
        # Define MS lesion types (adjust based on your data)
        self.lesion_types = [1, 2, 3]  # e.g., T1, T2, FLAIR lesions

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

        # Handle multiple lesion types in MS
        mask_data = mask.data
        
        # Calculate histograms for each lesion type
        histograms = []
        
        for lesion_type in self.lesion_types:
            if (mask_data == lesion_type).sum().item() > 0:
                # Calculate histogram for this lesion type
                lesion_hist = torch.histc(
                    img.data[mask_data == lesion_type], 
                    bins=16, min=-1, max=1
                ) / (mask_data == lesion_type).sum()
            else:
                # No lesions of this type, use zero histogram
                lesion_hist = torch.zeros(16)
            histograms.append(lesion_hist)
        
        # Combine histograms for all lesion types
        hist_combined = torch.cat(histograms, dim=0)
        
        # Duplicate data and mask for multi-channel processing
        # This allows the model to learn different lesion types separately
        num_lesion_types = len(self.lesion_types)
        img_repeated = img.data.repeat(num_lesion_types, 1, 1, 1)
        mask_repeated = mask_data.repeat(num_lesion_types, 1, 1, 1)

        return {
            'data': img_repeated,
            'label': mask_repeated,
            'hist': hist_combined,
            'GT_name': os.path.basename(path)
        }
