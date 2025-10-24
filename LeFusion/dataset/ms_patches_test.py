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

class MSPatchesTestDataset(Dataset):
    def __init__(self, root_dir='', test_txt_dir='', augmentation=False, max_pairs=10):
        self.root_dir = root_dir
        self.remove_test_path = test_txt_dir
        self.max_pairs = max_pairs
        self.file_pairs = self.get_file_pairs()
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

    def get_file_pairs(self):
        """Find matching image-mask pairs from your MS_data/patches structure"""
        file_pairs = []
        
        # Look for all patient directories (P1, P2, P3, P4, etc.)
        # Fix: self.root_dir is already MS_data/patches, so we don't need to add it again
        patient_dirs = glob.glob(os.path.join(self.root_dir, 'P*'))
        
        for patient_dir in patient_dirs:
            # Find all image files (containing 'image' in filename)
            image_files = glob.glob(os.path.join(patient_dir, '*_image.nii.gz'))
            
            for image_file in image_files:
                # Find corresponding mask file
                base_name = image_file.replace('_image.nii.gz', '')
                mask_file = base_name + '_mask.nii.gz'
                
                if os.path.exists(mask_file):
                    file_pairs.append((image_file, mask_file))
                    
                    # Stop when we have enough pairs
                    if len(file_pairs) >= self.max_pairs:
                        break
                else:
                    print(f"Warning: No mask found for {image_file}")
            
            # Stop if we have enough pairs
            if len(file_pairs) >= self.max_pairs:
                break
        
        print(f"Found {len(file_pairs)} image-mask pairs (limited to {self.max_pairs})")
        return file_pairs

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, index):
        image_path, mask_path = self.file_pairs[index]

        # Load image and mask
        img = tio.ScalarImage(image_path)
        mask = tio.LabelMap(mask_path)

        # Apply preprocessing
        img = self.preprocessing_img(img)
        mask = self.preprocessing_mask(mask)

        # Apply augmentation if enabled
        if self.augmentation:
            p = np.random.random()
            img, mask = self.train_transform(img, mask, p)

        # Calculate histogram features for conditioning
        hist = self.calculate_histogram(img.data, mask.data)

        return {
            'data': img.data,
            'label': mask.data,
            'hist': hist,
            'GT_name': os.path.basename(image_path)
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
