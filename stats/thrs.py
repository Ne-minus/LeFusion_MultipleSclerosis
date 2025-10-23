import os
import random
import torch as th
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.ndimage import binary_fill_holes, gaussian_filter, label


def process_3d_mask_threshold_sweep(
    mask_3d,
    output_dir="threshold_sweep",
    sigma=2,
    base_threshold=0.8,
    volume_min=300,
    volume_max=1200,
    min_component_volume=75
):


    os.makedirs(output_dir, exist_ok=True)
    print("sum:", th.sum(mask_3d))
    print("PARAMS:", sigma, base_threshold)

    device = mask_3d.device
    mask_3d = mask_3d.cpu().numpy()[0]
    filled_mask = binary_fill_holes(mask_3d)
    smoothed_mask = gaussian_filter(filled_mask.astype(float), sigma=sigma)

    thresholds, volumes, masks = [], [], []

    for t in np.arange(0.7, 0.85, 0.01):
        smoothed_mask_binary = smoothed_mask > t

        labeled_mask, num_features = label(smoothed_mask_binary)
        if num_features > 0:

            boundary_labels = set()
            boundary_labels.update(np.unique(labeled_mask[0, :, :]))
            boundary_labels.update(np.unique(labeled_mask[-1, :, :]))
            boundary_labels.update(np.unique(labeled_mask[:, 0, :]))
            boundary_labels.update(np.unique(labeled_mask[:, -1, :]))
            boundary_labels.update(np.unique(labeled_mask[:, :, 0]))
            boundary_labels.update(np.unique(labeled_mask[:, :, -1]))
            boundary_labels.discard(0)

            mask_clean = labeled_mask.copy()
            for b in boundary_labels:
                mask_clean[mask_clean == b] = 0


            component_sizes = np.bincount(mask_clean.ravel())
            small_labels = np.where(component_sizes < min_component_volume)[0]
            for lbl in small_labels:
                if lbl != 0:  
                    mask_clean[mask_clean == lbl] = 0

            mask_clean = mask_clean > 0
        else:
            mask_clean = smoothed_mask_binary

        mask_tensor = th.from_numpy(mask_clean).float().to(device)
        volume_voxels = int(th.sum(mask_tensor).item())

        thresholds.append(t)
        volumes.append(volume_voxels)
        masks.append(mask_clean)

        print(f"[thr={t:.2f}] voxels={volume_voxels}")

 
    valid_indices = [i for i, v in enumerate(volumes) if volume_min <= v <= volume_max]

    if not valid_indices:
        print(f"No maskls in a chosen range [{volume_min}, {volume_max}]")
        return None

    chosen_idx = random.choice(valid_indices)
    chosen_thr = thresholds[chosen_idx]
    chosen_volume = volumes[chosen_idx]
    chosen_mask = masks[chosen_idx]

    nii_name = f"mask_thr_{chosen_thr:.2f}_vox{chosen_volume}.nii.gz"
    nii_path = os.path.join(output_dir, nii_name)
    nib.save(nib.Nifti1Image(chosen_mask.astype(np.uint8), np.eye(4)), nii_path)

    print(f"\n✅ Выбрана и сохранена маска: {nii_path}")
    print(f"   Порог = {chosen_thr:.2f}, объём = {chosen_volume} вокселей")

    return nii_path, chosen_thr, chosen_volume


if __name__ == "__main__":
    img = nib.load("/workspace/LeFusion/DiffMask/MsLesSeg/P35_T1_MASK_lesion65_GenMask.nii.gz")
    mask_data = img.get_fdata()
    mask_tensor = th.from_numpy(mask_data).unsqueeze(0).float().to("cpu")

    process_3d_mask_threshold_sweep(mask_tensor, output_dir="threshold_results_granular_custom_spheres_new_model")
