import os
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from skimage.measure import regionprops, label
from scipy.spatial.distance import cdist


mask_dir = "../patches/Mask"
out_dir = "results"
os.makedirs(out_dir, exist_ok=True)

results = []


for fname in sorted(os.listdir(mask_dir)):
    if fname.endswith(".nii.gz"):
        fpath = os.path.join(mask_dir, fname)
        img = nib.load(fpath)
        data = img.get_fdata().astype(np.uint8)

        labeled = label(data)
        props = regionprops(labeled)

        lesion_volumes = []
        lesion_radii = []

        for prop in props:
            coords = prop.coords
            lesion_volumes.append(prop.area)


            centroid = prop.centroid
            dists = cdist(coords, [centroid])
            radius = dists.max()
            lesion_radii.append(radius)

        total_volume = np.sum(lesion_volumes)
        n_lesions = len(lesion_volumes)

        for v, r in zip(lesion_volumes, lesion_radii):
            results.append({
                "mask": fname,
                "n_lesions": n_lesions,
                "lesion_volume": v,
                "total_volume": total_volume,
                "radius": r
            })


df = pd.DataFrame(results)

if len(df) == 0:

    exit()


summary = df.groupby("mask").agg({
    "n_lesions": "max",
    "lesion_volume": ["mean", "sum"],
    "radius": "mean"
}).reset_index()
summary.columns = ["mask", "n_lesions", "mean_lesion_volume", "total_volume", "mean_radius"]


df.to_csv(os.path.join(out_dir, "lesion_details.csv"), index=False)
summary.to_csv(os.path.join(out_dir, "lesion_summary.csv"), index=False)



import os
import matplotlib.pyplot as plt
import seaborn as sns

# Set Seaborn style
sns.set(style="whitegrid", context="talk", palette="deep")

# --- Boxplot: Number of lesions ---
plt.figure(figsize=(6, 5))
sns.boxplot(y=summary["n_lesions"], color="skyblue", width=0.4)
plt.title("Distribution of Lesion Counts per Mask")
plt.ylabel("Number of Lesions")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lesion_counts_boxplot.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Boxplot: Lesion Volume ---
plt.figure(figsize=(6, 5))
sns.boxplot(y=df["lesion_volume"], color="lightgreen", width=0.4)
plt.title("Distribution of Lesion Volumes")
plt.ylabel("Volume (voxels)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lesion_volume_boxplot.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Boxplot: Bounding Sphere Radius ---
plt.figure(figsize=(6, 5))
sns.boxplot(y=df["radius"], color="salmon", width=0.4)
plt.title("Distribution of Bounding Sphere Radii")
plt.ylabel("Radius (voxels)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "lesion_radius_boxplot.png"), dpi=300, bbox_inches="tight")
plt.close()

# --- Boxplot: Total Lesion Volume per Mask ---
plt.figure(figsize=(6, 5))
sns.boxplot(y=summary["total_volume"], color="plum", width=0.4)
plt.title("Distribution of Total Lesion Volume per Mask")
plt.ylabel("Total Volume (voxels)")
plt.tight_layout()
plt.savefig(os.path.join(out_dir, "total_volume_boxplot.png"), dpi=300, bbox_inches="tight")
plt.close()

print(f"📊 Figures saved in {out_dir}")
