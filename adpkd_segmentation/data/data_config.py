from pathlib import Path

# %%
dataroot = Path("/mnt/Samsung SSD/vast_ai_mirror/")

labeled_dirs = [
    dataroot / "training-data-01-60MR",
    dataroot / "training_data-61-110MR_AX_SSFSE_ABD_PEL_50",
]

unlabeled_dirs = [dataroot / "unlabelled_data"]
