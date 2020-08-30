from pathlib import Path

# %%
# adapt the file for actual data locations
# after modification, rename and place this file as
# adpkd_segmentation.data.data_config.py

dataroot = Path("/root/download/")

labeled_dirs = [
    dataroot / "training-data-01-60MR",
    dataroot / "training_data-61-110MR_AX_SSFSE_ABD_PEL_50",
]
unlabeled_dirs = [dataroot / "unlabelled_data"]

# modify for custom symbolic link locations
LABELED = None
UNLABELED = None

# default location
script_location = Path(__file__).resolve()

if LABELED is None:
    LABELED = script_location.parent.parent.parent / "labeled"
if UNLABELED is None:
    UNLABELED = script_location.parent.parent.parent / "unlabeled"
