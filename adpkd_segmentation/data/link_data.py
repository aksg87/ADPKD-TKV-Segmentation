# %%
import os
import shutil

from adpkd_segmentation.data.data_utils import get_dcms_paths, get_y_Path

# define data sources in data_config.py
from adpkd_segmentation.data.data_config import (
    labeled_dirs,
    unlabeled_dirs,
    LABELED,
    UNLABELED,
)


# %%
def symlink_force(target, link_name):
    try:
        os.symlink(target, link_name)
    except Exception as e:
        print(e)


def mkdir_force(dir):
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)


# %%
def makelinks():

    mkdir_force(LABELED)
    mkdir_force(UNLABELED)

    for dcm in get_dcms_paths(labeled_dirs):
        mask = get_y_Path(dcm)
        if not mask.exists():
            raise Exception("Labeled dcm [{}] does not have mask.".format(dcm))

        symlink_force(dcm, os.path.join(LABELED, os.path.basename(dcm)))
        symlink_force(mask, os.path.join(LABELED, os.path.basename(mask)))

    for dcm in get_dcms_paths(unlabeled_dirs):
        mask = get_y_Path(dcm)

        if mask.exists():
            raise Exception("Unlabeled dcm [{}] contains a mask.".format(dcm))

        symlink_force(dcm, os.path.join(UNLABELED, os.path.basename(dcm)))
