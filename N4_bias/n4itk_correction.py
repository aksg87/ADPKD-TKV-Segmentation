import glob
import os
from pathlib import Path

import pydicom
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import SimpleITK as sitk

from data.data_config import labeled_dirs, dataroot

# %%


# %%


def n4itk(pixel_data):
    """ performs n4 bias normalization on MRIs

    Args:
        pixel_data (numpy): original MRI data

    Returns:
        tuple (numpy, numpy): original, corrected
    """
    original_img = pixel_data.copy()
    pixel_data = np.interp(
        pixel_data, (pixel_data.min(), pixel_data.max()), (0, 255)
    )
    mr_img = sitk.GetImageFromArray(pixel_data)
    mask_img = sitk.OtsuThreshold(mr_img, 0, 1, 200)

    # Convert to sitkFloat32
    mr_img = sitk.Cast(mr_img, sitk.sitkFloat32)
    # N4 bias field correction
    num_fitting_levels = 4
    num_iters = 200
    try:
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations(
            [num_iters] * num_fitting_levels
        )
        corrected_img = corrector.Execute(mr_img, mask_img)
        corrected_img = sitk.GetArrayFromImage(corrected_img)

        (
            corrected_img[corrected_img < 0],
            corrected_img[corrected_img > 255],
        ) = (0, 255)
        corrected_img = corrected_img.astype(np.uint8)
        return original_img, corrected_img

    except (RuntimeError, TypeError, NameError):
        print("*** RuntimeError ***")
        return original_img, None


def get_dicom_paths(dcm_path):

    dcms = glob.glob(dcm_path + "/**/*.dcm", recursive=True)

    return dcms


def get_target_path(dcm_paths, target_root=f"{dataroot}N4_corrected"):
    """converts dicom paths to numpy target paths where N4 corrected images will stored

    Args:
        dcm_paths (list): dicom paths of uncorrected imgs
        target_root (string, optional): target dir for saved n4 imgs. Defaults to f"{dataroot}N4_corrected".

    Returns:
        list: target paths of n4 imgs
    """
    target_paths = []
    for dcm in dcm_paths:
        target = dcm.replace(os.path.dirname(dcm), target_root)
        target_dir = os.path.dirname(dcm)
        target_paths.append(target)
        os.makedirs(target_dir, exist_ok=True)
    return target_paths


def save_numpy_file(target_path, np_data):
    target_path = target_path.replace(".dcm", "")
    np.save(target_path, np_data)


def generate_n4correction(dcms, target_paths):

    for index, dcm in tqdm(enumerate(dcms)):
        inputImage = sitk.ReadImage(dcm)

        ds = pydicom.dcmread(dcm)
        pixel_data = ds.pixel_array
        original_img, corrected_img = n4itk(pixel_data)
        save_numpy_file(target_paths[index], corrected_img)


def get_n4_numpy(x):

    if isinstance(x, str):
        x = Path(x)

    parts = list(x.parts)

    for i, p in enumerate(parts):
        if p == "pkd-download":
            parts.insert(i + 1, "data_N4_corrected")
            break

    parts[-1] = parts[-1].replace(".dcm", ".npy")
    parts = tuple(parts)

    y = Path(*parts)
    if not y.exists():
        print("parts do not exist -> path: {}".format(parts))
        return None

    return y


def display_npy(npy):

    img = np.load(npy).astype(dtype=np.float32)
    plt.imshow(img, cmap="gray")


def main():

    for labeled_dir in tqdm(labeled_dirs):
        dcm_paths = get_dicom_paths(labeled_dir)
        target_paths = get_target_path(dcm_paths,)
        generate_n4correction(dcm_paths, target_paths)


main()
