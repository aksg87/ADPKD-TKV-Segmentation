"""
Notebook to explore inference and metrics on patients

The makelinks flag is needed only once to create symbolic links to the data.
"""

# %%
from collections import OrderedDict, defaultdict

import pandas as pd
import torch
import os
import yaml

# enable lib loading even if not installed as a pip package or in PYTHONPATH
# also convenient for relative paths in example config files
from pathlib import Path

os.chdir(Path(__file__).resolve().parent.parent)

from adpkd_segmentation.config.config_utils import get_object_instance  # noqa
from adpkd_segmentation.data.link_data import makelinks  # noqa
from adpkd_segmentation.data.data_utils import display_sample  # noqa
from adpkd_segmentation.utils.train_utils import load_model_data  # noqa
from adpkd_segmentation.utils.stats_utils import (  # noqa
    bland_altman_plot,
    scatter_plot,
)

from adpkd_segmentation.utils.losses import SigmoidBinarize # noqa


# %%
def calc_dcm_metrics(
    dataloader, model, device, binarize_func,
):
    """Calculates additional information for each of the dcm slices and
    stores the values in an updated dcm2attrib dictionary.
    Utilized for TKV and Kidney Dice calculations.

    Args:
        dataloader
        model
        device
        binarize_func

    Returns:
        dictionary: updated dcm2attrib dictionary
    """

    num_examples = 0
    dataset = dataloader.dataset
    updated_dcm2attribs = {}
    output_example_idx = (
        hasattr(dataloader.dataset, "output_idx")
        and dataloader.dataset.output_idx
    )

    for batch_idx, output in enumerate(dataloader):
        if output_example_idx:
            x_batch, y_batch, index = output
        else:
            x_batch, y_batch = output
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        batch_size = y_batch.size(0)
        num_examples += batch_size
        with torch.no_grad():
            y_batch_hat = model(x_batch)
            y_batch_hat_binary = binarize_func(y_batch_hat)
            start_idx = num_examples - batch_size
            end_idx = num_examples

            for inbatch_idx, dataset_idx in enumerate(
                range(start_idx, end_idx)
            ):
                # calculate TKV and TKV inputs for each dcm
                # TODO:
                # support 3 channel setups where ones could mean background
                # needs mask standardization to single channel
                _, dcm_path, attribs = dataset.get_verbose(dataset_idx)
                gt = y_batch[inbatch_idx]
                pred = y_batch_hat_binary[inbatch_idx]

                attribs["pred_kidney_pixels"] = torch.sum(pred > 0).item()
                attribs["ground_kidney_pixels"] = torch.sum(gt > 0).item()

                # TODO: Clean up method of accessing Resize transform
                trans_resize = dataloader.dataset.augmentation[0]
                attribs["transform_resize_dim"] = (
                    trans_resize.height,
                    trans_resize.width,
                )

                # scale factor takes into account the difference
                # between the original image/mask size and the size
                # after transform based resizing
                scale_factor = (attribs["dim"][0] ** 2) / (
                    attribs["transform_resize_dim"][0] ** 2
                )
                attribs["Vol_GT"] = (
                    scale_factor
                    * attribs["vox_vol"]
                    * attribs["ground_kidney_pixels"]
                )
                attribs["Vol_Pred"] = (
                    scale_factor
                    * attribs["vox_vol"]
                    * attribs["pred_kidney_pixels"]
                )

                # TODO: check dimensions, and add 'channels_first' doc
                # TODO: check if pow=2 is valid for 3d dice
                power = 2
                attribs["intersection"] = torch.sum(pred * gt, (1, 2))
                attribs["set_addition"] = torch.sum(
                    torch.pow(pred, power) + torch.pow(gt, power), (1, 2)
                )

                updated_dcm2attribs[dcm_path] = attribs

    return updated_dcm2attribs


# %%
def load_config(config_path, run_makelinks=False):
    """Reads config file and calculates additional dcm attributes such as
    slice volume. Returns a dictionary used for patient wide calculations
    such as TKV.
    """
    if run_makelinks:
        makelinks()
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model_config = config["_MODEL_CONFIG"]
    loader_to_eval = config["_LOADER_TO_EVAL"]
    split = config[loader_to_eval]["dataset"]["splitter_key"].lower()
    dataloader_config = config[loader_to_eval]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]
    checkpoint_format = config["_NEW_CKP_FORMAT"]

    model = get_object_instance(model_config)()
    if saved_checkpoint is not None:
        load_model_data(saved_checkpoint, model, new_format=checkpoint_format)

    dataloader = get_object_instance(dataloader_config)()

    # TODO: support other metrics as needed
    binarize_func = SigmoidBinarize(thresholds=[0.5])

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    return dataloader, model, device, binarize_func, split


# %%
def calculate_patient_metrics(updated_dcm2attrib, output=None):

    patient_MR_Metrics = defaultdict(float)
    Metric_data = OrderedDict()

    for key, value in updated_dcm2attrib.items():
        patient_MR = value["patient"] + value["MR"]
        patient_MR_Metrics[(patient_MR, "Vol_GT")] += value["Vol_GT"]
        patient_MR_Metrics[(patient_MR, "Vol_Pred")] += value["Vol_Pred"]
        patient_MR_Metrics[(patient_MR, "intersection")] += value[
            "intersection"
        ]
        patient_MR_Metrics[(patient_MR, "set_addition")] += value[
            "set_addition"
        ]

    for key, value in updated_dcm2attrib.items():
        patient_MR = value["patient"] + value["MR"]

        if patient_MR not in Metric_data:

            intersection = patient_MR_Metrics[(patient_MR, "intersection")]
            set_addition = patient_MR_Metrics[(patient_MR, "set_addition")]
            patient_dice = ((2.0 * intersection) / set_addition).item()

            summary = {
                "TKV_GT": patient_MR_Metrics[(patient_MR, "Vol_GT")],
                "TKV_Pred": patient_MR_Metrics[(patient_MR, "Vol_Pred")],
                "sequence": value["seq"],
                "split": split,
                "patient_dice": patient_dice,
            }

            Metric_data[patient_MR] = summary

    df = pd.DataFrame(Metric_data).transpose()

    if output is not None:
        df.to_csv(output)

    return df


# %%

# TKV on unfiltered + BA Plot ***
# pick another path for different experiments

# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_tkv/val/val.yaml" # noqa
# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_tkv/test/test.yaml" # noqa

# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_only/val/val.yaml"  # noqa
# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_only/test/test.yaml"  # noqa

# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_sd_tkv/val/val.yaml" # noqa
# path = "experiments/july14/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_sd_tkv/test/test.yaml" # noqa

# path = "experiments/july16/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_sd/val/val.yaml" # noqa

# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_batch_sd_pow_1/val/val.yaml" # noqa

# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_batch_sd_pow_2/val/val.yaml" # noqa

# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_batch_sd_pow_1/val/val.yaml" # noqa

# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_batch_sd_pow_2/val/val.yaml" # noqa
# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_bias_reduce/val/val.yaml" # noqa
# path = "experiments/july19/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_bce_bias_reduce_weighted/val/val.yaml" # noqa
# path = "experiments/july22/strat_seq_norm_b5_BN_bce_dice_simpler_albu_224_unet_double_batch_sd_pow_1_tkv/val/val.yaml" # noqa
# ---------------------
# path = "experiments/july17/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_tkv_loss/val/val.yaml" # noqa
# path = "experiments/july17/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_tkv_loss/test/test.yaml" # noqa

# path = "experiments/july16/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_tkv/val/val.yaml" # noqa
# path = "experiments/july16/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_tkv/test/test.yaml" # noqa

# path = "experiments/july17/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_only/val/val.yaml" # noqa

# path = "experiments/july17/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_sd/val/val.yaml" # noqa

# path = "experiments/july15/strat_seq_norm_b5_BN_even_more_albu_224_unet_double_bce_sd_tkv/val/val.yaml" # noqa

# path = "experiments/august18/stratified_run_example/val/val.yaml"  # noqa
# path = "experiments/august30/random_split_new_data_check/val/val.yaml"  # noqa
path = "experiments/august30/random_split_new_data_check/test/test.yaml" # noqa

# path = "./example_experiment/train_example_all_no_noise_patient_seq_norm_b5_BN/val/val.yaml"  # noqa
dataloader, model, device, binarize_func, split = load_config(config_path=path)

# %%
dcm2attrib = calc_dcm_metrics(dataloader, model, device, binarize_func)
patient_metric_data = calculate_patient_metrics(dcm2attrib)

pred = patient_metric_data["TKV_Pred"].to_numpy()
gt = patient_metric_data["TKV_GT"].to_numpy()
bland_altman_plot(pred, gt, percent=True, title="BA Plot: TKV all - % error")

# %%
relative_error = abs((gt - pred) / gt)
print(relative_error.mean())
print(relative_error.std(ddof=1))

# %%
# Patient Dice with TKV-GT on Scatter Plot ***
patient_dice = patient_metric_data["patient_dice"].to_numpy()
scatter_plot(patient_dice, gt, title="plot: Patient Dice by TKV")

# %%
# 3D dice
print(patient_dice.mean())
print(patient_dice.std(ddof=1))

# %%
# check some high error studies
studies = [
    "WC-ADPKD_AB9-001467-MR1",
    "WC-ADPKD_BD9-000403MR2",
    "WC-ADPKD_KC9-000836MR1",
]
for idx, label in enumerate(patient_metric_data.index):
    if label in studies:
        print(relative_error[idx], patient_dice[idx])

# %%
dataset = dataloader.dataset
path_to_index = {path: idx for idx, path in enumerate(dataset.dcm_paths)}
study_to_indices = defaultdict(list)
for dcm_path, attrib in dcm2attrib.items():
    study = attrib["patient"] + attrib["MR"]
    study_to_indices[study].append(path_to_index[dcm_path])

# %%
study = studies[1]
print(study)
for num, idx in enumerate(study_to_indices[study]):
    # assuming that dataset outputs all 3
    # all 3 image channels equal
    image, mask, index = dataset[idx]
    print(idx, index)  # should be the same
    display_sample((image[0], mask))
    if num > 44:
        break

# %%
# TKV on positive slices + BA Plot ***
dcm2attrib_pos = {}
for key, value in dcm2attrib.items():
    if value["ground_kidney_pixels"] > 0:
        dcm2attrib_pos[key] = value

patient_metric_data_pos = calculate_patient_metrics(dcm2attrib_pos)

pred_pos = patient_metric_data_pos["TKV_Pred"].to_numpy()
gt_pos = patient_metric_data_pos["TKV_GT"].to_numpy()
bland_altman_plot(
    pred_pos, gt_pos, percent=True, title="BA Plot: TKV positives - % error"
)

# %%
# TKV on negative slices + BA Plot ***
dcm2attrib_neg = {}
for key, value in dcm2attrib.items():
    if value["ground_kidney_pixels"] == 0:
        dcm2attrib_neg[key] = value

patient_metric_data_neg = calculate_patient_metrics(dcm2attrib_neg)

pred_neg = patient_metric_data_neg["TKV_Pred"].to_numpy()
gt_neg = patient_metric_data_neg["TKV_GT"].to_numpy()
bland_altman_plot(
    pred_neg, gt_neg, percent=False, title="BA Plot: TKV negatives"
)  # percent throws division by zero error


# can result in an error for some examples
# due to different lengths
# %%
scatter_plot(gt_neg - pred_neg, gt, title="TKV negatives")

# %%
