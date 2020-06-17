"""
Model evaluation script

python -m evaluate --config path_to_config_yaml
"""

# %%
from collections import defaultdict
import argparse
import json
import os
import yaml
import copy

import torch

from config.config_utils import get_object_instance
from data.link_data import makelinks
from data.data_utils import masks_to_colorimg
from matplotlib import pyplot as plt


# %%
def validate(
    dataloader,
    model,
    loss_metric,
    device,
    saving_metric=None,
    best_metric_type=None,
    is_better=None,
    plot_validate=None,
    val_batch_idx=None,
    val_img_idx=None,
):
    all_losses_and_metrics = defaultdict(list)
    num_examples = 0
    # worst_batch = None
    # worst_metric = None

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_batch_hat = model(x_batch)

        batch_size = y_batch.size(0)
        num_examples += batch_size
        losses_and_metrics = loss_metric(y_batch_hat, y_batch)
        for key, value in losses_and_metrics.items():
            all_losses_and_metrics[key].append(value.item() * batch_size)

            x_check, y_check = dataloader.dataset[
                batch_size * val_batch_idx + val_img_idx
            ]
        x_check = torch.from_numpy(x_check).unsqueeze(0).to(device)
        y_check = torch.from_numpy(y_check).unsqueeze(0).to(device)
        pred_check = model(x_check)
        plot_validate(batch=x_check, prediction=pred_check, target=y_check)

    losses_and_metrics_batched = copy.deepcopy(
        all_losses_and_metrics
    )  # non-reduced

    for key, value in all_losses_and_metrics.items():
        all_losses_and_metrics[key] = (
            sum(all_losses_and_metrics[key]) / num_examples
        )
    return all_losses_and_metrics, losses_and_metrics_batched


# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    loader_to_eval = config["_LOADER_TO_EVAL"]
    dataloader_config = config[loader_to_eval]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]
    saved_checkpoint = config["_MODEL_CHECKPOINT"]

    # TODO: support new checkpoint types
    model = get_object_instance(model_config)()
    model.load_state_dict(torch.load(saved_checkpoint))

    dataloader = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()
    all_losses_and_metrics = validate(dataloader, model, loss_metric, device)

    os.makedirs(results_path, exist_ok=True)
    with open("{}/val_results.json".format(results_path), "w") as fp:
        print(all_losses_and_metrics)
        json.dump(all_losses_and_metrics, fp, indent=4)

    data_iter = iter(dataloader)
    inputs, labels = next(data_iter)
    inputs = inputs.to(device)
    preds = model(inputs)
    inputs = inputs.cpu()
    preds = preds.cpu()

    plot_figure_from_batch(inputs, preds)


def plot_figure_from_batch(inputs, preds, target=None, idx=0):

    f, axarr = plt.subplots(1, 2)
    axarr[0].imshow(inputs[idx][1], cmap="gray")
    axarr[1].imshow(inputs[idx][1], cmap="gray")  # background for mask
    axarr[1].imshow(masks_to_colorimg(preds[idx]), alpha=0.5)

    return f


# %%
def quick_check(run_makelinks=False):
    if run_makelinks:
        makelinks()
    path = "./config/examples/eval_example.yaml"
    with open(path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    evaluate(config)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="YAML config path", type=str, required=True
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    makelinks()
    evaluate(config)
