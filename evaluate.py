"""
Model evaluation script

python -m evaluate --config path_to_config_yaml
"""

# %%
from collections import defaultdict
import argparse
import yaml
import pickle

import torch

from config.config_utils import get_object_instance
from data.link_data import makelinks

# %%
makelinks()

# %%
def evaluate(config):
    model_config = config["_MODEL_CONFIG"]
    dataloader_config = config["_DATALOADER_CONFIG"]
    loss_metric_config = config["_LOSSES_METRICS_CONFIG"]
    results_path = config["_RESULTS_PATH"]

    model = get_object_instance(model_config)()
    dataloaders = get_object_instance(dataloader_config)()
    loss_metric = get_object_instance(loss_metric_config)()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # TODO: add as a config option
    # train, val, test ordering
    dataloader_index = 1
    dataloader = dataloaders[dataloader_index]

    model = model.to(device)
    model.eval()
    all_losses_and_metrics = defaultdict(list)
    num_examples = 0

    for x_batch, y_batch in dataloader:
        x_batch = x_batch.to(device)
        y_batch = y_batch.to(device)
        with torch.no_grad():
            y_batch_hat = model(x_batch)

        batch_size = y_batch.size(0)
        num_examples += batch_size
        losses_and_metrics = loss_metric(y_batch_hat, y_batch)
        for key, value in losses_and_metrics.items():
            all_losses_and_metrics[key].append(value * batch_size)

    for key, value in all_losses_and_metrics.items():
        all_losses_and_metrics[key] = (
            torch.sum(torch.stack(all_losses_and_metrics[key])) / num_examples
        )

    # TODO: Pickle vs JSON for this dict
    with open("{}/val_results.pickle".format(results_path), "wb") as fp:
        pickle.dump(all_losses_and_metrics, fp)


# %%
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", help="YAML config path", type=str, required=True
    )

    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    evaluate(config)

# uncomment and run for a quick check
# %%
path = "./config/examples/eval_example.yaml"
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
evaluate(config)


# %%
