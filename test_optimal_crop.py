"""
Script for determining optimal crop around kidneys
"""

# %%
import yaml

import matplotlib.pyplot as plt


from config.config_utils import get_object_instance
from data.link_data import makelinks
from data.data_utils import masks_to_colorimg

import numpy as np

# %%
makelinks()

# %%
def evaluate(config):

    dataloader_config = config["_VAL_DATALOADER_CONFIG"]
    dataloader = get_object_instance(dataloader_config)()

    return dataloader  # add return types for debugging/testing


# %%
path = "./config/examples/eval_example.yaml"

# %%
with open(path, "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# %%
dataloader = evaluate(config)

# %%
img_idx = 180

dataset = dataloader.dataset
x, y = dataset[img_idx]
print("Dataset Length: {}".format(len(dataset)))
print("image -> shape {},  dtype {}".format(x.shape, x.dtype))
print("mask -> shape {},  dtype {}".format(y.shape, y.dtype))

# %%
print("Image and Mask: \n")

dcm, mask = x[0, ...], y

f, (ax1, ax2) = plt.subplots(1, 2)
ax1.imshow(dcm, cmap="gray")
ax2.imshow(dcm, cmap="gray")
ax2.imshow(masks_to_colorimg(mask), alpha=0.5)
# %%
x, y, attribs = dataset.get_verbose(img_idx)
print("Image Attributes: \n\n{}".format(attribs))

# %%
# Check pixels after crop
data_iter = iter(dataloader)

sum = 0
for _, labels_batch in data_iter:
    sum += labels_batch.sum()

print(f"cumulative sum {sum}")
# %%
"""
Center Crop Size Experiment Results:

original image is 96 x 96

crop 58 -> cumulative sum is 68540
crop 64 -> cumulative sum is 70046
crop 72 -> cumulative sum is 70430
crop 84 -> cumulative sum is 70435
crop 96 -> cumulative sum is 70435
"""

# %%

# GRAPH OF RESULTS

plt.style.use("seaborn-whitegrid")
fig = plt.figure()
plt.title("optimal center crop for 96x96 input")
plt.xlabel("crop size")
plt.ylabel("num pixels")

x = [58, 64, 72, 84, 96]
y = [68540, 70046, 70430, 70435, 70435]

plt.plot(x, y)

# %%
