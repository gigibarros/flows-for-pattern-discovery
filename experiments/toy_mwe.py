# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "pattern-flows",
# ]
#
# [tool.uv.sources]
# pattern-flows = { path = "../", editable = true }
# ///

from pattern_flows.data.toy_data import ToyDataset
import matplotlib.pyplot as plt
import numpy as np
import torch


if __name__ == "__main__":

    dataset = ToyDataset(num_samples=1, num_neurons=100, num_words=10, num_timesteps=60)

    x_0 = dataset.generate_x(y=0)
    x_1 = dataset.generate_x(y=1)

    if isinstance(x_0, torch.Tensor):
        x_0 = x_0.detach().cpu().numpy()
    if isinstance(x_1, torch.Tensor):
        x_1 = x_1.detach().cpu().numpy()

    f = plt.figure(figsize=(10, 4))
    f.add_subplot(1, 2, 1)
    plt.imshow(x_0.T, aspect="auto", cmap="viridis")  # transpose so neurons appear on the y-axis and time on x-axis
    plt.ylabel("neuron (sampled)")
    f.add_subplot(1, 2, 2)
    plt.imshow(x_1.T, aspect="auto", cmap="viridis")
    plt.xlabel("time")
    plt.colorbar(label="activity")
    plt.show()