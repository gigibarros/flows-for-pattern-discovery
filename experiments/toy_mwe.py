from pattern_flows.data.toy_data import ToyDataset
import matplotlib.pyplot as plt
import numpy as np
import torch

if __name__ == "__main__":

    dataset = ToyDataset(num_samples=2, num_neurons=100, num_words=10, num_timesteps=60)

    # generate two samples each for y = 0 and y = 1
    x0_1 = dataset.generate_x(y=0).detach().cpu().numpy()
    x0_2 = dataset.generate_x(y=0).detach().cpu().numpy()
    x1_1 = dataset.generate_x(y=1).detach().cpu().numpy()
    x1_2 = dataset.generate_x(y=1).detach().cpu().numpy()

    # set consistent color scale across all plots
    vmin = min(x0_1.min(), x0_2.min(), x1_1.min(), x1_2.min())
    vmax = max(x0_1.max(), x0_2.max(), x1_1.max(), x1_2.max())

    f = plt.figure(figsize=(12, 8))

    ax = f.add_subplot(2, 2, 1)
    im = ax.imshow(x0_1.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron (sampled)")
    ax.set_xlabel("time")
    ax.set_title("y = 0 (sample 1)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 2)
    im = ax.imshow(x0_2.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron (sampled)")
    ax.set_xlabel("time")
    ax.set_title("y = 0 (sample 2)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 3)
    im = ax.imshow(x1_1.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron (sampled)")
    ax.set_xlabel("time")
    ax.set_title("y = 1 (sample 1)")
    plt.colorbar(im, ax=ax, label="activity")

    ax = f.add_subplot(2, 2, 4)
    im = ax.imshow(x1_2.T, aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax)
    ax.set_ylabel("neuron (sampled)")
    ax.set_xlabel("time")
    ax.set_title("y = 1 (sample 2)")
    plt.colorbar(im, ax=ax, label="activity")

    plt.tight_layout()
    plt.show()
