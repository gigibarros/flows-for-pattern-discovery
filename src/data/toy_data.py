import random
import torch
import math
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def generate_words(num_words, num_neurons, overlap):
    """
    num_words   : number of words
    num_neurons : total number of neurons
    base_size   : average block size per word
    overlap     : number of neurons to overlap between neighbors
    jitter      : how much to randomly vary block size
    """

    words = torch.zeros(num_words, num_neurons, dtype=torch.float16)
    word_size = math.ceil((num_neurons - overlap) / num_words) + overlap  # ensure all neurons covered

    start = 0
    for k in range(num_words):
        end = min(start + word_size, num_neurons)
        words[k, start:end] = torch.rand(end - start)

        start = max(0, end - overlap)  # shift start forward, with some overlap

    return words

def generate_alpha(num_timepoints, num_words, padding=11):
    """
    num_timepoints : number of timepoints
    num_words      : number of words
    padding        : kernel size for temporal smoothing
    """

    alpha = torch.rand(num_timepoints, num_words, dtype=torch.float16)  # shape : (T, K)

    # smooth with 1D conv.
    kernel = torch.ones(1, 1, padding) / padding
    alpha = alpha.T.unsqueeze(1)  # shape : (K, 1, T)
    alpha = torch.nn.functional.conv1d(alpha, kernel, padding=padding//2)
    alpha = alpha.squeeze(1).T  # shape : (T, K)

    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)  # normalize

    return alpha

class ToyDataset(Dataset):
    def __init__(self, num_samples):
        self.num_samples = num_samples
        self.samples = []
        self.ys = []

    def generate_x(self, y):
        N = 100_000              # num neurons
        T = 600                  # num timesteps
        K = 100                  # num words
        overlap = 0.2 * (N / K)  # overlap between neighboring words

        w = generate_words(K, N, overlap)
        alpha = generate_alpha(T, K)
        x = alpha @ w  # shape: (T, N)

        return x
