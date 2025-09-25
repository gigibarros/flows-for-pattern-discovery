import random
import torch
import math
import yaml
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def generate_words(num_words, num_neurons, overlap, diff_y, scale=2):
    """
    num_words   : number of words
    num_neurons : total number of neurons
    overlap     : number of neurons to overlap between neighboring words
    diff_y0     : indices of words conditioned on y = 0
    diff_y1     : indices of words conditioned on y = 1
    scale       : scaling factor
    """

    words = torch.zeros(num_words, num_neurons, dtype=torch.float16)  # shape : (K, N)
    word_size = math.ceil((num_neurons - overlap) / num_words) + overlap  # ensure all neurons covered

    start = 0
    for k in range(num_words):
        end = min(start + word_size, num_neurons)
        words[k, start:end] = torch.rand(end - start)

        start = max(0, end - overlap)  # shift start forward, with some overlap

    words[diff_y, :] *= scale  # scale words affected by y

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
        self.num_neurons = 100_000
        self.num_words = 100
        self.num_timesteps = 600

        # pre-select words to change based on y
        all_words = list(range(self.num_words))
        num_changed = 0.1 * all_words
        self.diff_y0 = random.sample(all_words, k=num_changed)
        self.diff_y1 = random.sample(all_words, k=num_changed)

        self.samples = []
        self.ys = []

        for _ in range(num_samples):
            y = random.randint(0, 1)
            x = self.generate_x(y)

            self.ys.append(y)
            self.x.append(x)

    def generate_x(self, y):
        """
        Generate toy neural data.
        """

        N = self.num_neurons              # num neurons
        T = self.num_timesteps            # num timesteps
        K = self.num_words                 # num words
        overlap = int(0.2 * (N / K))  # overlap between neighboring words

        if y == 0:
            w = generate_words(K, N, overlap, self.diff_y0)
        if y == 1:
            w = generate_words(K, N, overlap, self.diff_y1)

        alpha = generate_alpha(T, K)
        x = alpha @ w  # shape: (T, N)

        return x
