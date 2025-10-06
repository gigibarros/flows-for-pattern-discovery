import random
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def generate_words(num_words, num_neurons):
    """
    num_words   : number of words
    num_neurons : total number of neurons
    """

    words = torch.rand(num_words, num_neurons, dtype=torch.float32)  # shape : (K, N)
    words *= torch.randint(0, 2, size=(num_words, num_neurons))  # randomly zero out weights

    return words

def generate_alpha(num_timepoints, num_words, padding=11):
    """
    num_timepoints : number of timepoints
    num_words      : number of words
    padding        : kernel size for temporal smoothing
    """
    alpha = torch.rand(num_timepoints, num_words, dtype=torch.float32)  # shape : (T, K)
    # alpha = torch.ones(num_timepoints, num_words, dtype=torch.float32)  # debugging

    # smooth with 1D conv.
    kernel = torch.ones((1, 1, padding), dtype=torch.float32) / padding
    alpha = alpha.T.unsqueeze(1)  # shape : (K, 1, T)
    alpha = torch.nn.functional.conv1d(alpha, kernel, padding=padding//2)
    alpha = alpha.squeeze(1).T  # shape : (T, K)

    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)  # normalize

    return alpha

class ToyDataset(Dataset):
    def __init__(self, num_samples, num_neurons=100_000, num_words=100, num_timesteps=600):
        self.num_samples = num_samples
        self.num_neurons = num_neurons
        self.num_words = num_words
        self.num_timesteps = num_timesteps

        # pre-select words to change based on y
        num_changed = self.num_words // 10  # change 10% of words
        all_words = list(range(self.num_words))
        self.diff_y1 = random.sample(all_words, k=num_changed)
        
        # words to use for y = 0
        self.words_0 = generate_words(self.num_words, self.num_neurons)
        
        # words to use for y = 1
        self.words_1 = torch.tensor(self.words_0)
        self.words_1[:num_changed] = generate_words(self.num_words, self.num_neurons)[:num_changed]

        # self.xs = []
        # self.ys = []

        # for _ in range(num_samples):
        #     y = random.randint(0, 1) # will not work as expected! 1 is exclusive
        #     x = self.generate_x(y)

        #     self.ys.append(y)
        #     self.xs.append(x)

    def generate_x(self, y):
        """
        Generate toy neural data.
        """

        T = self.num_timesteps        # num timesteps
        K = self.num_words            # num words        

        if y == 0:
            w = self.words_0
        if y == 1:
            w = self.words_1

        alpha = generate_alpha(T, K)

        x = alpha @ w  # shape: (T, N)

        # normalize x
        min_x, max_x = x.min(), x.max()
        x = (x - min_x)/(max_x - min_x)

        return x

