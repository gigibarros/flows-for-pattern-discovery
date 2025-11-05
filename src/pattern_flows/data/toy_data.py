import random
import torch
import math
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

def generate_words(num_words, num_neurons, overlap=0):
    """
    Generate a set of words as contiguous blocks of activity across neurons,
    where each word contributes to a subset of neurons.
    
    num_words       : number of words
    num_time:points : number of timepoints
    overlap         : number of neurons overlapping between words
    """
    # words = torch.rand(num_words, num_neurons, dtype=torch.float32)  # shape : (K, N)
    # words *= torch.randint(0, 2, size=(num_words, num_neurons))  # randomly zero out weights

    if overlap < 0:
        overlap = 0

    word_size = math.ceil((num_neurons - overlap) / num_words) + overlap  # contiguous blocks

    words = torch.zeros(num_words, num_neurons)
    step = max(1, word_size - overlap)
    max_start = max(0, num_neurons - word_size)
    for k in range(num_words):
        start = min(k * step, max_start)
        end = start + word_size
        words[k, start:end] = torch.rand(end - start)  # fill with random values within that block

    return words

def generate_alpha(num_timepoints, num_words, peaks_per_word=5, scale=1.0, smooth=11):
    """
    Generate mixture-weight matrix with random Gaussian peaks, smoothed temporally.

    num_timepoints : number of timepoints
    num_words      : number of words
    peaks_per_word : number of peaks per word
    scale          :
    smooth         : size of smoothing kernel
    """

    t = torch.arange(num_timepoints).float()
    alpha = torch.zeros(num_timepoints, num_words)
    for k in range(num_words):
        for _ in range(peaks_per_word):
            mu = random.uniform(0, num_timepoints)            # peak center
            sigma = random.uniform(1.0, num_timepoints*0.05)  # peak width
            amp = torch.rand(1).item() * scale                # peak height
            bump = amp * torch.exp(-0.5 * ((t - mu) / sigma)**2)  # Gaussian peaks
            alpha[:, k] += bump
            
    # smooth using 1D convolution
    kernel = torch.ones((1, 1, smooth), dtype=torch.float32) / smooth
    alpha = alpha.T.unsqueeze(1)
    alpha = torch.nn.functional.conv1d(alpha, kernel, padding=smooth//2)
    alpha = alpha.squeeze(1).T
    alpha = alpha / (alpha.sum(dim=1, keepdim=True) + 1e-8)
    return alpha

class ToyData(Dataset):
    def __init__(self, num_samples, num_neurons, num_words, num_timesteps):
        self.num_samples = num_samples
        self.num_neurons = num_neurons
        self.num_words = num_words
        self.num_timesteps = num_timesteps
        self.overlap = int(num_neurons // (num_words * 2))  # arbitrary, may change later

        # Create words with subset of signal words for learnable class differences
        num_signal = max(1, self.num_words // 10)
        self.signal_words = random.sample(range(self.num_words), k=num_signal)

        base_words = generate_words(self.num_words, self.num_neurons, overlap=self.overlap)
        self.words_0 = base_words.clone()
        self.words_1 = base_words.clone()

        for k in self.signal_words:
            self.words_1[k] *= 2.0

        self.xs = []
        self.ys = []

        for _ in range(num_samples):
            y = random.randint(0, 1)
            x = self.generate_x(y)

            self.ys.append(y)
            self.xs.append(x)

    def __len__(self):
        return len(self.xs)


    def __getitem__(self, idx):
        x = self.xs[idx]
        y = self.ys[idx]
        return x, y

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
        # x = x ** 2   # emphasize peaks

        return x

def get_toy_dataset(config):
    num_samples   = config["data"]["num_samples"]
    num_neurons   = config["data"]["num_neurons"]
    num_words     = config["data"]["num_words"]
    num_timesteps = config["data"]["num_timesteps"]

    return ToyData(num_samples, num_neurons, num_words, num_timesteps)

def get_train_loader(config):
    num_samples   = config["data"]["num_samples"]
    num_neurons   = config["data"]["num_neurons"]
    num_words     = config["data"]["num_words"]
    num_timesteps = config["data"]["num_timesteps"]
    batch_size    = config["training"]["batch_size"]

    return DataLoader(dataset=ToyData(num_samples, num_neurons, num_words, num_timesteps), batch_size=batch_size, shuffle=True, drop_last=True)

def get_valid_loader(config):
    num_samples   = config["data"]["num_samples"]
    num_neurons   = config["data"]["num_neurons"]
    num_words     = config["data"]["num_words"]
    num_timesteps = config["data"]["num_timesteps"]
    batch_size    = config["training"]["batch_size"]

    return DataLoader(dataset=ToyData(num_samples, num_neurons, num_words, num_timesteps), batch_size=batch_size, shuffle=False, drop_last=True)
