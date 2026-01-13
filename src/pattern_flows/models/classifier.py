import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)
    
def get_classifier(config):
    input_dim = config["data"]["input_dim"]
    
    return Classifier(input_dim)