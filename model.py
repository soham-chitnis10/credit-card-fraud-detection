import torch.nn as nn


class CreditCardFraudDetector(nn.Module):
    def __init__(self, input_size, hidden_size=256):
        super(CreditCardFraudDetector, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Linear(hidden_size // 2, 2),  # Output layer for binary classification
        )

    def forward(self, x):
        return self.model(x)
