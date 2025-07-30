from torch import nn


class CreditCardFraudDetector(nn.Module):
    """
    A simple MLP model for credit card fraud detection.
    """

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
        """
        Forward pass through the model.
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        """
        return self.model(x)
