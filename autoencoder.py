import numpy as np
import torch.nn as nn
import torch

class AutoEncoderNetwork(nn.Module):

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # Encoder: Input -> Bottleneck
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)  # Bottleneck layer
        )

        # Decoder: Bottleneck -> Output
        self.decoder = nn.Sequential(
            nn.Linear(output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim)
        )

    def project(self, x: torch.Tensor) -> torch.Tensor:
        """
        Maps input to the bottleneck representation
        """
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs forward pass: Encoder -> Decoder
        """
        bottleneck = self.encoder(x)
        reconstruction = self.decoder(bottleneck)
        return reconstruction


class AutoEncoder:
    def __init__(self, input_dim: int, projection_dim: int, learning_rate: float, iteration_count: int):
        """
        Initializes the Auto Encoder method
        """
        self.input_dim = input_dim
        self.projection_dim = projection_dim
        self.iteration_count = iteration_count

        # Autoencoder Model
        self.autoencoder_model = AutoEncoderNetwork(input_dim, projection_dim)

        # Optimizer and Loss
        self.optimizer = torch.optim.Adam(self.autoencoder_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()  # Reconstruction loss

    def fit(self, x: torch.Tensor) -> None:
        """
        Trains the autoencoder on the given dataset
        """
        self.autoencoder_model.train()  # Set model to training mode

        for epoch in range(self.iteration_count):
            # Zero gradients
            self.optimizer.zero_grad()

            # Forward pass
            reconstruction = self.autoencoder_model(x)

            # Compute loss
            loss = self.criterion(reconstruction, x)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Print loss every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.iteration_count}, Loss: {loss.item():.4f}")

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """
        Projects new data onto the bottleneck layer
        """
        self.autoencoder_model.eval()  # Set model to evaluation mode
        with torch.no_grad():
            return self.autoencoder_model.project(x)
