"""Five-block CNN for windowed biosignals (Atzori et al. 2016), ported to EEG epochs.

Every window is treated as a one-channel image of shape (time, electrodes). Tensor shapes are
(batch, feature maps, time, electrodes), with B = batch, C = n_channels, T = window length in samples,
K = n_classes:

  input   (B, C, T)    -> transposed and given a map axis -> (B, 1, T, C)
  1. Conv, 32 filters spanning all electrodes (1 x C)                -> ReLU            (B, 32, T, 1)
  2. Conv, 32 filters 3 x 3                                          -> ReLU -> AvgPool (B, 32, T', 1)
  3. Conv, 64 filters 5 x 5                                          -> ReLU -> AvgPool (B, 64, T'', 1)
  4. Conv, 64 filters (time_kernel x 1): 5 x 1 (Ottobock) or 9 x 1 (Delsys) in the paper
                                                                     -> ReLU            (B, 64, T'', 1)
  5. Conv, K filters 1 x 1                                                              (B, K, T'', 1)
  output  mean over time -> logits (B, K) -> softmax loss

  
Choices the description leaves open:
  - Block 2,3,4 use convolution padding to keep size
  - Pooling uses stride 1 with padding, so the time axis keeps its length (`pool_stride` changes this according to T' = (T - 1) // pool_stride + 1).
  - The 1 x 1 convolution gives one score per time step; they are averaged over time to get one
    score per class per window. The softmax is applied by the loss (`nn.CrossEntropyLoss`).
"""

import torch
from torch import nn


class TCNN(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        time_kernel: int = 5,
        pool_stride: int = 1,
        n_filters: tuple[int, int, int, int] = (32, 32, 64, 64),
    ) -> None:
        super().__init__()
        if time_kernel % 2 == 0:
            raise ValueError("time_kernel must be odd so that 'same' padding keeps the time length")
        f1, f2, f3, f4 = n_filters

        self.block1 = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, n_channels)),
            nn.ReLU(),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(f1, f2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=pool_stride, padding=1, count_include_pad=False),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(f2, f3, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=pool_stride, padding=1, count_include_pad=False),
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(f3, f4, kernel_size=(time_kernel, 1), padding=(time_kernel // 2, 0)),
            nn.ReLU(),
        )
        self.block5 = nn.Conv2d(f4, n_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, channels, time) -> logits (batch, n_classes)."""
        x = x.transpose(1, 2).unsqueeze(1)  # (B, 1, T, C)
        x = self.block1(x)                  # (B, 32, T, 1)    electrode axis collapsed to width 1
        x = self.block2(x)                  # (B, 32, T', 1)
        x = self.block3(x)                  # (B, 64, T'', 1)
        x = self.block4(x)                  # (B, 64, T'', 1)  padding keeps the time length
        x = self.block5(x)                  # (B, K, T'', 1)   one score per class and time step
        return x.mean(dim=(2, 3))           # (B, K)           one scor per class


if __name__ == "__main__":
    model = TCNN(n_channels=18, n_classes=3)
    batch = torch.randn(4, 18, 256)  # 4 windows of 1 s at 256 Hz
    print(model(batch).shape, sum(p.numel() for p in model.parameters()), "parameters")
