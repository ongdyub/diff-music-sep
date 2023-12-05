import torch
from demucs.htdemucs import HTDemucs
import torch.nn as nn

class HTDemucs(torch.nn.Module):
    def __init__(self,
                 sources,
                 audio_channels,
                 channels):
        super().__init__()
        
        self.model = HTDemucs(source_names,
                              audio_channels=audio_channels,
                              channels=channels
                              )
        self.out_head = nn.Conv2d(5,1,1)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, time)
            time_cond: (batch,)
        Returns:
            x: (batch, channels, time) same size as input
        """
        x = self.model(x)
        return self.out_head(x.transpose(1,2)).squeeze(dim=1)