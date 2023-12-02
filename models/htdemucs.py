import torch
from demucs.htdemucs import HTDemucs
import torch.nn as nn

class HtDemucs(torch.nn.Module):
    def __init__(self,
                 source_names,
                 in_channels,
                 initial_layer_num):
        super().__init__()
        
        self.model = HTDemucs(source_names,
                              audio_channels=in_channels,
                              channels=initial_layer_num
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