import torch
import torchaudio
from hydra.utils import instantiate
from demucs.htdemucs import HTDemucs

class ScoreModelHTDemucs(torch.nn.Module):
    def __init__(
        self,
        backbone_args,
    ):
        super().__init__()
        self.backbone = instantiate(backbone_args)

    def forward(self, xt, time_cond, mix):
        """
        Args:
            x: (batch, channels, time)
            time_cond: (batch,)
        Returns:
            x: (batch, channels, time) same size as input
        """
        x = torch.cat((xt, mix), dim=1)
        x = self.backbone(x, time_cond)
        return x