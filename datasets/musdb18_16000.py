import logging
import os
from pathlib import Path
from typing import Optional, Union

import pytorch_lightning as pl
import torch
import torchaudio
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

torchaudio.set_audio_backend("sox_io")

log = logging.getLogger(__name__)

split_map = [
    "test",
    "val",
    "train"
]


class musdb_mix_16000(torch.utils.data.Dataset):
    def __init__(
        self,
        path: Union[str, Path],
        fs: Optional[int] = 16000,
        cut: Optional[str] = "max",
        split: Optional[str] = "train",
        max_len_s: Optional[float] = 3,
        max_n_samples: Optional[int] = None,
    ):
        super().__init__()

        self.base_folder = Path(to_absolute_path(str(path)))
        self.fs = int(fs)
        self.cut = cut
        self.max_len = int(self.fs * max_len_s) if max_len_s is not None else None

        # if fs not in [8000, 16000]:
        #     raise ValueError(
        #         f"The sampling frequency fs can be only 8000 or 16000 (passed {fs})"
        #     )


        if cut not in ["min", "max"]:
            raise ValueError(
                f"The cut parameter has to be 'min' or 'max' (passed {cut})"
            )

        if split not in split_map:
            raise ValueError(
                f"The split parameter must be 'train', 'val', or 'test' (passed {split})"
            )

        self.path = self.base_folder / split

        self.file_list = []
        for path_ in os.listdir(self.path):
            if os.path.isdir(self.path/ path_):
                self.file_list.append(self.path/path_)

        if max_n_samples is not None:
            self.file_list = self.file_list[:max_n_samples]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        rand_channel = torch.randint(0,2, size=(1,))
        mix = torchaudio.load(filename/'mixture.wav')[0][rand_channel]
        tgt = torch.cat(
            [torchaudio.load(filename / inst)[0][rand_channel] for inst in ['drums.wav', 'bass.wav', 'other.wav', 'vocals.wav']], dim=0
        )

        if self.max_len is not None and tgt.shape[-1] > self.max_len:
            # take a random cut of the right size
            p = int(torch.randint(0, tgt.shape[-1] - self.max_len, size=(1,)))
            tgt = tgt[..., p : p + self.max_len]
            mix = mix[..., p : p + self.max_len]

        return mix, tgt


def max_collator(batch):
    """
    Collate a bunch of multichannel signals based
    on the size of the longest sample. The samples are cut at the center
    """
    max_len = max([s[0].shape[-1] for s in batch])

    new_batch = []
    for bidx, row in enumerate(batch):
        new_row = []
        for eidx, el in enumerate(row):
            if isinstance(el, torch.Tensor):
                off = max_len - el.shape[-1]
                new_row.append(torch.nn.functional.pad(el, (off // 2, off - off // 2)))
        new_batch.append(tuple(new_row))

    return torch.utils.data.default_collate(new_batch)


class musdb_mix_16000_Module(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.cfg = config
        self.datasets = {}

    def setup(self, *args, **kwargs):
        for split in self.cfg.datamodule:
            self.datasets[split] = instantiate(self.cfg.datamodule[split].dataset)

    def _get_dataloader(self, split):
        return torch.utils.data.DataLoader(
            self.datasets[split],
            collate_fn=max_collator,
            **self.cfg.datamodule[split].dl_opts,
        )

    def train_dataloader(self):
        return self._get_dataloader("train")

    def val_dataloader(self):
        return self._get_dataloader("val")

    def test_dataloader(self):
        return self._get_dataloader("test")
