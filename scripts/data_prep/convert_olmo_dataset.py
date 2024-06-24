# Copyright 2022 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

"""Streaming dataset conversion scripts for C4 and The Pile."""

import os
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Iterable, Optional, Union

import torch
from cached_path import cached_path
from numpy.typing import NDArray
from olmo import TrainConfig
from olmo.data import build_train_dataloader
from olmo.util import clean_opt
from streaming import MDSWriter
from torch.utils.data import DataLoader
from tqdm import tqdm


class ConcatMode(Enum):
    NO_CONCAT = "NO_CONCAT"
    CONCAT_TOKENS = "CONCAT_TOKENS"


def parse_args() -> Namespace:
    """Parse commandline arguments."""
    parser = ArgumentParser(
        description="Convert OLMo tokenized dataset into MDS format, optionally concatenating",
    )
    parser.add_argument("config_path", type=str)
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "train_small"],
    )
    parser.add_argument("--out_root", type=str, required=True)
    parser.add_argument("--compression", type=str, default=None)

    parser.add_argument("--no_wrap", default=False, action="store_true")
    parser.add_argument("--num_workers", type=int, required=False, default=None)

    parsed = parser.parse_args()

    if (
        os.path.isdir(parsed.out_root)
        and len(
            set(os.listdir(parsed.out_root)).intersection(set(parsed.splits)),
        )
        > 0
    ):
        raise ValueError(
            f"--out_root={parsed.out_root} contains {os.listdir(parsed.out_root)} which cannot overlap with the requested splits {parsed.splits}.",
        )

    return parsed


def generate_samples(
    loader: DataLoader,
    truncate_num_samples: Optional[int] = None,
) -> Iterable[Union[Dict[str, bytes], Dict[str, NDArray]]]:
    """Generator over samples of a dataloader.

    Args:
       loader (DataLoader): A dataloader emitting batches like {key: [sample0_bytes, sample1_bytes, sample2_bytes, ...]}
       truncate_num_samples (Optional[int]): An optional # of samples to stop at.

    Yields:
        Sample dicts.
    """
    n_samples = 0
    for batch in loader:
        keys = list(batch.keys())
        current_bs = len(batch[keys[0]])
        for idx in range(current_bs):
            if truncate_num_samples is not None and n_samples == truncate_num_samples:
                return
            n_samples += 1
            yield {
                k: v[idx].numpy() if isinstance(v[idx], torch.Tensor) else v[idx]
                for k, v in batch.items()
            }


@dataclass
class DataSplitConstants:
    hf_split: str
    folder_split: str
    raw_samples: Optional[int]
    truncated_samples: Union[int, None]


@dataclass
class DatasetConstants:
    chars_per_sample: int
    chars_per_token: int
    splits: Dict[str, DataSplitConstants] = field(default_factory=dict)

    def __iter__(self):
        for v in self.splits.values():
            yield v


dolmaconstants = DatasetConstants(
    chars_per_sample=6212,  # Computed over validation set
    chars_per_token=4,  # OpenAI estimate
)
dolmaconstants.splits["train"] = DataSplitConstants(
    hf_split="train",
    folder_split="train",
    raw_samples=1_000_000_000,
    truncated_samples=1_000_000_000,
)
dolmaconstants.splits["train_medium"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_medium",
    raw_samples=40_960_000,
    truncated_samples=40_960_000,
)
dolmaconstants.splits["train_small"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_small",
    raw_samples=100_000,
    truncated_samples=100_000,
)
dolmaconstants.splits["train_debug"] = DataSplitConstants(
    hf_split="train",
    folder_split="train_debug",
    raw_samples=100,
    truncated_samples=100,
)


def main(args: Namespace) -> None:
    """Main: create OLMo streaming dataset.

    Args:
        args (Namespace): Commandline arguments.
    """

    dataset_constants = dolmaconstants
    columns = {"tokens": "ndarray:int32"}
    cfg = TrainConfig.load(
        cached_path(args.config_path),
        overrides=[clean_opt("--evaluators=[]"), clean_opt("--save_overwrite")],
    )

    for split_name in args.splits:
        try:
            split = dataset_constants.splits[split_name]
        except KeyError:
            raise KeyError(f"Constants not defined for split {split_name}.")
        folder_split = split.folder_split
        expected_num_samples = split.raw_samples
        truncate_num_samples = split.truncated_samples
        # Only generate the splits requested
        if folder_split not in args.splits:
            continue

        # Get samples
        cfg.device_train_batch_size = cfg.global_train_batch_size
        cfg.data.num_workers = 0
        loader = build_train_dataloader(cfg, world_size=1, rank=0, fs_local_rank=0)
        samples = generate_samples(
            loader,
            truncate_num_samples=truncate_num_samples,
        )

        if expected_num_samples is not None:
            denominator = truncate_num_samples
        else:
            denominator = None

        # Write samples
        print(f"Converting {folder_split} to MDS format...")
        print(
            f"Note: the progress bar is based on the dataset length before tokenization, and may finish at a value before 100%.",
        )
        with MDSWriter(
            columns=columns,
            out=os.path.join(args.out_root, folder_split),
            compression=args.compression,
        ) as out:
            if denominator is not None:
                for sample in tqdm(
                    samples,
                    desc=folder_split,
                    total=denominator,
                ):
                    out.write(sample)
            else:
                for sample in tqdm(samples, desc=folder_split):
                    out.write(sample)


if __name__ == "__main__":
    main(parse_args())
