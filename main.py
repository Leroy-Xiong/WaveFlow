"""Sampling entry point for the released WaveFlow checkpoints.

The paper's sampler is driven through the ``waveflow`` package, which fills in
the checkpoint path, the data directory and the WeatherBench2 forcing:

    python -m waveflow sample --model hk --timestamps 20211013T00

Calling this file directly is the low-level form and expects every config
override to be supplied on the command line.
"""
import argparse

from modules.config import apply_overrides, load_config
from modules.training import (
    TrainerSampler,
    cleanup_ddp,
    setup_ddp,
    validate_checkpoint_for_sample_mode,
)


def main():
    parser = argparse.ArgumentParser(description="Sample from a WaveFlow checkpoint")
    parser.add_argument("--config", default="./configs/release/hk_4var.yaml",
                        help="path to a release config")
    parser.add_argument("-o", "--override", action="append", default=[],
                        help="override config values: section.key=value")
    args = parser.parse_args()
    args.mode = "sample"          # shared code path with the training entry point

    setup_ddp()
    config = load_config(args.config)
    if args.override:
        config = apply_overrides(config, args.override)

    validate_checkpoint_for_sample_mode(args, config)
    TrainerSampler(args.config, mode="sample", config=config).sample()
    cleanup_ddp()


if __name__ == "__main__":
    main()
