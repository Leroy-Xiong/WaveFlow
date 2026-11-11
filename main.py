import argparse

from modules.training import (
    TrainerSampler,
    setup_ddp,
    cleanup_ddp,
    validate_checkpoint_for_sample_mode,
)
from modules.config import load_config, apply_overrides


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train or sample from Flow Diffusion model')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'sample'],
                        help='Mode: train or sample')
    parser.add_argument('--config', type=str, default='./configs/configs.yaml',
                        help='Path to config file (default: ./configs/configs.yaml)')
    parser.add_argument('-o', '--override', action='append', default=[],
                        help='Override config values: section.key=value (can be used multiple times)')
    args = parser.parse_args()

    # 1. Setup DDP Environment
    setup_ddp()

    # 2. Load config
    config_path = args.config
    config = load_config(config_path)

    # 2.5 Apply CLI overrides
    if args.override:
        config = apply_overrides(config, args.override)

    # 3. Validate checkpoint for sample mode
    validate_checkpoint_for_sample_mode(args, config)

    # 4. Initialize Trainer with mode
    # Note: batch_size in config is per GPU. Global batch size = batch_size * num_gpus
    trainer_sampler = TrainerSampler(config_path, mode=args.mode, config=config)

    # 5. Execute based on mode
    if args.mode == 'train':
        trainer_sampler.train()
    elif args.mode == 'sample':
        trainer_sampler.sample()

    # 6. Cleanup
    cleanup_ddp()


if __name__ == "__main__":
    main()
