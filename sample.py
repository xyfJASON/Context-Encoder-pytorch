import os
import tqdm
import argparse
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image
from torch.utils.data import Subset, DataLoader

import accelerate

import models
from utils.data import get_dataset
from utils.logger import get_logger
from utils.misc import image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--model_path', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--n_samples', type=int, default=100,
        help='Number of images to sample',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=256,
        help='Batch size on each process',
    )
    return parser


@torch.no_grad()
def sample():
    idx = 0
    for gt_img in tqdm.tqdm(
            test_loader, desc='Sampling', disable=not accelerator.is_main_process,
    ):
        gt_img = gt_img.float()
        cor_img = gt_img.clone()
        cor_img[:, :, 32:96, 32:96] = 0.
        inpainted_img = gt_img.clone()
        inpainted_img[:, :, 32:96, 32:96] = G(cor_img)

        cor_img, gt_img, inpainted_img = accelerator.gather_for_metrics(
            (cor_img, gt_img, inpainted_img),
        )
        cor_img = image_norm_to_float(cor_img)
        gt_img = image_norm_to_float(gt_img)
        inpainted_img = image_norm_to_float(inpainted_img)

        if accelerator.is_main_process:
            for i in range(len(cor_img)):
                save_image(
                    tensor=[gt_img[i], cor_img[i], inpainted_img[i]],
                    fp=os.path.join(args.save_dir, f'{idx}.png'), nrow=3,
                )
                idx += 1


if __name__ == '__main__':
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    test_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='test',
    )
    test_set = Subset(test_set, torch.arange(min(args.n_samples, len(test_set))))
    test_loader = DataLoader(
        dataset=test_set,
        shuffle=False,
        drop_last=False,
        batch_size=args.micro_batch,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of test set: {len(test_set)}')
    logger.info(f'Batch size per process: {args.micro_batch}')
    logger.info(f'Total batch size: {args.micro_batch * accelerator.num_processes}')

    # BUILD MODELS
    G = models.Generator()
    # LOAD MODEL WEIGHTS
    ckpt_model = torch.load(args.model_path, map_location='cpu')
    G.load_state_dict(ckpt_model['G'])
    logger.info(f'Successfully load G from {args.model_path}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    G, test_loader = accelerator.prepare(G, test_loader)  # type: ignore
    G.eval()

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
