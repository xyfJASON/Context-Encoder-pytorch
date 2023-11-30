import tqdm
import argparse
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import Subset, DataLoader

import accelerate

import models
import metrics
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
        '--n_eval', type=int, default=10000,
        help='Number of images to evaluate on',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=256,
        help='Batch size on each process',
    )
    return parser


@torch.no_grad()
def evaluate():
    metric_l1 = metrics.L1(reduction='none').to(device)
    metric_psnr = metrics.PSNR(reduction='none', data_range=1.).to(device)
    metric_ssim = metrics.SSIM(size_average=False, data_range=1.).to(device)
    metric_meter = metrics.KeyValueAverageMeter(keys=['l1', 'psnr', 'ssim'])

    for gt_img in tqdm.tqdm(
            test_loader, desc='Evaluating', disable=not accelerator.is_main_process,
    ):
        gt_img = gt_img.float()
        cor_img = gt_img.clone()
        cor_img[:, :, 32:96, 32:96] = 0.
        inpainted_img = G(cor_img)

        gt_img = image_norm_to_float(gt_img)
        inpainted_img = image_norm_to_float(inpainted_img)
        l1 = metric_l1(inpainted_img, gt_img[:, :, 32:96, 32:96])
        psnr = metric_psnr(inpainted_img, gt_img[:, :, 32:96, 32:96])
        ssim = metric_ssim(inpainted_img, gt_img[:, :, 32:96, 32:96])
        l1, psnr, ssim = accelerator.gather_for_metrics((l1, psnr, ssim))
        metric_meter.update(dict(
            l1=l1.mean(),
            psnr=psnr.mean(),
            ssim=ssim.mean(),
        ), l1.shape[0])
    if accelerator.is_main_process:
        for k, v in metric_meter.avg.items():
            logger.info(f'{k}: {v.item()}')


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
    test_set = Subset(test_set, torch.arange(min(args.n_eval, len(test_set))))
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
    logger.info('Start evaluating...')
    evaluate()
    logger.info('End of evaluation')
