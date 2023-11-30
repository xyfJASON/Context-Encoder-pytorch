import os
import tqdm
import argparse
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

import models
import metrics
from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(broadcast_buffers=False)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(sort_keys=False),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    micro_batch = cfg.dataloader.micro_batch or batch_size_per_process
    train_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='train',
    )
    valid_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='valid',
        subset_ids=torch.arange(5000),
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    valid_loader = DataLoader(
        dataset=valid_set,
        shuffle=False,
        drop_last=False,
        batch_size=micro_batch,
        pin_memory=cfg.dataloader.pin_memory,
        num_workers=cfg.dataloader.num_workers,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Size of validation set: {len(valid_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')

    # BUILD MODELS AND OPTIMIZERS
    G = models.Generator()
    D = models.Discriminator()
    optimizer_G = optim.Adam(G.parameters(), lr=cfg.train.optim_g.lr, betas=cfg.train.optim_g.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=cfg.train.optim_d.lr, betas=cfg.train.optim_d.betas)
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load models
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        G.load_state_dict(ckpt_model['G'])
        D.load_state_dict(ckpt_model['D'])
        logger.info(f'Successfully load G and D from {ckpt_path}')
        # load optimizers
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer_G.load_state_dict(ckpt_optimizer['optimizer_G'])
        optimizer_D.load_state_dict(ckpt_optimizer['optimizer_D'])
        logger.info(f'Successfully load optimizers from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save models
        model_state_dicts = dict(
            G=accelerator.unwrap_model(G).state_dict(),
            D=accelerator.unwrap_model(D).state_dict(),
        )
        accelerator.save(model_state_dicts, os.path.join(save_path, 'model.pt'))
        # save optimizers
        optimizer_state_dicts = dict(
            optimizer_G=optimizer_G.state_dict(),
            optimizer_D=optimizer_D.state_dict(),
        )
        accelerator.save(optimizer_state_dicts, os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        meta_state_dicts = dict(step=step)
        accelerator.save(meta_state_dicts, os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    G, D, optimizer_G, optimizer_D = accelerator.prepare(G, D, optimizer_G, optimizer_D)  # type: ignore
    train_loader, valid_loader = accelerator.prepare(train_loader, valid_loader)  # type: ignore

    # DEFINE LOSSES
    mse = nn.MSELoss()
    bce = nn.BCELoss()

    # EVALUATION METRICS
    metric_l1 = metrics.L1(reduction='none').to(device)
    metric_psnr = metrics.PSNR(reduction='none', data_range=1.).to(device)
    metric_ssim = metrics.SSIM(size_average=False, data_range=1.).to(device)

    accelerator.wait_for_everyone()

    def run_step_G(_batch):
        optimizer_G.zero_grad()
        batch_size = _batch.shape[0]
        loss_meter = metrics.KeyValueAverageMeter(keys=['loss_rec', 'loss_adv_G'])
        for i in range(0, batch_size, micro_batch):
            gt_img = _batch[i:i+micro_batch].float()
            cor_img = gt_img.clone()
            cor_img[:, :, 32:96, 32:96] = 0.
            loss_scale = gt_img.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(G) if no_sync else nullcontext()
            with cm:
                inpainted_img = G(cor_img)
                d_fake = D(inpainted_img)
                loss_rec = mse(inpainted_img, gt_img[:, :, 32:96, 32:96])
                loss_adv_G = bce(d_fake, torch.ones_like(d_fake))
                lossG = cfg.train.coef_rec * loss_rec + cfg.train.coef_adv * loss_adv_G
                accelerator.backward(lossG * loss_scale)
            loss_meter.update(dict(
                loss_rec=loss_rec.detach(),
                loss_adv_G=loss_adv_G.detach(),
            ), cor_img.shape[0])
        optimizer_G.step()
        return dict(
            **loss_meter.avg,
            lr_G=optimizer_G.param_groups[0]['lr'],
        )

    def run_step_D(_batch):
        optimizer_D.zero_grad()
        batch_size = _batch.shape[0]
        loss_meter = metrics.AverageMeter()
        for i in range(0, batch_size, micro_batch):
            gt_img = _batch[i:i+micro_batch].float()
            cor_img = gt_img.clone()
            cor_img[:, :, 32:96, 32:96] = 0.
            loss_scale = gt_img.shape[0] / batch_size

            with torch.no_grad():
                inpainted_img = G(cor_img).detach()
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(D) if no_sync else nullcontext()
            with cm:
                d_real = D(gt_img[:, :, 32:96, 32:96])
                d_fake = D(inpainted_img)
                loss_adv_D = (bce(d_real, torch.ones_like(d_real)) +
                              bce(d_fake, torch.zeros_like(d_fake)))
                accelerator.backward(loss_adv_D * loss_scale)
            loss_meter.update(loss_adv_D.detach(), cor_img.shape[0])
        optimizer_D.step()
        return dict(
            loss_adv_D=loss_meter.avg,
            lr_D=optimizer_D.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def evaluate(dataloader):
        metric_meter = metrics.KeyValueAverageMeter(keys=['l1', 'psnr', 'ssim'])
        for gt_img in tqdm.tqdm(
                dataloader, desc='Evaluating', leave=False,
                disable=not accelerator.is_main_process,
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
        return metric_meter.avg

    @accelerator.on_main_process
    @torch.no_grad()
    def sample(savepath: str):
        unwrapped_G = accelerator.unwrap_model(G)
        gt_img = torch.stack([valid_set[i] for i in range(12)], dim=0).float().to(device)
        cor_img = gt_img.clone()
        cor_img[:, :, 32:96, 32:96] = 0.

        inpainted_img = gt_img.clone()
        inpainted_img[:, :, 32:96, 32:96] = unwrapped_G(cor_img)

        show = []
        for i in tqdm.tqdm(range(12), desc='Sampling', leave=False,
                           disable=not accelerator.is_main_process):
            show.extend([
                image_norm_to_float(gt_img[i]).cpu(),
                image_norm_to_float(cor_img[i]).cpu(),
                image_norm_to_float(inpainted_img[i]).cpu()
            ])
        save_image(show, savepath, nrow=6)

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        G.train(); D.train()
        train_status = run_step_D(batch)
        status_tracker.track_status('Train', train_status, step)
        train_status = run_step_G(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        G.eval(); D.eval()
        # evaluate
        if check_freq(cfg.train.eval_freq, step):
            eval_status = evaluate(valid_loader)
            status_tracker.track_status('Eval', eval_status, step)
            accelerator.wait_for_everyone()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(cfg.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()
