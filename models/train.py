"""Native PyTorch three-stage training, optional DDP, AMP and exact CPU resume."""
import copy
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import default_collate
from tqdm import trange
from ldm.data.intent_pairs import IntentPairDataset
from ldm.models.diffusion.IdeaVisual import IntentDrivenDiffusion
from .runtime import seed_all, to_device, checkpoint, load_checkpoint, restore_rng, write_json


def distributed_rng():
    # Reuse checkpoint RNG format for independent DDP rank streams.
    import random
    import numpy as np
    ns = np.random.get_state()
    result = dict(python=random.getstate(), torch=torch.get_rng_state(), numpy_name=ns[0],
                  numpy_keys=torch.tensor(ns[1].astype(np.int64)), numpy_pos=ns[2], numpy_gauss=ns[3], numpy_cached=ns[4])
    if torch.cuda.is_available(): result['cuda'] = torch.cuda.get_rng_state_all()
    return result


def train(config, device, resume=None, init_from=None):
    settings = config['train']; distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0')); world = int(os.environ.get('WORLD_SIZE', '1'))
    if distributed:
        if device.type != 'cuda': raise RuntimeError('DDP launcher requires CUDA devices')
        local_rank = int(os.environ['LOCAL_RANK']); torch.cuda.set_device(local_rank); device = torch.device('cuda', local_rank)
        dist.init_process_group('nccl')
    seed_all(settings.get('seed', 42) + rank); torch.set_num_threads(settings.get('cpu_threads', 4))
    if min(settings['steps'], settings.get('batch_size', 1)) < 1: raise ValueError('Steps/batch size must be positive')
    root = Path(settings['output']); root.mkdir(parents=True, exist_ok=True)
    dataset = IntentPairDataset(**config['data'], split='train', image_size=config['model']['image_size'])
    model = IntentDrivenDiffusion(**config['model']).to(device)
    if init_from:
        state = load_checkpoint(init_from)
        old = dict(state['config']['model']); new = dict(config['model']); old.pop('stage', None); new.pop('stage', None)
        if old != new: raise ValueError('Stage initialization architecture differs')
        model.load_state_dict(state['model'], strict=True)
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=settings.get('lr', 5e-4), weight_decay=settings.get('weight_decay', .01))
    amp = settings.get('amp', False) and device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp); start = 0
    if resume:
        state = load_checkpoint(resume)
        if state['config']['model'] != config['model'] or state['config']['data'] != config['data']: raise ValueError('Resume model/data configuration differs')
        for key in ('lr', 'min_lr', 'schedule_steps', 'batch_size', 'loss_weights', 'amp', 'weight_decay', 'grad_clip'):
            if state['config']['train'].get(key) != settings.get(key): raise ValueError(f'Resume configuration differs: {key}')
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        if distributed:
            if 'rank_rng' not in state or len(state['rank_rng']) != world: raise ValueError('DDP resume requires same world size/rank RNG checkpoint')
            restore_rng(state['rank_rng'][rank])
        else: restore_rng(state['rng'])
        start = state['step']
    if settings['steps'] <= start: raise ValueError('Total steps must exceed resumed step')
    if rank == 0: write_json(config, root / 'config.json')
    runner = DistributedDataParallel(model, device_ids=[device.index], find_unused_parameters=True) if distributed else model
    progress = trange(start, settings['steps'], disable=rank != 0, desc=f"IdeaVisual stage {model.stage}")
    for step in progress:
        model.train()
        items = [dataset[torch.randint(len(dataset), ()).item()] for _ in range(settings.get('batch_size', 1))]
        batch = to_device(default_collate(items), device)
        fraction = min(step / max(settings.get('schedule_steps', settings['steps']) - 1, 1), 1.)
        lr = settings.get('lr', 5e-4) * (1 - fraction) + settings.get('min_lr', 1e-5) * fraction
        for group in optimizer.param_groups: group['lr'] = lr
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp): losses = runner(batch, settings.get('loss_weights'))
        if not torch.isfinite(losses['loss']): raise FloatingPointError(f'Nonfinite loss at {step + 1}')
        scaler.scale(losses['loss']).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, settings.get('grad_clip', 1.), error_if_nonfinite=not amp)
        scaler.step(optimizer); scaler.update()
        if rank == 0 and (step == start or (step + 1) % settings.get('log_every', 50) == 0):
            row = dict(step=step + 1, lr=lr, **{k: v.detach().item() for k, v in losses.items()})
            with (root / 'train.jsonl').open('a', encoding='utf-8') as handle: handle.write(json.dumps(row) + '\n')
            progress.set_postfix(loss=f"{row['loss']:.4f}")
        if (step + 1) % settings.get('save_every', 1000) == 0 or step + 1 == settings['steps']:
            states = [None] * world if distributed else None
            if distributed: dist.all_gather_object(states, distributed_rng())
            if rank == 0:
                path = root / 'latest.pt'; checkpoint(path, model, config, step + 1, optimizer, scaler)
                if distributed:
                    value = load_checkpoint(path); value['rank_rng'] = states
                    temporary = path.with_suffix('.tmp'); torch.save(value, temporary); temporary.replace(path)
    if distributed: dist.barrier(); dist.destroy_process_group()
    return root / 'latest.pt'
