import json
import random
from pathlib import Path
import numpy as np
import torch
import yaml
from PIL import Image
from ldm.models.diffusion.IdeaVisual import IntentDrivenDiffusion


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def device_for(name):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if name == 'auto' else torch.device(name)
    if device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable')
    return device


def load_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if not {'model', 'data', 'train'}.issubset(config): raise ValueError('Require model/data/train sections')
    return config


def to_device(batch, device):
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def save_rgb(image, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().float().cpu().permute(1, 2, 0).clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def write_json(value, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def checkpoint(path, model, config, step, optimizer, scaler):
    ns = np.random.get_state()
    rng = dict(python=random.getstate(), torch=torch.get_rng_state(), numpy_name=ns[0],
               numpy_keys=torch.tensor(ns[1].astype(np.int64)), numpy_pos=ns[2], numpy_gauss=ns[3], numpy_cached=ns[4])
    if torch.cuda.is_available(): rng['cuda'] = torch.cuda.get_rng_state_all()
    value = dict(format_version=1, model=model.state_dict(), config=config, step=step,
                 optimizer=optimizer.state_dict(), scaler=scaler.state_dict(), rng=rng)
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp'); torch.save(value, temporary); temporary.replace(path)


def load_checkpoint(path): return torch.load(path, map_location='cpu', weights_only=True)


def restore_rng(rng):
    random.setstate(rng['python']); torch.set_rng_state(rng['torch'])
    np.random.set_state((rng['numpy_name'], rng['numpy_keys'].numpy().astype(np.uint32), rng['numpy_pos'], rng['numpy_gauss'], rng['numpy_cached']))
    if torch.cuda.is_available() and 'cuda' in rng: torch.cuda.set_rng_state_all(rng['cuda'])


def model_from_checkpoint(path, device):
    state = load_checkpoint(path)
    model = IntentDrivenDiffusion(**state['config']['model']).to(device)
    model.load_state_dict(state['model'], strict=True)
    return model.eval(), state
