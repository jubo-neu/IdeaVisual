import argparse
from pathlib import Path
import torch
from ldm.data.intent_pairs import load_rgb, load_depth
from ideavisual.runtime import model_from_checkpoint, device_for, save_rgb, write_json


def main():
    parser = argparse.ArgumentParser(description='Generate intent-conditioned views from source RGB and source depth')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--image', required=True)
    parser.add_argument('--depth', required=True)
    parser.add_argument('--instruction', action='append', required=True, help='Repeat for multiple requested views')
    parser.add_argument('--output', default='runs/inference')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg-scale', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args(); torch.set_num_threads(4); device = device_for(args.device)
    model, state = model_from_checkpoint(args.checkpoint, device)
    size = model.image_size; output = Path(args.output); output.mkdir(parents=True, exist_ok=True)
    batch = dict(source=load_rgb(args.image, size)[None].to(device), depth=load_depth(args.depth, size)[None].to(device))
    for i, text in enumerate(args.instruction):
        if not text.strip(): raise ValueError('Instruction must not be empty')
        batch['instruction'] = [text]
        result = model.sample(batch, args.steps, args.cfg_scale, args.seed)
        save_rgb(result[0], output / f'{i:03d}.png')
    write_json(dict(instructions=args.instruction, source=str(Path(args.image).resolve()),
                    depth=str(Path(args.depth).resolve()), backend=model.backend_kind,
                    checkpoint_step=state['step'], seed=args.seed, steps=args.steps, cfg_scale=args.cfg_scale), output / 'generation.json')
    print(f'Generated {len(args.instruction)} images: {output}')


if __name__ == '__main__': main()
