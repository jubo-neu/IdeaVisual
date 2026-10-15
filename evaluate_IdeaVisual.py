import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from ldm.data.intent_pairs import IntentPairDataset
from ideavisual.metrics import psnr, ssim
from ideavisual.runtime import model_from_checkpoint, device_for, to_device, save_rgb, write_json


def main():
    parser = argparse.ArgumentParser(description='Paired intent-driven generation metrics')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', default='test')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--output', default='runs/evaluation')
    parser.add_argument('--steps', type=int, default=30)
    parser.add_argument('--cfg-scale', type=float, default=3.)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--lpips', action='store_true')
    parser.add_argument('--fid', action='store_true', help='Optional Inception-v3 FID, requires sufficient images')
    args = parser.parse_args(); torch.set_num_threads(4); device = device_for(args.device)
    model, _ = model_from_checkpoint(args.checkpoint, device)
    dataset = IntentPairDataset(args.manifest, split=args.split, image_size=model.image_size)
    perceptual, fid = None, None
    if args.lpips:
        import lpips
        perceptual = lpips.LPIPS(net='alex').to(device).eval()
    if args.fid:
        if len(dataset) < 2: raise ValueError('FID needs at least two samples; meaningful estimates need many more')
        from torchmetrics.image.fid import FrechetInceptionDistance
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)
    output = Path(args.output); rows = []
    with torch.no_grad():
        for index, batch in enumerate(DataLoader(dataset, batch_size=1, shuffle=False)):
            batch = to_device(batch, device)
            predicted = model.sample(batch, args.steps, args.cfg_scale, args.seed + index)
            row = dict(id=batch['id'][0], instruction=batch['instruction'][0], psnr=psnr(predicted, batch['target']).item(),
                       ssim=ssim(predicted, batch['target']).item())
            if perceptual is not None: row['lpips'] = perceptual(2 * predicted - 1, 2 * batch['target'] - 1).mean().item()
            if fid is not None: fid.update(batch['target'], real=True); fid.update(predicted, real=False)
            # Numeric filenames avoid trusting external ids as filesystem paths.
            save_rgb(predicted[0], output / 'predicted' / f'{index:06d}.png')
            save_rgb(batch['target'][0], output / 'target' / f'{index:06d}.png'); rows.append(row)
    keys = ['psnr', 'ssim'] + (['lpips'] if perceptual is not None else [])
    result = dict(samples=len(rows), backend=model.backend_kind, per_pair=rows,
                  mean={k: sum(row[k] for row in rows) / len(rows) for k in keys})
    if fid is not None: result['fid'] = fid.compute().item()
    write_json(result, output / 'metrics.json'); print(result['mean'])


if __name__ == '__main__': main()
