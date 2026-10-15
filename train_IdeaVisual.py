"""Complete training entry point for the public IdeaVisual repository."""
import argparse
from ideavisual.runtime import load_config, device_for
from ideavisual.train import train


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('-conf', '--config', default='configs/smoke.yaml')
    parser.add_argument('-c', '--output')
    parser.add_argument('--device', default='auto')
    parser.add_argument('--stage', type=int, choices=[1, 2, 3])
    parser.add_argument('--resume')
    parser.add_argument('--init-from', help='Completed previous-stage checkpoint; weights only, new optimizer')
    parser.add_argument('--steps', type=int)
    parser.add_argument('--manifest')
    args = parser.parse_args()
    if args.resume and args.init_from: parser.error('--resume and --init-from are mutually exclusive')
    config = load_config(args.config)
    if args.output: config['train']['output'] = args.output
    if args.stage: config['model']['stage'] = args.stage
    if args.steps is not None: config['train']['steps'] = args.steps
    if args.manifest: config['data']['manifest'] = args.manifest
    print(f"Checkpoint: {train(config, device_for(args.device), args.resume, args.init_from)}")


if __name__ == '__main__': main()
