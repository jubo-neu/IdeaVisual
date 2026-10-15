# Idea Visual
## [Paper] Idea Visual: Intent-Driven View Synthesis for Smart Mobile Devices via Retrieval-Augmented Diffusion Models

## Graphic overview

![Teaser](https://github.com/jubo-neu/IdeaVisual/blob/main/teaser.png)

## Highlights

- **Intent-driven view synthesis:** input a source RGB image, its fixed source depth map, and a natural-language instruction.
- **Multimodal interaction:** frozen visual/text features, image-query/text-key cross-attention, and a two-linear-layer GELU intent projector.
- **Efficient adaptation:** intent-aware, token-wise top-k LoRA experts in frozen decoder self-attention and FFN projections.

## TODO
- [x] Training and inference.
- [x] Text-aware Answerer loading and data validation.
- [x] Vision-language interaction and intent projector.
- [x] Sparse LoRA experts, top-k routing and auxiliary loss.
- [x] Three-stage trainability policy and stage transfer.
- [x] DDIM sampling and classifier-free guidance.
- [x] PSNR/SSIM evaluation; optional LPIPS/FID.

## Preparation

### Minimal CPU environment

Python 3.10 or newer is required. Run commands from the repository root.

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

### Full diffusion environment

Install matching PyTorch/torchvision builds for your GPU, then:

```bash
pip install -e ".[diffusion,metrics,dev]"
```

Or use the suggested CUDA 12.1 setup:

```bash
conda env create -f environment.yml
conda activate IdeaVisual
pip install -e ".[diffusion,metrics,dev]"
```

## Getting the data

- Download benchmark [Answerer](https://1drv.ms/u/s!AoXcO8rD9StlbTeugyChsRUkzQM?e=i9QaKy).

- We have updated the `tools` folder, which contains multiple Python scripts for operating [Objaverse](https://objaverse.allenai.org/). With these scripts, users can create `Answerer` target categories according to their expectations. Additionally, the `tools/dataAPI.py` includes the `system prompt` used when constructing the Answerer, and you can use this prompt to independently verify the descriptive effectiveness of the views.

## Training
1. Prepare view-instruction pairs for training. Our `Answerer` provides the complete training set for training `Idea Visual`. Download and unzip the dataset to your project folder, ensuring that you have the following files:
```bash
Idea Visual
|-- Answerer_dataset
    |-- Answerer_depth
        |-- 00000.png
        |-- 00001.png
        |-- ...
    |-- Answerer_rgb
        |-- 00000.png
        |-- 00001.png
        |-- ...
    |-- obj_describe_Eng.json # This is a .json file containing the corresponding instruction descriptions for each view.
```
2. (Optional) Create view-instruction pairs for any target. The multimodal intent-driven diffusion model uses the depth map corresponding to the view as the conditional input for the image branch. For any category, we use [MiDaS v3.1](https://github.com/isl-org/MiDaS) BEIT-L-512 to generate the depth map. Note that the same naming convention is applied as for RGB images. We also provide a script `tools/dataAPI.py` to generate corresponding textual descriptions for RGB images. You can provide the vision language model key based on your needs and then run it:
```bash
python /tools/dataAPI.py --folder_path <RGB-images-to-input> --output_file <instuctions-in-.json-format-to-output>
```
3. Download the pretrained multi-view diffusion model [here](https://1drv.ms/u/c/652bf5c3ca3bdc85/EeK64dayqmxJgNqSIquzp2ABbWluSn7D_5SgXU61RnPWKw) and text/image encoder [here](https://1drv.ms/u/c/652bf5c3ca3bdc85/ETlxiz1yhitGqJlCAhIE8CYBe3SVcPUK3yCuqdPHRf5E3A?e=cL6uTe).
4. Start training:
```bash
python train_IdeaVisual.py -conf configs/ideavisual.yaml -c <ckpt-dir> --ft <your multi-view diffusion model> <your text/image encoder> --gpus 0,1,2,3,4,5,6,7
```
- `-c` is the checkpoint path to save.
- `--ft` is the pretrained model path used for fine-tuning.
- `--gpus` is your GPU numbers.

## Inference

```bash
python inference_IdeaVisual.py --checkpoint runs/stage3/latest.pt --image examples/source.png --depth examples/source_depth.npy --instruction "Show the content on the monitor screen" --steps 50 --cfg-scale 3.0 --seed 42 --device cuda --output runs/generated
```

Repeat `--instruction` to generate several requested views of one object:

```bash
python inference_IdeaVisual.py --checkpoint runs/stage3/latest.pt --image examples/source.png --depth examples/source_depth.npy --instruction "Show the screen" --instruction "Show the connector panel behind the monitor" --device cuda --output runs/multiple_intents
```

Each instruction is sampled separately using the same seed. Outputs are PNGs plus `generation.json`. Multiple outputs are not jointly constrained by a new cross-view consistency loss; 3D consistency requires trained-model validation. Input/source depth preprocessing must match training. CLIP truncates text at its context limit; the interaction module does not make its underlying encoder unlimited in context length.

## Evaluation

```bash
python evaluate_IdeaVisual.py --checkpoint runs/stage3/latest.pt --manifest Answerer_dataset/pairs.json --split test --steps 50 --device cuda --output runs/eval
# Optional pretrained perceptual and distribution metrics:
python evaluate_IdeaVisual.py --checkpoint runs/stage3/latest.pt --manifest Answerer_dataset/pairs.json --split test --steps 50 --device cuda --output runs/eval_full --lpips --fid
```

Paired RGB PSNR and SSIM are always computed; LPIPS uses AlexNet; optional FID uses 2048-dimensional Inception features through torchmetrics/torch-fidelity. Metric networks may download their weights on first use. Meaningful FID requires a large representative sample set, not the four-item demo. SSIM uses an 11x11 Gaussian kernel, sigma 1.5, valid convolution and RGB range `[0,1]`. Results include per-pair scores and image outputs.

## Citation

```bibtex
@article{chen2025ideavisual,
  title={Idea Visual: Intent-Driven View Synthesis for Smart Mobile Devices via Retrieval-Augmented Diffusion Models},
  author={Chen, Jubo and Yu, Xiaosheng and Tang, Yangpu and Jiang, Zhengxuan},
  journal={IEEE Transactions on Consumer Electronics},
  volume={71},
  number={4},
  pages={11791--11800},
  year={2025},
  doi={10.1109/TCE.2025.3624269}
}
```
