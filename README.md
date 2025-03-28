# Idea Visual
## [Paper] Idea Visual: Intent-Driven View Synthesis via Multimodal Diffusion Model
These are codes and dataset releases for submission.

![Teaser](https://github.com/jubo-neu/IdeaVisual/blob/main/teaser.png)

## TODO
- [x] Benchmark.
- [x] Models and tools.
- [ ] Training code and inference code.
- [ ] Checkpoints.

❗️2025-01-20: To resist plagiarism, Training code and Inference code will be available upon acceptance.

### News
- Training code is being released.
- Inference code and pretrained weight are coming soon.

## Preparation

To start, we prefer creating the environment using conda:

```
conda env create -f environment.yml
conda activate IdeaVisual
```

Alternatively use

```
pip install -r requirements.txt
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
