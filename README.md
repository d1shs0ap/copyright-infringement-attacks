# [ICML 2024] Disguised Copyright Infringement of Latent Diffusion Models ([arXiv](https://arxiv.org/abs/2404.06737))

[Yiwei Lu*](https://cs.uwaterloo.ca/~y485lu/), [Matthew Y.R. Yang*](https://www.matthewyryang.com/), Zuoqiu Liu*, [Gautam Kamath](http://www.gautamkamath.com/), [Yaoliang Yu](https://cs.uwaterloo.ca/~y328yu/)


![Alt text](https://github.com/d1shs0ap/copyright-infringement-attacks/blob/main/img/intro2.png)

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion) and [Textual Inversion](https://github.com/rinongal/textual_inversion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm-copyright
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Data
- Symbol experiment: `poison/bottle_watermark_clipped/`
-  Content experiment: `poison/sunflowers_clipped/`
- Style experiment: `poison/style_clipped/` 

`/base_images` contains all the base images used, `/target_images` contains all corresponding target images used, `/poison_clipped_output` includes some demo results


## Disguised Poison Generation
To generate disguised image from a base image to a target image, run

```
python create_poison.py
```

The following parameters can be tweaked for your experiments:

- `base_instance`: base image (in e.g., png, jpg)
- `target_instance`: target image
- `gpu`: id of the GPU that you want to run the experiments on
- `alpha`: hyperparameter that controls the tradeoff between input space constraint and feature space constraint
- `save_folder`
- `load_folder`
- `clipped`: if set to `True`, bound the image between 0 and 1

For each 1000 iterations, three images will be saved:
1. `poison_$iteration.pt`: your poisoned image in `.pt` format. **This, rather than the images, should be fed into textual inversion's `invert.sh`.**
2. `poison_$iteration.jpg`: your poisoned image, displayed in `.jpg` format.
3. `poison_$iteration_decoded.jpg`: the "revealed" disguise for encoder-decoder examination.

Be sure to define the ```TO_BE_DEFINED Parameters``` in `create_poison.py`. (i.e. make sure `save_folder`, `base_image_path`, `target_image_path` are defined properly)

## Textual inversion
To invert an image set, run:

```
sh invert.sh
```

To generate new images of the learned concept, run:

```
sh generate.sh
```

You can find learned concepts in `/outputs` and `/logs/$your_log_name/images/samples_scaled_gs....jpg`.

### Style
To invert an image set style, run:

```
sh invert_style.sh
```

To generate new images of the learned style, run:

```
sh generate_style.sh
```

For more details on the textual inversion process, please refer to [Textual Inversion](https://github.com/rinongal/textual_inversion).

## Appendix A: unbounded disguises

To include unbounded disguises, set `loss = feature_similarity_loss` in `create_poison.py`.

## Appendix B: data augmentation

To create poisons that are robust against the horizontal flip data augmentation, set `loss = feature_similarity_loss + flipped_feature_similarity_loss + noise_loss` in `create_poison.py`.

To add in the horizontal flip data augmentation in textual inversion, uncomment `image = transforms.functional.hflip(image)` in `ldm/data/personalized.py`.

## Appendix C: quantitative detection

To reproduce the quantitative detection results, run `detection.ipynb` **after generating all symbol, style, and content poisons**.

## Appendix D : circumventing detection

To create poisons that circumvent detection, set `loss = feature_similarity_loss + noise_loss + reconstruction_loss` in `create_poison.py`.

## Appendix E : additional experiments on disguised style

The based images and corresponding target images are provided at `poison/appendix_style_clipped/`.
(`/base_images` contains all the base images used, `/target_images` contains all corresponding target images used, `/poison_clipped_output` includes some demo results)


## BibTeX Citation

If you use our code in a scientific publication, we would appreciate using the following citations:

```
@inproceedings{LuYLKY24,
  title       ={Disguised Copyright Infringement of Latent Diffusion Model},
  author      ={Lu, Yiwei and Yang, Matthew YR and Liu, Zuoqiu and Kamath, Gautam and Yu, Yaoliang},
  booktitle   ={International Conference on Machine Learning},
  year        ={2024},
  url         ={https://arxiv.org/abs/2404.06737},
}
```
