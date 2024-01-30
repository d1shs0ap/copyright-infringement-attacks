# Disguised Copyright Infringement of Latent Diffusion Models

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion) and [Textual Inversion](https://github.com/rinongal/textual_inversion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

### Disguised Poison Generation
To generate disguised image from a base image to a target image, run

```
python create_poison.py
```

and change the paths for `base_instance` and `target_instance` accordingly.

### Textual inversion
To invert an image set, run:

```
sh invert.sh
```

To generate new images of the learned concept, run:

```
sh generate.sh
```

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