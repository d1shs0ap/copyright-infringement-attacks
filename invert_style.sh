python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune_style.yaml \
               -t \
               --actual_resume models/ldm/text2img-large/model.ckpt \
               -n invert \
               --gpus 1, \
               --data_root poison/img_style \
               --init_word van