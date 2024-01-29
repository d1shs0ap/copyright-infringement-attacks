python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml \
               -t \
               --actual_resume models/ldm/text2img-large/model.ckpt \
               -n invert \
               --gpus 1, \
               --data_root poison/img1 \
               --init_word house