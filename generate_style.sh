python scripts/txt2img.py --ddim_eta 0.0 \
                          --n_samples 8 \
                          --n_iter 2 \
                          --scale 10.0 \
                          --ddim_steps 50 \
                          --embedding_path logs/img_style2024-01-24T11-12-34_invert/checkpoints/embeddings_gs-6099.pt \
                          --ckpt_path models/ldm/text2img-large/model.ckpt \
                          --prompt "a painting in the style of *"
