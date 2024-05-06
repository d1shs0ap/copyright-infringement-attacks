from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from FSSAAD.pytorch_msssim import msssim
from torch.nn import functional as F

import torch
import torchvision
import gc
import pickle
import os
import time


from ldm.util import instantiate_from_config
 
''' Parameters
image_instance: the index/id of input sample, i.e. for bottle_watermark_clipped, use one of {1, 2, 3, 4}
gpu: which gpu device to use
alpha: hyperparameter on input space constraint
save_folder: the folder to save all poison images
load_folder: the folder to load image_instance
clipped: if set to True, bound the image between 0 and 1 (this should expect to be True in most cases)
'''

gpu = 0
alpha = 8000
save_folder = f'poison/bottle_watermark_clipped/img_train_1'
base_image_path = "poison/bottle_watermark_clipped/base_images/1.png" 
target_image_path = "poison/bottle_watermark_clipped/target_images/1.png"
clipped = True

''' load_model_from_config will return a torch model, by given a config and a model ckpt
'''
def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    config.model.params.ckpt_path = ckpt
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    
    return model

''' load_image_toTensor will load an image from path and convert it to a PyTorch tensor
'''
def load_image_toTensor(path):
    image_path = path
    image = Image.open(image_path)  # Load the image using PIL
    image = image.convert("RGB")


    transform = ToTensor()  # Create a ToTensor transform
    input_image_tensor = transform(image)  # Convert image to tensor

    # Reshape the tensor to match the expected input shape of the encoder
    input_image_tensor = input_image_tensor.unsqueeze(0)  # Add batch dimension
    return input_image_tensor

def inference(model, x):
    posterior = model.encode(x)
    return posterior.mode()

''' Our Disguise Generation Algorithm
'''
def poison_noise(model, noise, base_instance, target_instance, optimizer, i):

    ''' compute_feature_similarity_loss will compute the feature similarity loss between current and target instance for monitoring use
    '''
    def compute_feature_similarity_loss(current, target):
        current_decoded, current_posterior = model(current)
        target_decoded, target_posterior = model(target)

        current_feature = current_posterior.mode()
        target_feature = target_posterior.mode()
        l2_feature_loss = 1/alpha * torch.norm(current_feature - target_feature, p=2)
        return l2_feature_loss, current_decoded, target_decoded

    # we do not modify the model, so we set it to evaluation mode
    model.eval()

    if clipped:
        clipped_instance = torch.clamp(base_instance + noise, min=0, max=1)
    else:
        clipped_instance = base_instance + noise

    # feature similarity loss
    feature_similarity_loss, current_decoded, target_decoded = compute_feature_similarity_loss(clipped_instance, target_instance)
    flipped_feature_similarity_loss = compute_feature_similarity_loss(torchvision.transforms.functional.hflip(clipped_instance), torchvision.transforms.functional.hflip(target_instance))

    # constrain noise size
    msssim_noise_loss = (1 - msssim(base_instance, clipped_instance)) / 2
    f1_noise_loss = F.l1_loss(base_instance, clipped_instance)
    noise_loss = msssim_noise_loss + f1_noise_loss

    # detection evasion loss
    msssim_reconstruction_loss = (1 - msssim(current_decoded, clipped_instance)) / 2
    f1_reconstruction_loss = F.l1_loss(current_decoded, clipped_instance)
    reconstruction_loss = msssim_reconstruction_loss + f1_reconstruction_loss
    
    loss = feature_similarity_loss + noise_loss
    # loss = feature_similarity_loss + noise_loss + reconstruction_loss
    # loss = feature_similarity_loss + flipped_feature_similarity_loss + noise_loss
    # loss = feature_similarity_loss

    loss.backward()
    optimizer.step()

    if i % 1000 == 0 and i > 0:
        print(f"Epoch {i}")
        print(f"feature loss: {feature_similarity_loss.item()}")
        # print(f"flipped feature loss: {flipped_feature_similarity_loss.item()}")
        print(f"noise loss: {noise_loss.item()}")
        print(f"total loss: {loss.item()}")
        
        save_image(clipped_instance, i)
        torch.save(clipped_instance, f'{save_folder}/poison_{i}.pt')

        decoded_instance, _ = model(clipped_instance)
        save_image(decoded_instance, i, decoded=True)

    return noise

''' save_image will save a image tensor to an image file
'''
def save_image(tensor, iteration, decoded=False):
    tensor = tensor.squeeze(0)
    tensor = tensor.clone().detach().cpu()
    
    transform_to_image = ToPILImage()
    image_converted = transform_to_image(tensor)
    
    image_converted.save(f'{save_folder}/poison_{iteration}{"_decoded" if decoded else ""}.jpg')


start = time.time()

# make directory
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# set up
torch.cuda.empty_cache()
gc.collect()

device = torch.device(f"cuda:{gpu}" if torch.cuda.is_available() else "cpu")

with open('config.pkl', 'rb') as f:
    config = pickle.load(f)
model = load_model_from_config(config, "models/ldm/text2img-large/model.ckpt").first_stage_model.to(device)

for param in model.parameters():
    param.requires_grad = False

# load images and noise
base_instance = load_image_toTensor(base_image_path).to(device) # share similar input space with this image
target_instance = load_image_toTensor(target_image_path).to(device) # share similar target space with this image
# base_instance = torch.zeros(target_instance.shape).to(device)
# base_instance = 0.5 + torch.randn(target_instance.shape).to(device)

noise = torch.zeros(base_instance.shape, requires_grad=True, device=device)
optimizer = torch.optim.SGD([noise], lr=0.01)

# training loop
print(0)
save_image(base_instance, 0)

decoded_instance, _ = model(base_instance)
save_image(decoded_instance, 0, decoded=True)

# run our poison algorithm for 100000 iterations.
for i in range(100001):
    noise = poison_noise(model, noise, base_instance, target_instance, optimizer, i)

print(f"Time: {(time.time() - start) / 3600} hours")