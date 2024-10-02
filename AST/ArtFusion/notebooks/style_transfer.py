CFG_PATH = '../configs/kl16_content12.yaml'
CKPT_PATH = '../checkpoints/artfusion/artfusion_r12_step=317673.ckpt'

H = 256
W = 256
DDIM_STEPS = 250
ETA = 1.
SEED = 2023
DEVICE = 'cuda'
import sys
sys.path.append('../')

import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from PIL import ImageDraw, ImageFont, Image
import matplotlib.pyplot as plt
from einops import rearrange
from omegaconf import OmegaConf
import albumentations

from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler

seed_everything(SEED)

config = OmegaConf.load(CFG_PATH)
config.model.params.ckpt_path = CKPT_PATH
config.model.params.first_stage_config.params.ckpt_path = None
model = instantiate_from_config(config.model)
model = model.eval().to(DEVICE)


def preprocess_image(image_path, size=(W, H)):
    image = Image.open(image_path)
    if not image.mode == "RGB":
        image = image.convert("RGB")
    image = image.resize(size)
    image = np.array(image).astype(np.uint8)
    image = (image / 127.5 - 1.0).astype(np.float32)
    image = rearrange(image, 'h w c -> c h w')
    return torch.from_numpy(image)


def display_samples(samples, n_columns=1, figsize=(12, 12)):
    if isinstance(samples, (list, tuple)):
        samples = torch.cat(samples, dim=0)

    samples = rearrange(samples, '(n m) c h w -> (m h) (n w) c', n=n_columns).cpu().numpy() * 255.
    samples = Image.fromarray(samples.astype(np.uint8))
    plt.rcParams["figure.figsize"] = figsize
    plt.imshow(samples)
    plt.axis('off')
    plt.show()


def tensor_to_rgb(x):
    return torch.clamp((x + 1.0) / 2.0, min=0.0, max=1.0)


'''
style_image_paths = [
    '../data/styles/d523d66a2f745aff1d3db21be993093fc.jpg',
    '../data/styles/the_scream.jpg',
    '../data/styles/Claude_Monet_73.jpg',
    '../data/styles/430f12a69a198bf3228f8177ed436624c.jpg'
]
'''
root_style = '/home/hfle/dataset/website_for_replace/source_style/'
style_image_paths = os.listdir("/home/hfle/dataset/website_for_replace/source_style")
style_image_paths.sort()
for index in range(len(style_image_paths)):
    style_image_paths[index] = root_style + style_image_paths[index]


style_images = torch.stack([preprocess_image(p) for p in style_image_paths], dim=0).to(DEVICE)

display_samples(tensor_to_rgb(style_images), n_columns=len(style_images))

bs = len(style_images)

with torch.no_grad(), model.ema_scope("Plotting"):
    c_content = torch.zeros((bs, model.channels, model.image_size, model.image_size)).to(DEVICE)

    vgg_features = model.vgg(model.vgg_scaling_layer(style_images))
    c_style = model.get_style_features(vgg_features)

    c = {'c1': c_content, 'c2': c_style}

    samples = model.sample_log(cond=c, batch_size=bs, ddim=True, ddim_steps=DDIM_STEPS, eta=1.)[0]

    x_samples = model.decode_first_stage(samples)
    x_samples = tensor_to_rgb(x_samples)

display_samples(x_samples, n_columns=bs)


def get_content_style_features(content_image_path, style_image_path, h=H, w=W):
    style_image = preprocess_image(style_image_path)[None, :].to(DEVICE)
    content_image = preprocess_image(content_image_path, size=(w, h))[None, :].to(DEVICE)

    with torch.no_grad(), model.ema_scope("Plotting"):
        vgg_features = model.vgg(model.vgg_scaling_layer(style_image))
        c_style = model.get_style_features(vgg_features)
        null_style = c_style.clone()
        null_style[:] = model.null_style_vector.weight[0]

        content_encoder_posterior = model.encode_first_stage(content_image)
        content_encoder_posterior = model.get_first_stage_encoding(content_encoder_posterior)
        c_content = model.get_content_features(content_encoder_posterior)
        null_content = torch.zeros_like(c_content)

    c = {'c1': c_content, 'c2': c_style}
    c_null_style = {'c1': c_content, 'c2': null_style}
    c_null_content = {'c1': null_content, 'c2': c_style}

    return c, c_null_style, c_null_content


def style_transfer(
        content_image_path, style_image_path,
        h=H, w=W,
        content_s=1., style_s=1.,
        ddim_steps=DDIM_STEPS, eta=ETA,
):
    c, c_null_style, c_null_content = get_content_style_features(content_image_path, style_image_path, h, w)

    with torch.no_grad(), model.ema_scope("Plotting"):
        samples = model.sample_log(
            cond=c, batch_size=1, x_T=torch.rand_like(c['c1']),
            ddim=True, ddim_steps=ddim_steps, eta=eta,
            unconditional_guidance_scale=content_s, unconditional_conditioning=c_null_content,
            unconditional_guidance_scale_2=style_s, unconditional_conditioning_2=c_null_style)[0]

        x_samples = model.decode_first_stage(samples)
        x_samples = tensor_to_rgb(x_samples)

    return x_samples

content_image_path = '../data/contents/lofoton.jpg'
style_image_path = '../data/styles/d523d66a2f745aff1d3db21be993093fc.jpg'

style_image = preprocess_image(style_image_path)[None, :]
content_image = preprocess_image(content_image_path)[None, :]

display_samples((tensor_to_rgb(content_image), tensor_to_rgb(style_image)), figsize=(6, 3), n_columns=2)

x_samples = style_transfer(content_image_path, style_image_path, content_s=0.5, style_s=2.)

display_samples(x_samples, figsize=(3, 3))


def two_dim_cfg(
        content_image_path, style_image_path,
        content_scalers=[0.25, 0.5, 1.0, 2.0, 4.0], style_scalers=[0.15, 0.5, 1., 3., 5.],
        ddim_steps=DDIM_STEPS, eta=ETA,
):
    c, c_null_style, c_null_content = get_content_style_features(content_image_path, style_image_path)

    with torch.no_grad(), model.ema_scope("Plotting"):
        samples = list()
        for style_s in style_scalers:
            for content_s in content_scalers:
                sample = model.sample_log(
                    cond=c, batch_size=1,
                    ddim=True, ddim_steps=ddim_steps, eta=eta,
                    unconditional_guidance_scale=content_s, unconditional_conditioning=c_null_content,
                    unconditional_guidance_scale_2=style_s, unconditional_conditioning_2=c_null_style)[0]
                samples.append(sample)
        samples = torch.cat(samples, dim=0)
        x_samples = model.decode_first_stage(samples)
        x_samples = tensor_to_rgb(x_samples)

    return x_samples

x_samples = two_dim_cfg(content_image_path, style_image_path)
display_samples(x_samples, n_columns=5)