from diffusers import AutoencoderKL
from diffusers import StableDiffusionPipeline
import torch
import cv2
import numpy as np

image = cv2.imread("/rmx/diffuser/code/image.png")
image = torch.tensor(image).to("cuda").permute(2, 0, 1).unsqueeze(0).half() / 255.0

# initialize encoder
model_path = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

# encode image
latent = pipe.vae.encode(image)

res_image = latent.latent_dist.mean[0]


rgba_image = (res_image.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

# set rgb channels to 255
rgba_image[:, :, 3] = rgba_image[:, :, 0]
rgba_image[:, :, 0] = rgba_image[:, :, 1]
rgba_image[:, :, 1] = rgba_image[:, :, 2]
rgba_image[:, :, 2] = rgba_image[:, :, 3]


rgb_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2RGB)
cv2.imwrite("/rmx/diffuser/output/image.png", rgb_image)





