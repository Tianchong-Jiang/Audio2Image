from diffusers import StableDiffusionPipeline
import torch

model_path = "stabilityai/stable-diffusion-2"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16)
pipe.to("cuda")

image = pipe(prompt="yoda").images[0]
image.save("/rmx/diffuser/output/image.png")