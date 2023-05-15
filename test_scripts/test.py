import sys
from diffusers import DDPMScheduler, UNet2DModel
from PIL import Image
import torch
import numpy as np
import cv2
print(sys.path)

from diffusers import DiffusionPipeline

pipeline = DiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
pipeline.to("cuda")
pipeline("An image of a squirrel in Picasso style").images[0]

scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256").to("cuda")
scheduler.set_timesteps(50)

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size)).to("cuda")
input = noise

output = []
for t in scheduler.timesteps:
    with torch.no_grad():
        output.append(input)
        noisy_residual = model(input, t).sample
        prev_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
        input = prev_noisy_sample

# saving output as a mp4 video using opencv
out = cv2.VideoWriter("/rmx/diffuser/output/video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 5, (256, 256))
for frame in output:
    frame = (frame / 2 + 0.5).clamp(0, 1)
    frame = frame.cpu().permute(0, 2, 3, 1).numpy()[0]
    frame = (frame * 255).round().astype("uint8")
    out.write(frame)

# save image to output directory
image = (input / 2 + 0.5).clamp(0, 1)
image = image.cpu().permute(0, 2, 3, 1).numpy()[0]
image = Image.fromarray((image * 255).round().astype("uint8"))

image.save("/rmx/diffuser/output/image.png")