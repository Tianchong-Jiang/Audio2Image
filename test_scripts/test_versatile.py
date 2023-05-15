from diffusers import VersatileDiffusionPipeline
import torch
import requests
from io import BytesIO
from PIL import Image

# let's download an initial image
url = "https://huggingface.co/datasets/diffusers/images/resolve/main/benz.jpg"

response = requests.get(url)
image = Image.open(BytesIO(response.content)).convert("RGB")
text = "a red car in the sun"

pipe = VersatileDiffusionPipeline.from_pretrained(
    "shi-labs/versatile-diffusion", torch_dtype=torch.float16
)
pipe = pipe.to("cuda")

generator = torch.Generator(device="cuda").manual_seed(0)
text_to_image_strength = 0.75

image = pipe.dual_guided(
    prompt=text, image=image, text_to_image_strength=text_to_image_strength, generator=generator
).images[0]
image.save("/rmx/diffuser/output/image.png")