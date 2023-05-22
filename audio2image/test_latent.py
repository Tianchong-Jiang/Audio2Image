from versatile_diffusion.app import vd_inference
from PIL import Image
import pdb

inferece = vd_inference(which='v1.0', fp16=True)

image = Image.open("/rmx/diffuser/code/image.png")

text, latent_code = inferece.inference_i2l(image, 1234)

# latent_code = latent_code[0].repeat(77, 1)

output_image = inferece.inference_l2i(latent_code, 1234)

output_image[0].save("/rmx/diffuser/output/image.png")
