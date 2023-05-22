from versatile_diffusion.app import vd_inference
from PIL import Image
import os
import pdb

inferece = vd_inference(which='v1.0', fp16=True)

# iterate over images in one folder

file_list = os.listdir("/data/frames")
file_list.sort()

for i in range(10):
    image = Image.open("/data/frames/" + file_list[i])
    image.resize((512, 512), Image.ANTIALIAS)
    text = inferece.inference_i2t(image, 1234)

    with open('/rmx/diffuser/output/prompts.txt', 'a') as f:
        f.write(text + '\n')



