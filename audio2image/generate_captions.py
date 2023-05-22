from versatile_diffusion.app import vd_inference
from PIL import Image
import os
import csv
import pdb

inferece = vd_inference(which='v1.0', fp16=True)

# iterate over images in one folder

file_list = os.listdir("/data/frames")
file_list.sort()

res = []

for i in range(len(file_list)):
    image = Image.open("/data/frames/" + file_list[i])
    image.resize((512, 512), Image.ANTIALIAS)
    text = inferece.inference_i2t(image, 1234)
    text = text.split('\n')
    audio_name = file_list[i].split('.')[0]

    for text_line in text:
        text_line = text_line.replace('"', '').replace(',', '')
        res.append([audio_name, str(i), text_line])

with open('/rmx/diffuser/output/prompts_10per.csv', 'w') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(res)


