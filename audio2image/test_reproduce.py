from versatile_diffusion.app import vd_inference

inferece = vd_inference(which='v1.0', fp16=True)

image = inferece.inference_t2i("a car is driving on the road", 1234)

image[0].save("/rmx/diffuser/output/image.png")

text = inferece.inference_i2t(image[0], 1234)

print(text)