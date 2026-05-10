from PIL import Image
import os

img = Image.open('Slide/media/denoising_training_loss_all.png')
width, height = img.size

# Assuming 3 vertical subplots
h_step = height // 3

# Top: Standard
img.crop((0, 0, width, h_step)).save('Slide/media/loss_standard.png')
# Middle: MinMax
img.crop((0, h_step, width, 2*h_step)).save('Slide/media/loss_minmax.png')
# Bottom: LogMinMax
img.crop((0, 2*h_step, width, height)).save('Slide/media/loss_logminmax.png')

print("Loss plots split successfully.")
