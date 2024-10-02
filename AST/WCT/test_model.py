import torch
import models.vgg_normalised_conv2_1 as vgg_normalised_conv2_1
model = vgg_normalised_conv2_1.vgg_normalised_conv2_1
checkpoint= torch.load('models/vgg_normalised_conv2_1.pth')
model.load_state_dict(checkpoint)
print(checkpoint.keys())