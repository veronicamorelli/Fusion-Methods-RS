import torchvision.models as models
import torch
from ptflops import get_model_complexity_info
import json

# from upernet_swin import UperNet_swinù
from upernet_swin2 import UperNet_swin

# with torch.cuda.device(0):
  # net = models.densenet161()
  #macs, params = get_model_complexity_info(net, (3, 224, 224), as_strings=True,
  #                                         print_per_layer_stat=True, verbose=True)
  # print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
  # print('{:<30}  {:<8}'.format('Number of parameters: ', params))



with open('/home/veronica/Scrivania/RSIm/Fusion/Method_1/config.json', 'r') as f:
  config = json.load(f)

import torch
from ptflops import get_model_complexity_info

model = UperNet_swin(params = config,
                     num_classes=config["num_classes"]).to("cuda")

print(model)

import torch
import torchvision.models as models
 # Replace with the correct import

# Initialize your model
# model = UperNet_swin()

# Create example input tensors for im1, im2, and im3
im1 = torch.randn(8, 3, 224, 224)  # Example input shape for im1
im2 = torch.randn(8, 4, 224, 224)  # Example input shape for im2
im3 = torch.randn(8, 1, 224, 224)  # Example input shape for im3

# Combine them into a single tuple to represent x
x = (im1, im2, im3)

input_res = (8, 256, 256)

# Calculate FLOPs and parameters using ptflops
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(model, input_res, as_strings=True,
                                             print_per_layer_stat=True, verbose=True)

print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
print('{:<30}  {:<8}'.format('Number of parameters: ', params))


