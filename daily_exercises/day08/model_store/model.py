import torchvision
import torch
from time import perf_counter
import numpy as np
import torch.nn.functional as F

# Lets make a resnet model
model = torchvision.models.resnet18(pretrained = True)

# Script model
script_model = torch.jit.script(model)
script_model.save('deployable_model.pt')

# Let's make some random input
x = torch.rand(1,3,224,224) # 3 colors, 224 x 224 px


unscripted_top5_indices = F.softmax(model(x), dim=1).topk(5).indices

scripted_top5_indices = F.softmax(script_model(x), dim=1).topk(5).indices

assert torch.allclose(unscripted_top5_indices, scripted_top5_indices)

# I ran following in WSL prompt:
# curl http://127.0.0.1:8080/predictions/resnet18_model -T exercise_files/my_cat.jpg
# {
#   "whippet": 0.18087074160575867,
#   "weasel": 0.08569776266813278,
#   "black-footed_ferret": 0.07569919526576996,
#   "Cardigan": 0.06711097061634064,
#   "timber_wolf": 0.05724381282925606
# }