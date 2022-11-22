import os
import time
import torch
from torch import nn
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import requests
from io import BytesIO
from model.modules import Unet2D
import open_clip
from open_clip import tokenizer
from rudalle import get_vae
from einops import rearrange

import math
import queue



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


x = torch.randint(0, 1024, (1, 32, 32)).long().to(device)
c = torch.randn((1, 1024)).to(device)
r = torch.rand(1).to(device)