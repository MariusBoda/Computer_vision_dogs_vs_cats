from collections import defaultdict
from glob import iglob
from pathlib import Path

import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np

DEVICE = torch.device('gpu')
OUTPUT_SIZE = 2048

model = models.resnext50_32x4d(weights = models.ResNeXt50_32X4D_Weights.IMAGENET1K_V2)