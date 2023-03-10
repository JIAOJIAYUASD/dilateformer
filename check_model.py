import torch
from fvcore.nn import FlopCountAnalysis
from fvcore.nn import flop_count_table
from models import *

model = dilateformer_tiny()
model.eval()
flops = FlopCountAnalysis(model, torch.rand(1, 3, 224, 224))
print(flop_count_table(flops))
