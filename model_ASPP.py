import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair
from module import create_featex, ASPP
import load_weights
from module.unet.__init__ import *

# ./pretrained_weights/peakdice2.pt
# 0.78 AUC
class ManTraNet(nn.Module):
    def __init__(self, Featex, pool_size_list=[7,15,31], is_dynamic_shape=True, apply_normalization=True, mid_channel = 3):
        super().__init__()
        self.Featex = Featex # Mantranet前半段
        self.aspp = ASPP.ASPP(3)
        self.unet = UNet(mid_channel, 1)
    def forward(self,x):
        rf = self.Featex(x) 
        pp = self.aspp(rf)
        pred_out = self.unet(pp)
        pred_out = torch.sigmoid(pred_out)
        return pred_out

def create_model(IMC_model_idx, freeze_featex, window_size_list=[7,15,31]):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    type_idx = IMC_model_idx if IMC_model_idx < 4 else 2
    Featex = create_featex.Featex_vgg16_base(type_idx)
    if freeze_featex:
        print("INFO: freeze feature extraction part, trainable=False")
        for p in Featex.parameters():
            p.requires_grad = False
    else:
        print ("INFO: unfreeze feature extraction part, trainable=True")

    model = ManTraNet(Featex, pool_size_list=window_size_list, is_dynamic_shape=True, apply_normalization=True)
    return model


def model_load_weights(weight_path, model, Featex_only=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = load_weights.load_weights(weight_path, model, Featex_only)    
    model.to(device)

    return model





