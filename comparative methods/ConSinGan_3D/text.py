import torch as torch

from SinGan import models
import os
import torch

def init_G(opt):
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    return netG


def init_D(opt):
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    return netD
import functions as functions
from config import get_arguments
parser = get_arguments()
opt = parser.parse_args()
p1 = '/home1/Usr/shentong/lab_code/SinGan/delta_75/2023_02_06_22_22_20_generation_train_depth_1_lr_scale_0.1_act_lrelu_0.05'
g_curr=init_G(opt)
for i in range(0,8):
    path=os.path.join(p1,str(i),'netG.pth')
    print(path)
    gg = torch.load(path)
    g_curr.load_state_dict(gg, strict=False)
  #  generator.init_next_stage(scale_num)  # 加前一尺度模型



