import math

from skimage import io as img
from skimage import color, morphology, filters
import os
import random
import datetime
import dateutil.tz
import torch
import numpy as np
import copy
from albumentations import HueSaturationValue, IAAAdditiveGaussianNoise, GaussNoise, OneOf,\
    Compose, MultiplicativeNoise, ToSepia, ChannelDropout, ChannelShuffle, Cutout, InvertImg
from tifffile import tifffile

from imresize import imresize

# def norm(x):
#     out = (x - 0.5) * 2
#     return out.clamp(-1, 1)
#
# def denorm(x):
#     out = (x + 1) / 2
#     return out.clamp(0, 1)

def norm(x):
    out = (x-1)
    out = torch.where(out==torch.tensor(0.0,device='cuda:0'), torch.tensor(0.01,device='cuda:0'),out)
    return out.clamp(-1,1)

def denorm(x):
    out = (x+1)
    out = torch.where(out==torch.tensor(1.01,device='cuda:0'),torch.tensor(1.0,device='cuda:0'),out)
    return out

def read_image(opt):
    f = open('%s' % (opt.input_name))
    data = f.readlines()
    data_size = data[0].replace('\n','').split(' ')
    data_size = list(map(eval, data_size))
    x1=np.loadtxt(opt.input_name,skiprows=3)

    x = x1.reshape((data_size[2],data_size[0],data_size[1],1))
    # x[x==2]=150
    # x[x==1]=255
    # print(x.shape)
    # show_img(x[0])
    x = np2torch(x,opt)
    return x

# def read_image(opt):
#     x = img.imread('%s' % (opt.input_name))
#     x = np2torch(x,opt)
#     return x

def move_to_gpu(t):
    if torch.cuda.is_available():
        t =t.to(torch.device('cuda'))
    return t

def move_to_cpu(t):
    t =t.to(torch.device('cpu'))
    return t

def np2torch(x, opt):
    if opt.nc_im == 3:
        x = x[:,:,:,None]
        x = x.transpose((3, 2, 0, 1))/255
    else:
        # x = color.rgb2gray(x)
        x = x[:,:,:,:,None]
        x = x.transpose(4, 3,0 ,1,2)
    x = torch.from_numpy(x)
    if not(opt.not_cuda):
        x = move_to_gpu(x)
    x = x.type(torch.cuda.FloatTensor) if not(opt.not_cuda) else x.type(torch.FloatTensor)
    x = norm(x)
    return x

def torch2uint8(x):
    x = x[0,:,:,:,:]
    x = x.permute((1,2,3,0))
    x = denorm(x)
    try:
        x = x.cpu().numpy()
    except:
        x = x.detach().cpu().numpy()
    x = x.astype(np.uint8)
    return x

def save_networks(netG, netDs, z, opt):
    torch.save(netG.state_dict(), '%s/netG.pth' % (opt.outf))
    if isinstance(netDs, list):
        for i, netD in enumerate(netDs):
            torch.save(netD.state_dict(), '%s/netD_%s.pth' % (opt.outf, str(i)))
    else:
        torch.save(netDs.state_dict(), '%s/netD.pth' % (opt.outf))
    torch.save(z, '%s/z_opt.pth' % (opt.outf))

def adjust_scale2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([ real_.shape[3], real_.shape[4]]), 1)
    real = imresize(real_, opt.scale1, opt)
    opt.stop_scale = opt.train_stages - 1
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[3],real.shape[4])), 1/opt.stop_scale)
    return real
def adjust_scales2image(real_, opt):
    opt.scale1 = min(opt.max_size / max([real_.shape[2], real_.shape[3], real_.shape[4]]),1)
    # print(opt.scale1)
    real = imresize(real_, opt.scale1, opt)
    opt.stop_scale = opt.train_stages - 1
    # print(opt.stop_scale)
    opt.scale_factor = math.pow(opt.min_size / (min(real.shape[2], real.shape[3], real.shape[4])), 1 / opt.stop_scale)
    # print(opt.scale_factor)
    return real
def create_reals_pyramid(real,opt):
    reals = []
    for i in range(opt.stop_scale):
        scale = math.pow(opt.scale_factor, ((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
        curr_real = imresize(real, scale, opt)
        reals.append(curr_real)
    reals.append(real)
    return reals


def create_reals_pyramid(real, opt):
    reals = []
    # use old rescaling method for harmonization
    # if opt.train_mode == "harmonization":
    #     for i in range(opt.stop_scale):
    #         scale = math.pow(opt.scale_factor, opt.stop_scale - i)
    #         curr_real = imresize(real, scale, opt)
    #         reals.append(curr_real)
    # # use new rescaling method for all other tasks
    # else:
    for i in range(opt.stop_scale):
        scale = math.pow(opt.scale_factor,((opt.stop_scale-1)/math.log(opt.stop_scale))*math.log(opt.stop_scale-i)+1)
        # print(real.shape)
        curr_real = imresize(real,scale,opt)
       # print(curr_real.shape)
        reals.append(curr_real)
    reals.append(real)
    return reals

def generate_dir2save(opt):
    training_image_name = opt.input_name[:-4].split("/")[-1]
    #dir2save = 'TrainedModels/\{}/'.format(training_image_name)
    dir2save = '{}/'.format(training_image_name)
    dir2save += opt.timestamp
    dir2save += "_{}".format(opt.train_mode)
    if opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            dir2save += "_{}".format("fine-tune")
    dir2save += "_train_depth_{}_lr_scale_{}".format(opt.train_depth, opt.lr_scale)
    if opt.batch_norm:
        dir2save += "_BN"
    dir2save += "_act_" + opt.activation
    if opt.activation == "lrelu":
        dir2save += "_" + str(opt.lrelu_alpha)

    return dir2save


def show_img(input_np, input_np2=None, name=None):
    import matplotlib.pyplot as plt
    try:
        k1 = input_np2.any()
    except Exception as e:
        print(str(e))
        k1 = None
    if k1:
        plt.subplot(1, 2, 1)
        plt.imshow(input_np)
        if name:
            plt.title(name)  # 图像题目
        plt.subplot(1, 2, 2)
        plt.imshow(input_np2)
    #      plt.title(name)  # 图像题目
    else:
        plt.imshow(input_np)
        plt.axis('on')  # 关掉坐标轴为 off
        if name:
            plt.title(name)  # 图像题目
        
        # 必须有这个，要不然无法显示
    plt.show()
def save_image(name,image):
    c1=(convert_image_np(image)*255).astype(np.uint8)
    if 'fake_sam'  in name:
        show_img(c1[0])
      
    # print(c1[0][0])
    #import ipdb;ipdb.set_trace()
    tifffile.imsave(name,c1)

def convert_image_np(inp):
    inp = denorm(inp)
    inp = move_to_cpu(inp[-1,-1,:,:,:])
    inp = inp.numpy()
    inp = np.clip(inp,0,2)
    return inp

def sample_random_noise(depth, reals_shapes, opt):
    noise = []
    for d in range(depth + 1):
        if d == 0:
            noise.append(generate_noise([opt.nc_im, reals_shapes[d][2], reals_shapes[d][3], reals_shapes[d][4]],
                                         device=opt.device).detach())
        else:
            if opt.train_mode == "generation":
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2],
                                             reals_shapes[d][3] + opt.num_layer * 2,
                                             reals_shapes[d][4] + opt.num_layer * 2],
                                             device=opt.device).detach())

            else:
                noise.append(generate_noise([opt.nfc, reals_shapes[d][2], reals_shapes[d][3], reals_shapes[d][4]],
                                             device=opt.device).detach())

    return noise

def calc_gradient_penalty(netD, real_data, fake_data, LAMBDA, device):
    MSGGan = False
    if MSGGan:
        alpha = torch.rand(1, 1)
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = [alpha * rd + ((1 - alpha) * fd) for rd, fd in zip(real_data, fake_data)]
        interpolates = [i.to(device) for i in interpolates]
        interpolates = [torch.autograd.Variable(i, requires_grad=True) for i in interpolates]

        disc_interpolates = netD(interpolates)
    else:
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.to(device)  # cuda() #gpu) #if use_cuda else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)
        interpolates = interpolates.to(device)  # .cuda()
        interpolates = torch.autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                    # .cuda(), #if use_cuda else torch.ones(
                                    # disc_interpolates.size()),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # LAMBDA = 1
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty


def post_config(opt):
    # init fixed parameters
    opt.device = torch.device("cpu" if opt.not_cuda else "cuda:{}".format(opt.gpu))
    opt.noise_amp_init = opt.noise_amp
    opt.timestamp = datetime.datetime.now(dateutil.tz.tzlocal()).strftime('%Y_%m_%d_%H_%M_%S')

    if opt.manualSeed is None:
        opt.manualSeed = random.randint(1, 10000)
    random.seed(opt.manualSeed)
    torch.manual_seed(opt.manualSeed)

    if torch.cuda.is_available() and opt.not_cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return opt


def load_config(opt):
    if not os.path.exists(opt.model_dir):
        print("Model not found: {}".format(opt.model_dir))
        exit()

    with open(os.path.join(opt.model_dir, 'parameters.txt'), 'r') as f:
        params = f.readlines()
        for param in params:
            param = param.split("-")
            param = [p.strip() for p in param]
            param_name = param[0]
            param_value = param[1]
            try:
                param_value = int(param_value)
            except ValueError:
                try:
                    param_value = float(param_value)
                except ValueError:
                    pass
            setattr(opt, param_name, param_value)
    return opt

def generate_noise(size,num_samp=1,device='cuda',type='gaussian', scale=1):
    if type == 'gaussian':
        noise = torch.randn(num_samp,size[0],round(size[1]/scale),round(size[2]/scale), round(size[3]/scale), device=device)
        noise = upsampling(noise,size[1],size[2],size[3])
    else:
        NotImplementedError
    return noise

def upsampling(im,sz,sx,sy):
    m = torch.nn.Upsample(size=[round(sz),round(sx), round(sy)], mode='trilinear', align_corners=True)
    return m(im)



