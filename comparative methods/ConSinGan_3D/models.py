import torch
import torch.nn as nn
import copy


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('Norm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_activation(opt):
    activations = {"lrelu": nn.LeakyReLU(opt.lrelu_alpha, inplace=True),
                   "elu": nn.ELU(alpha=1.0, inplace=True),
                   "prelu": nn.PReLU(num_parameters=1, init=0.25),
                   "selu": nn.SELU(inplace=True)
                   }
    return activations[opt.activation]


def upsample(x, size):
    x_up = torch.nn.functional.interpolate(x, size=size, mode='trilinear', align_corners=True)
    return x_up


class Conv3dBlock(nn.Sequential):
    def __init__(self, in_channel, out_channel, ker_size, padd, opt, generator=False):
        super(Conv3dBlock, self).__init__()
        self.add_module('conv3d',nn.Conv3d(in_channel, out_channel, kernel_size=ker_size, stride=1, padding=padd))
        if generator and opt.batch_norm:
            self.add_module('norm3d',nn.BatchNorm3d(out_channel))
        self.add_module(opt.activation, get_activation(opt))


class Discriminator(nn.Module):
    def __init__(self, opt):
        super(Discriminator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)
        self.pad_ = nn.ConstantPad3d((0,0,0,0,2,2),0)
        self.head = Conv3dBlock(opt.nc_im, N, (1,opt.ker_size,opt.ker_size), padd=opt.padd_size,opt= opt)

        self.body = nn.Sequential()
        for i in range(opt.num_layer):
            block = Conv3dBlock(N, N, (1,opt.ker_size,opt.ker_size), padd=opt.padd_size, opt=opt)
            self.body.add_module('block%d' % (i), block)
        self.tail = nn.Conv3d(N, 1, kernel_size=(1,opt.ker_size,opt.ker_size), padding=opt.padd_size)

    def forward(self, x):
        # pad = self.pad_(x)
        head = self.head(x)
        body = self.body(head)
        out = self.tail(body)
        return out


class GrowingGenerator(nn.Module):
    def __init__(self, opt):
        super(GrowingGenerator, self).__init__()

        self.opt = opt
        N = int(opt.nfc)

        # self._pad = nn.ZeroPad3d((1,1,1,1,0,0))
        self._pad = nn.ConstantPad3d((1,1,1,1,0,0),0)
        # self._pad_block = nn.ZeroPad3d((opt.num_layer - 1,opt.num_layer - 1,opt.num_layer - 1,opt.num_layer - 1,0,0)) if opt.train_mode == "generation" \
        #     else nn.ZeroPad3d((opt.num_layer,opt.num_layer,opt.num_layer,opt.num_layer,0,0))
        self._pad_block = nn.ConstantPad3d((opt.num_layer - 1, opt.num_layer - 1, opt.num_layer - 1, opt.num_layer - 1, 0,0),0) if opt.train_mode == "generation" \
            else nn.ConstantPad3d((opt.num_layer, opt.num_layer, opt.num_layer, opt.num_layer, 0, 0),0)

        self.head = Conv3dBlock(opt.nc_im, N, (1,opt.ker_size,opt.ker_size), opt.padd_size, opt, generator=True)
        self.body = torch.nn.ModuleList([])
        _first_stage = nn.Sequential()
        for i in range(opt.num_layer):
            block = Conv3dBlock(N, N, (1,opt.ker_size,opt.ker_size), opt.padd_size, opt, generator=True)
            _first_stage.add_module('block%d'%(i),block)
        self.body.append(_first_stage)
        self.tail = nn.Sequential(
            nn.Conv3d(N, opt.nc_im, kernel_size=(1,opt.ker_size,opt.ker_size), padding=opt.padd_size),
            nn.Tanh())

    def init_next_stage(self,current_scalenum):
        self.body.append(copy.deepcopy(self.body[-1]))

    def forward(self, noise, real_shapes, noise_amp):
        x = self.head(self._pad(noise[0]))
        if self.opt.train_mode == 'generation':
            x = upsample(x, size=[x.shape[2], x.shape[3] + 2, x.shape[4]+2])
        x = self._pad_block(x)
        x_prev_out = self.body[0](x)
        for idx, block in enumerate(self.body[1:], 1):
            if self.opt.train_mode == "generation":
                x_prev_out_1 = upsample(x_prev_out, size=[real_shapes[idx][2], real_shapes[idx][3], real_shapes[idx][4]])
                x_prev_out_2 = upsample(x_prev_out, size=[real_shapes[idx][2],
                                                          real_shapes[idx][3] + self.opt.num_layer*2,
                                                          real_shapes[idx][4] + self.opt.num_layer*2])
                x_prev = block(x_prev_out_2 + noise[idx] * noise_amp[idx])
            else:
                x_prev_out_1 = upsample(x_prev_out, size=real_shapes[idx][2:])
                x_prev = block(self._pad_block(x_prev_out_1+noise[idx]*noise_amp[idx]))
            x_prev_out = x_prev + x_prev_out_1
        out = self.tail(self._pad(x_prev_out))
        return out
