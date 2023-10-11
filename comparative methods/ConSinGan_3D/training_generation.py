import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from tqdm import tqdm
from cpu_gpu_mem_rate import linux_monitor
import functions
import models as models

def train(opt):
    print("Training model with the following parameters:")
    print("\t number of stages: {}".format(opt.train_stages))
    print("\t number of concurrently trained stages: {}".format(opt.train_depth))
    print("\t learning rate scaling: {}".format(opt.lr_scale))
    print("\t non-linearity: {}".format(opt.activation))
#    import ipdb;ipdb.set_trace()
    real = functions.read_image(opt)
    #import ipdb;ipdb.set_trace()
    print(real.shape)
    print('1 ok')
    real = functions.adjust_scale2image(real,opt)
    print('2 ok')
    reals = functions.create_reals_pyramid(real,opt)
    print("Training on image pyramid: {}".format([r.shape for r in reals]))

    generator = init_G(opt)
    fixed_noise = []
    noise_amp = []

    opt.out_ = functions.generate_dir2save(opt)
    f = open(os.path.join(opt.out_, "cpu_gpu_mem.csv"), "w")
    f.write('time,cpu,menory,gpu' + "\n")

    for scale_num in range(opt.stop_scale+1):
        opt.out_ = functions.generate_dir2save(opt)
        opt.outf = '%s/%d' %(opt.out_, scale_num)
        try:
            os.makedirs(opt.outf)
        except OSError:
            print(OSError)
            pass
        try:
            functions.save_image('{}/real_scale.npy'.format(opt.outf), reals[scale_num])
        except Exception as e:
            print(str(e))
            import ipdb;ipdb.set_trace()

        d_curr = init_D(opt)

        if scale_num >0 :
            gg = torch.load('%s/%d/netD.pth' % (opt.out_,scale_num-1))
            d_curr.load_state_dict(gg, strict = False)
            generator.init_next_stage(scale_num)#加前一尺度模型

        writer = SummaryWriter(log_dir=opt.outf)
        fixed_noise, noise_amp, generator, d_curr = train_single_scale(d_curr, generator, reals, fixed_noise, noise_amp, opt, scale_num, writer,f)

        torch.save(fixed_noise, '%s/fixed_noise.pth' % (opt.out_))
        torch.save(generator, '%s/G.pth' % (opt.out_))
        torch.save(reals, '%s/reals.pth' % (opt.out_))
        torch.save(noise_amp, '%s/noise_amp.pth' % (opt.out_))
        del d_curr
    writer.close()
    return

def train_single_scale(netD, netG, reals, fixed_noise, noise_amp, opt, depth, writer,f):
    reals_shapes = [real.shape for real in reals]
    real = reals[depth]
    alpha = opt.alpha

    if depth == 0:
        z_opt = reals[0]
    else:
        if opt.train_mode == 'generation':
            z_opt = functions.generate_noise([opt.nfc,
                                      reals_shapes[depth][2],
                                      reals_shapes[depth][3]+opt.num_layer*2,
                                      reals_shapes[depth][4]+opt.num_layer*2],
                                      device=opt.device)
        else:
            z_opt = functions.generate_noise([opt.nfc, reals_shapes[depth][2], reals_shapes[depth][3]],
                                             device=opt.device).detach()
    fixed_noise.append(z_opt.detach())

    optimizerD = optim.Adam(netD.parameters(), lr=opt.lr_d, betas=(opt.beta1,0.999))

    for block in netG.body[:-opt.train_depth]:
        for param in block.parameters():
            param.requires_grad = False

    parameter_list = [{"params": block.parameters(), "lr": opt.lr_g * (opt.lr_scale**(len(netG.body[-opt.train_depth:])-1-idx))}
               for idx, block in enumerate(netG.body[-opt.train_depth:])]

    if depth - opt.train_depth <0:
        parameter_list += [{"params": netG.head.parameters(), "lr": opt.lr_g * (opt.lr_scale ** depth)}]
    parameter_list += [{"params": netG.tail.parameters(), "lr": opt.lr_g}]
    optimizerG = optim.Adam(parameter_list, lr = opt.lr_g, betas=(opt.beta1, 0.999))

    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerD, milestones=[0.8*opt.niter], gamma=opt.gamma)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizerG, milestones=[0.8*opt.niter], gamma=opt.gamma)

    if depth == 0:
        noise_amp.append(1)
    else:
        noise_amp.append(0)
        z_reconstruction = netG(fixed_noise, reals_shapes, noise_amp)
        criterion = nn.MSELoss()
        rec_loss = criterion(z_reconstruction, real)
        RMSE = torch.sqrt(rec_loss).detach()
        _noise_amp = opt.noise_amp_init*RMSE
        noise_amp[-1] = _noise_amp

    _iter = tqdm(range(opt.niter))
    for iter in _iter:
        _iter.set_description('stage [{}/{}]:'.format(depth, opt.stop_scale))
        if iter%500==0:
            line = linux_monitor(0.001)
            print(line)
            f.write(line + "\n")
        noise = functions.sample_random_noise(depth, reals_shapes, opt)

        for j in range(opt.Dsteps):
            netD.zero_grad()
            output = netD(real)
            errD_real = -output.mean()

            if j == opt.Dsteps - 1:
                fake = netG(noise, reals_shapes, noise_amp)
            else:
                with torch.no_grad():
                    fake = netG(noise, reals_shapes, noise_amp)
            output = netD(fake.detach())
            errD_fake = output.mean()

            gradient_penalty = functions.calc_gradient_penalty(netD, real, fake, opt.lambda_grad, opt.device)
            errD_total = errD_real + errD_fake + gradient_penalty
            errD_total.backward()
            optimizerD.step()

        output = netD(fake)
        errG = -output.mean()

        if alpha != 0:
            loss = nn.MSELoss()
            rec = netG(fixed_noise, reals_shapes, noise_amp)
            rec_loss = alpha * loss(rec, real)
        else:
            rec_loss = 0

        netG.zero_grad()
        errG_total = errG + rec_loss
        errG_total.backward()
        for _ in range(opt.Gsteps):
            optimizerG.step()
        # if iter % 250 == 0 or iter+1 == opt.niter:
        #     writer.add_scalar('Loss/train/D/real/{}'.format(j), -errD_real.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/fake/{}'.format(j), errD_fake.item(), iter+1)
        #     writer.add_scalar('Loss/train/D/gradient_penalty/{}'.format(j), gradient_penalty.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/gen', errG.item(), iter+1)
        #     writer.add_scalar('Loss/train/G/reconstruction', rec_loss.item(), iter+1)
        #

        if iter % 500 == 0 or iter+1 == opt.niter:
            
            functions.save_image('{}/fake_sample_{}.tif'.format(opt.outf, iter+1), fake.detach())
            functions.save_image('{}/reconstruction_{}.tif'.format(opt.outf, iter+1), rec.detach())
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)
        elif depth==3 and iter % 50 == 0:
            
            functions.save_image('{}/fake_sample_{}.tif'.format(opt.outf, iter+1), fake.detach())
            functions.save_image('{}/reconstruction_{}.tif'.format(opt.outf, iter+1), rec.detach())
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)
       # import ipdb;ipdb.set_trace()
        if iter % 1000 ==0 or iter+1 == opt.niter:
            print(iter)
            generate_samples(netG, opt, depth, noise_amp, writer, reals, iter+1)
        schedulerD.step()
        schedulerG.step()
    functions.save_networks(netG, netD, z_opt, opt)
    # print('canshu:'.format(netG.body[depth].parameters))
    return fixed_noise, noise_amp, netG, netD


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
def generate_samples(netG, opt, depth, noise_amp, writer, reals, iter, n=10):
    opt.out_ =  functions.generate_dir2save(opt)
    dir2save =  '{}/gen_samples_stage_{}'.format(opt.out_, depth)
    reals_shapes = [r.shape for r in reals]
    all_images = []
    try:
        os.makedirs(dir2save)
    except OSError:
        pass
    with torch.no_grad():
        for idx in range(n):
            noise = functions.sample_random_noise(depth,reals_shapes,opt)
            sample = netG(noise,reals_shapes, noise_amp)
            all_images.append(sample)
    
          #  show_img(sample.detach().cpu().numpy()[0][0][0])
            functions.save_image('{}/gen_sample_{}'.format(dir2save, idx), sample.detach())


def init_G(opt):
    netG = models.GrowingGenerator(opt).to(opt.device)
    netG.apply(models.weights_init)
    return netG

def init_D(opt):
    netD = models.Discriminator(opt).to(opt.device)
    netD.apply(models.weights_init)
    return netD