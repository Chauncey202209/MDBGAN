import time

import torch
import os
import os.path as osp
from shutil import copyfile, copytree
import glob
import time
from config import get_arguments
import functions as functions

from training_generation import train
if __name__ == '__main__':
    parser = get_arguments()
    # parser.add_argument('--input_name', help='input image name for training', required=True)
    #parser.add_argument('--input_name', help='input image name for training',
    # default='/home/lab209-3/Alldata/morooi/ConSinGAN3D/SGEMS/ti_64.txt')
   # parser.add_argument('--input_name', help='input image name for training',default='/home1/Usr/shentong/lab_code/SinGan/con_delta_250.txt')
    parser.add_argument('--input_name', help='input image name for training',default='/home1/Usr/shentong/lab_code/SinGan_3D/delta_225.txt')
    parser.add_argument('--naive_img', help='naive input image  (harmonization or editing)', default="")
    parser.add_argument('--gpu', type=int, help='which GPU to use', default=0)
    parser.add_argument('--train_mode', default='generation',
                        choices=['generation', 'retarget', 'harmonization', 'editing'],
                        help="generation, retarget, harmonization, editing")
    
    parser.add_argument('--lr_scale', type=float, help='scaling of learning rate for lower stages', default=0.2)
    parser.add_argument('--train_stages', type=int, help='how many stages to use for training', default=4)
    parser.add_argument('--niter', type=int, default=100, help='number of epochs to train per scale')
    parser.add_argument('--min_size', type=int, help='image minimal size at the coarser scale', default=32)
    parser.add_argument('--max_size', type=int, help='image minimal size at the coarser scale', default=225)
    parser.add_argument('--fine_tune', action='store_true', help='whether to fine tune on a given image', default=0)
    parser.add_argument('--model_dir', help='model to be used for fine tuning (harmonization or editing)', default="")

    opt = parser.parse_args()

    opt = functions.post_config(opt)

    if opt.fine_tune:
        _gpu = opt.gpu
        _model_dir = opt.model_dir
        _timestamp = opt.timestamp
        _naive_img = opt.naive_img
        _niter = opt.niter
        opt = functions.load_config(opt)
        opt.gpu = _gpu
        opt.model_dir = _model_dir
        opt.start_scale = opt.train_stages - 1
        opt.timestamp = _timestamp
        opt.fine_tune = True
        opt.naive_img = _naive_img
        opt.niter = _niter

    if not os.path.exists(opt.input_name):
        print("Image does not exist: {}".format(opt.input_name))
        print("Please specify a valid image.")
        exit()

    if torch.cuda.is_available():
        torch.cuda.set_device(opt.gpu)

    if opt.train_mode == "generation" or opt.train_mode == "retarget":
        from training_generation import *
    elif opt.train_mode == "harmonization" or opt.train_mode == "editing":
        if opt.fine_tune:
            if opt.model_dir == "":
                print("Model for fine tuning not specified.")
                print("Please use --model_dir to define model location.")
                exit()
            else:
                if not os.path.exists(opt.model_dir):
                    print("Model does not exist: {}".format(opt.model_dir))
                    print("Please specify a valid model.")
                    exit()
            if not os.path.exists(opt.naive_img):
                print("Image for harmonization/editing not found: {}".format(opt.naive_img))
                exit()
       # from ConSinGAN_3D.training_harmonization_editing import *

    dir2save = functions.generate_dir2save(opt).replace('TrainedModels/','')

    if osp.exists(dir2save):
        print('Trained model already exist: {}'.format(dir2save))
        exit()

        # create log dir
    try:
        os.makedirs(dir2save)
    except OSError:
        pass

        # save hyperparameters and code files
    with open(osp.join(dir2save, 'parameters.txt'), 'w') as f:
        for o in opt.__dict__:
            f.write("{}\t-\t{}\n".format(o, opt.__dict__[o]))
    current_path = os.path.dirname(os.path.abspath(__file__))
    # for py_file in glob.glob(osp.join(current_path, "*.py")):
    #     copyfile(py_file, osp.join(dir2save, py_file.split("\\")[-1]))
    # copytree(osp.join(current_path, "ConSinGAN3D"), osp.join(dir2save, "ConSinGAN3D"))
    # train model
    print("Training model ({})".format(dir2save))
    # opt.max_size = 199
    # opt.niter = 1000
    start = time.time()
    train(opt)

    end = time.time()
    elapsed_time = end - start
    print("Time for training: {} seconds".format(elapsed_time))