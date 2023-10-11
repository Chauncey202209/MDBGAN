import numpy as np
import matplotlib.pyplot as plt
import tifffile

from SinGan_3D import models


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
def read_npy():
    # path = r'TrainedModels/delta/2020_10_13_21_06_33_generation_train_depth_3_lr_scale_0.1_act_lrelu_0.05/gen_samples_stage_0/gen_sample_1.npy'
   # path=r'C:\Users\st\Desktop\管其杰代码\代码\浠ｇ爜\ConSinGAN3D\ConSinGAN3D\TrainedModels\ti_64\2021_03_04_19_23_30_generation_train_depth_1_lr_scale_0.1_act_lrelu_0.05\gen_samples_stage_4\gen_sample_0.npy'
 #   path=r'/home1/Usr/shentong/lab_code/SinGan/delta/2023_01_30_00_39_57_generation_train_depth_1_lr_scale_0.1_act_lrelu_0.05/gen_samples_stage_7/gen_sample_9.npy'
    path=r'/home1/Usr/shentong/lab_code/SinGan/delta/2023_01_30_00_39_57_generation_train_dept' \
         r'h_1_lr_scale_0.1_act_lrelu_0.05/gen_samples_stage_7/gen_sample_1.npy'
    x = np.load(path)
    x=x/2
  #  show_img(x[0])
    q1=x*255
    for i in range(q1.shape[0]):
        for m in range(q1.shape[1]):
            for n in range(x.shape[2]):
                if q1[i][m][n]<20:
                    q1[i][m][n]=0
                elif q1[i][m][n]>=20 and q1[i][m][n]<=180 :
                    q1[i][m][n]=255
                elif q1[i][m][n]>=180 :
                    q1[i][m][n]=150
    show_img(q1[1])
    #tifffile.imsave('SinGan.tiff',q1)
    import ipdb;ipdb.set_trace()
    # f = open(r'final_img','w+')
    # f.write(str(x.shape)+'\n')
    # f.write('1\nfacies\n')
    # for i in range(x.shape[0]):
    #     for m in range(x.shape[1]):
    #         for n in range(x.shape[2]):
    #             if x[i,m,n]<=1:
    #                 f.write('0\n')
    #             # elif x[i,m,n]>0.5 and x[i,m,n]<1.5:
    #             #     f.write('1\n')
    #             else:
    #                 f.write('1\n')
    #
    # print(x.shape)
    # print(x)
    # plt.imshow(x[2])
    # r1=plt.imread(r'C:\Users\st\Desktop\管其杰代码\代码\浠ｇ爜\ConSinGAN3D\ConSinGAN3D\data\channel250.png')
    # # plt.imshow(r1)
    # print(r1)
    #
    # plt.axis('off')
    # plt.show()



if __name__ == '__main__':
    read_npy()