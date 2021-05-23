import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
import math
#device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device='cuda'
class CorrBlock:
    def __init__(self, fmap1, fmap2,num_levels=4, radius=4):
        #block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,block12, block13, block14, block15, block16

        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []

        # all pairs correlation
        corr = CorrBlock.corr(fmap1, fmap2)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch*h1*w1, dim, h2, w2)
        
        self.corr_pyramid.append(corr)
        for i in range(self.num_levels-1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2*r+1)
            dy = torch.linspace(-r, r, 2*r+1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch*h1*w1, 1, 1, 2) / 2**i
            delta_lvl = delta.view(1, 2*r+1, 2*r+1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()


    @staticmethod
    def corr(fmap1, fmap2):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht*wd)
        fmap2 = fmap2.view(batch, dim, ht*wd)

        corr = torch.matmul(fmap1.transpose(1,2), fmap2)
        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr  / torch.sqrt(torch.tensor(dim).float())


    # def corr(fmap1, fmap2, region1, region2, region3, region4, region5, region6):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # fmap1 = fmap1.transpose(1, 2)
    #     #  4份窄矩形       阈值设置为5
    #     # 将相关区域分为强相关和弱相关
    #
    #     region1[:, :, 0:3 * ht // 6, 0: wd] = fmap2[:, :, 0:3 * ht // 6, 0: wd]
    #     region1[:, :, 3 * ht // 6:5 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 3 * ht // 6:5 * ht // 6, 0:wd]
    #
    #     region2[:, :, 0:4 * ht // 6, 0:wd] = fmap2[:, :, 0:4 * ht // 6, 0: wd]
    #     region2[:, :, 4 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 6:ht, 0: wd]
    #
    #     region3[:, :, 0:5 * ht // 6, 0:wd] = fmap2[:, :, 0:5 * ht // 6, 0:wd]
    #     region3[:, :, 5 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 5 * ht // 6:ht, 0:wd]
    #
    #     region4[:, :, 0:ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:ht // 6, 0:wd]
    #     region4[:, :, ht // 6:ht, 0:wd] = fmap2[:, :, ht // 6:ht, 0:wd]
    #
    #     region5[:, :, 0:2 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:2 * ht // 6, 0:wd]
    #     region5[:, :, 2 * ht // 6:ht, 0:wd] = fmap2[:, :, 2 * ht // 6:ht, 0:wd]
    #
    #     region6[:, :, ht // 6:3 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, ht // 6:3 * ht // 6, 0:wd]
    #     region6[:, :, 3 * ht // 6:ht, 0:wd] = fmap2[:, :, 3 * ht // 6:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #     region5 = region5.view(batch, dim, ht * wd)
    #     region6 = region6.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 6]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 6:2 * ht * wd // 6]
    #     r3_fmap1 = fmap1[:, :, 2 * ht * wd // 6:3 * ht * wd // 6]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 6:4 * ht * wd // 6]
    #     r5_fmap1 = fmap1[:, :, 4 * ht * wd // 6:5 * ht * wd // 6]
    #     r6_fmap1 = fmap1[:, :, 5 * ht * wd // 6:ht * wd]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #     corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
    #     corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6), dim=1)
    #
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())


    # def corr(fmap1, fmap2, region1, region2, region3, region4):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # KITTI4分条
    #     # 将相关区域分为强相关和弱相关
    #     region1[:, :, 0:ht // 2, 0:wd] = fmap2[:, :, 0:ht // 2, 0:wd]
    #     region1[:, :, ht // 2:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 2:3 * ht // 4, 0:wd]
    #
    #     region2[:, :, ht // 4 - 9:ht // 2 + 9, 0:wd] = fmap2[:, :, ht // 4 - 9:ht // 2 + 9, 0:wd]
    #
    #     region3[:, :, ht // 2 - 9:3 * ht // 4 + 9, 0:wd] = fmap2[:, :, ht // 2 - 9:3 * ht // 4 + 9, 0:wd]
    #
    #     region4[:, :, ht // 4:ht // 2, 0:wd] = 0.75 * fmap2[:, :, ht // 4:ht // 2, 0:wd]
    #     region4[:, :, ht // 2:ht, 0:wd] = fmap2[:, :, ht // 2:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 4]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 4:ht * wd // 2]
    #     r3_fmap1 = fmap1[:, :, ht * wd // 2:3 * ht * wd // 4]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 4:ht * wd]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4), dim=1)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())

    # def corr(fmap1, fmap2, region1, region2, region3, region4, region5, region6, region7, region8):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # fmap1 = fmap1.transpose(1, 2)
    #     #  4份窄矩形       阈值设置为5
    #     # 将相关区域分为强相关和弱相关
    #     #KITTI and sintel 8分条
    #     region1[:, :, 0:4 * ht // 8, 0: wd] = fmap2[:, :, 0:4 * ht // 8, 0: wd]
    #     region1[:, :, 4 * ht // 8:6 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd]
    #
    #     region2[:, :, 0:5 * ht // 8, 0: wd] = fmap2[:, :, 0:5 * ht // 8, 0: wd]
    #     region2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd]
    #
    #     region3[:, :, 0:5 * ht // 8, 0: wd] = fmap2[:, :, 0:5 * ht // 8, 0: wd]
    #     region3[:, :, 5 * ht // 8:7 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 5 * ht // 8:7 * ht // 8, 0:wd]
    #
    #     region4[:, :, 0:ht // 8, 0: wd] = 0.75 * fmap2[:, :, 0:ht // 8, 0: wd]
    #     region4[:, :, ht // 8:6 * ht // 8, 0:wd] = fmap2[:, :, ht // 8:6 * ht // 8, 0:wd]
    #     region4[:, :, 6 * ht // 8:ht, 0:wd] = 0.75 * fmap2[:, :, 6 * ht // 8:ht, 0:wd]
    #
    #     region5[:, :, 0:2 * ht // 8, 0: wd] = 0.75 * fmap2[:, :, 0:2 * ht // 8, 0: wd]
    #     region5[:, :, 2 * ht // 8:7 * ht // 8, 0:wd] = fmap2[:, :, 2 * ht // 8:7 * ht // 8, 0:wd]
    #     region5[:, :, 7 * ht // 8:ht, 0:wd] = 0.75 * fmap2[:, :, 7 * ht // 8:ht, 0:wd]
    #
    #     region6[:, :, ht // 8:3 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, ht // 8:3 * ht // 8, 0:wd]
    #     region6[:, :, 3 * ht // 8:ht, 0:wd] = fmap2[:, :, 3 * ht // 8:ht, 0:wd]
    #
    #     region7[:, :, ht // 8:3 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, ht // 8:3 * ht // 8, 0:wd]
    #     region7[:, :, 3 * ht // 8:ht, 0:wd] = fmap2[:, :, 3 * ht // 8:ht, 0:wd]
    #
    #     region8[:, :, 2 * ht // 8:4 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 8:4 * ht // 8, 0:wd]
    #     region8[:, :, 4 * ht // 8:ht, 0:wd] = fmap2[:, :, 4 * ht // 8:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #     region5 = region5.view(batch, dim, ht * wd)
    #     region6 = region6.view(batch, dim, ht * wd)
    #     region7 = region7.view(batch, dim, ht * wd)
    #     region8 = region8.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 8]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 8:2 * ht * wd // 8]
    #     r3_fmap1 = fmap1[:, :, 2 * ht * wd // 8:3 * ht * wd // 8]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 8:4 * ht * wd // 8]
    #     r5_fmap1 = fmap1[:, :, 4 * ht * wd // 8:5 * ht * wd // 8]
    #     r6_fmap1 = fmap1[:, :, 5 * ht * wd // 8:6 * ht * wd // 8]
    #     r7_fmap1 = fmap1[:, :, 6 * ht * wd // 8:7 * ht * wd // 8]
    #     r8_fmap1 = fmap1[:, :, 7 * ht * wd // 8:8 * ht * wd // 8]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #     corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
    #     corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)
    #     corr7 = torch.matmul(r7_fmap1.transpose(1, 2), region7)
    #     corr8 = torch.matmul(r8_fmap1.transpose(1, 2), region8)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6, corr7, corr8), dim=1)
    #
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())





    # def corr(fmap1, fmap2, region1, region2, region3, region4):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # 4分条
    #     # 将相关区域分为强相关和弱相关
    #     region1[:, :, 0:ht // 2, 0:wd] = fmap2[:, :, 0:ht // 2, 0:wd]
    #     region1[:, :, ht // 2:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 2:3 * ht // 4, 0:wd]
    #
    #     region2[:, :, 0:3*ht // 4, 0:wd] = fmap2[:, :, 0:3*ht // 4, 0:wd]
    #
    #     region3[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     region4[:, :, ht // 4:ht // 2, 0:wd] = 0.75 * fmap2[:, :, ht // 4:ht // 2, 0:wd]
    #     region4[:, :, ht // 2:ht, 0:wd] = fmap2[:, :, ht // 2:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 4]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 4:ht * wd // 2]
    #     r3_fmap1 = fmap1[:, :, ht * wd // 2:3 * ht * wd // 4]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 4:ht * wd]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4), dim=1)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())


    # def corr(fmap1, fmap2, block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,
    #          block12, block13, block14, block15, block16):
    #     batch, dim, ht, wd = fmap1.shape
    #     region1 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region2 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region3 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region4 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #
    #     region5 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region6 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region7 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region8 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #
    #     region9 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region10 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region11 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region12 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #
    #     region13 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region14 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region15 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     region16 = torch.zeros([batch, dim, math.ceil(ht / 4), wd]).to(device)
    #     # print(id(region1))
    #     # 第一行
    #     # region对应的是fmap1 而block对应的是fmap2
    #     region1[:, :, :, 0:wd // 4] = fmap1[:, :, 0:ht // 4, 0:wd // 4]
    #     region2[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 0:ht // 4, wd // 4:2 * wd // 4]
    #     region3[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 0:ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region4[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 0:ht // 4, 3 * wd // 4:wd]
    #
    #     region5[:, :, :, 0:wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, 0:wd // 4]
    #     region6[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, wd // 4:2 * wd // 4]
    #     region7[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region8[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, ht // 4:2 * ht // 4, 3 * wd // 4:wd]
    #
    #     region9[:, :, :, 0:wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 0:wd // 4]
    #     region10[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, wd // 4:2 * wd // 4]
    #     region11[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region12[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 3 * wd // 4:wd]
    #
    #     region13[:, :, :, 0:wd // 4] = fmap1[:, :, 3 * ht // 4:ht, 0:wd // 4]
    #     region14[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 3 * ht // 4:ht, wd // 4:2 * wd // 4]
    #     region15[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 3 * ht // 4:ht, 2 * wd // 4:3 * wd // 4]
    #     region16[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 3 * ht // 4:ht, 3 * wd // 4:wd]
    #
    #     test1 = region1.view(batch, dim, (ht // 4) * wd)
    #     test2 = region2.view(batch, dim, (ht // 4) * wd)
    #     test3 = region3.view(batch, dim, (ht // 4) * wd)
    #     test4 = region4.view(batch, dim, (ht // 4) * wd)
    #
    #     test5 = region5.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test6 = region6.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test7 = region7.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test8 = region8.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #
    #     test9 = region9.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test10 = region10.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test11 = region11.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test12 = region12.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #
    #     test13 = region13.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test14 = region14.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test15 = region15.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #     test16 = region16.view(batch, dim, (math.ceil(ht / 4)) * wd)
    #
    #     # 横向多给一些，比如3/4
    #     block1[:, :, 0:2 * ht // 4, 0:3 * wd // 4] = fmap2[:, :, 0:2 * ht // 4, 0:3 * wd // 4]
    #     block1[:, :, 2 * ht // 4:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0:wd]
    #     block1[:, :, 0:2 * ht // 4, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 3 * wd // 4:wd]
    #
    #     block2[:, :, 0:2 * ht // 4, 0:wd] = fmap2[:, :, 0:2 * ht // 4, 0:wd]
    #     block2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd]
    #     # block2[:, :, 0:2 * ht // 4, 3 * wd // 4: wd] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 3 * wd // 4: wd]
    #
    #     block3[:, :, 0:2 * ht // 4, 0: wd] = fmap2[:, :, 0:2 * ht // 4, 0: wd]
    #     block3[:, :, 2 * ht // 4:3 * ht // 4, 0: wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd]
    #     # block3[:, :, 0:2 * ht // 4, 0: wd // 4] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 0: wd // 4]
    #
    #     block4[:, :, 0:2 * ht // 4, wd // 4: wd] = fmap2[:, :, 0:2 * ht // 4, wd // 4: wd]
    #     block4[:, :, 2 * ht // 4:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0:wd]
    #     block4[:, :, 0:2 * ht // 4, 0:wd // 4] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 0:wd // 4]
    #
    #     block5[:, :, 0:3 * ht // 4, 0:3 * wd // 4] = fmap2[:, :, 0:3 * ht // 4, 0:3 * wd // 4]
    #     # block5[:, :, 3 * ht // 4:ht, 0:3 * wd // 4] = 0.75 * fmap2[:, :, 3 * ht // 4:ht, 0:3 * wd // 4]
    #     # block5[:, :, 0:3 * ht // 4, 2 * wd // 4:3 * wd // 4] = 0.75 * fmap2[:, :, 0:3 * ht // 4,2 * wd // 4:3 * wd // 4]
    #
    #     block6[:, :, 0:3 * ht // 4, 0:wd] = fmap2[:, :, 0:3 * ht // 4, 0:wd]
    #
    #     block7[:, :, 0:3 * ht // 4, 0:wd] = fmap2[:, :, 0:3 * ht // 4, 0:wd]
    #
    #     block8[:, :, 0:3 * ht // 4, wd // 4:wd] = fmap2[:, :, 0:3 * ht // 4, wd // 4:wd]
    #     # block8[:, :, 3 * ht // 4:ht, wd // 4:wd] = 0.75 * fmap2[:, :, 3 * ht // 4:ht, wd // 4:wd]
    #     # block8[:, :, 0:3 * ht // 4, wd // 4:2 * wd // 4] = 0.75 * fmap2[:, :, 0:3 * ht // 4, wd // 4:2 * wd // 4]
    #
    #     block9[:, :, ht // 4:ht, 0:3 * wd // 4] = fmap2[:, :, ht // 4:ht, 0:3 * wd // 4]
    #     # block9[:, :, 0:ht // 4, 0:3 * wd // 4] = 0.75 * fmap2[:, :, 0:ht // 4, 0:3 * wd // 4]
    #     # block9[:, :, ht // 4:ht, 2 * wd // 4:3 * wd // 4] = 0.75 * fmap2[:, :, ht // 4:ht, 2 * wd // 4:3 * wd // 4]
    #
    #     block10[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     block11[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     block12[:, :, ht // 4:ht, wd // 4:wd] = fmap2[:, :, ht // 4:ht, wd // 4:wd]
    #     # block12[:, :, 0:ht // 4, wd // 4:wd] = 0.75 * fmap2[:, :, 0:ht // 4, wd // 4:wd]
    #     # block12[:, :, ht // 4:ht, wd // 4:2 * wd // 4] = 0.75 * fmap2[:, :, ht // 4:ht, wd // 4:2 * wd // 4]
    #
    #     block13[:, :, 2 * ht // 4:ht, 0:3 * wd // 4] = fmap2[:, :, 2 * ht // 4:ht, 0:3 * wd // 4]
    #     block13[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     block13[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd]
    #
    #     block14[:, :, 2 * ht // 4:ht, 0:wd] = fmap2[:, :, 2 * ht // 4:ht, 0:wd]
    #     block14[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     # block14[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd]
    #
    #     block15[:, :, 2 * ht // 4:ht, 0:wd] = fmap2[:, :, 2 * ht // 4:ht, 0:wd]
    #     block15[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     # block15[:, :, 2 * ht // 4:ht, 0:wd // 4] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 0:wd // 4]
    #
    #     block16[:, :, 2 * ht // 4:ht, wd // 4:wd] = fmap2[:, :, 2 * ht // 4:ht, wd // 4:wd]
    #     block16[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     block16[:, :, 2 * ht // 4:ht, 0:wd // 4] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 0:wd // 4]
    #     # print(block1 == fmap2)
    #     # print(block2 == block1)
    #     block1_view = block1.view(batch, dim, ht * wd)
    #     block2_view = block2.view(batch, dim, ht * wd)
    #     block3_view = block3.view(batch, dim, ht * wd)
    #     block4_view = block4.view(batch, dim, ht * wd)
    #     block5_view = block5.view(batch, dim, ht * wd)
    #     block6_view = block6.view(batch, dim, ht * wd)
    #     block7_view = block7.view(batch, dim, ht * wd)
    #     block8_view = block8.view(batch, dim, ht * wd)
    #     block9_view = block9.view(batch, dim, ht * wd)
    #     block10_view = block10.view(batch, dim, ht * wd)
    #     block11_view = block11.view(batch, dim, ht * wd)
    #     block12_view = block12.view(batch, dim, ht * wd)
    #     block13_view = block13.view(batch, dim, ht * wd)
    #     block14_view = block14.view(batch, dim, ht * wd)
    #     block15_view = block15.view(batch, dim, ht * wd)
    #     block16_view = block16.view(batch, dim, ht * wd)
    #
    #     corr1 = torch.matmul(test1.transpose(1, 2), block1_view)
    #     corr2 = torch.matmul(test2.transpose(1, 2), block2_view)
    #     corr3 = torch.matmul(test3.transpose(1, 2), block3_view)
    #     corr4 = torch.matmul(test4.transpose(1, 2), block4_view)
    #     corr5 = torch.matmul(test5.transpose(1, 2), block5_view)
    #     corr6 = torch.matmul(test6.transpose(1, 2), block6_view)
    #
    #     corr7 = torch.matmul(test7.transpose(1, 2), block7_view)
    #     corr8 = torch.matmul(test8.transpose(1, 2), block8_view)
    #     corr9 = torch.matmul(test9.transpose(1, 2), block9_view)
    #     corr10 = torch.matmul(test10.transpose(1, 2), block10_view)
    #     corr11 = torch.matmul(test11.transpose(1, 2), block11_view)
    #     corr12 = torch.matmul(test12.transpose(1, 2), block12_view)
    #
    #     corr13 = torch.matmul(test13.transpose(1, 2), block13_view)
    #     corr14 = torch.matmul(test14.transpose(1, 2), block14_view)
    #     corr15 = torch.matmul(test15.transpose(1, 2), block15_view)
    #     corr16 = torch.matmul(test16.transpose(1, 2), block16_view)
    #
    #     corr_region1 = corr1 + corr2 + corr3 + corr4
    #     corr_region2 = corr5 + corr6 + corr7 + corr8
    #     corr_region3 = corr9 + corr10 + corr11 + corr12
    #     corr_region4 = corr13 + corr14 + corr15 + corr16
    #
    #     corr = torch.cat((corr_region1, corr_region2, corr_region3, corr_region4), dim=1)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())
    # def corr(fmap1, fmap2, region1, region2, region3, region4, region5, region6, region7, region8):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # fmap1 = fmap1.transpose(1, 2)
    #     #  4份窄矩形       阈值设置为5
    #     # 将相关区域分为强相关和弱相关
    #
    #     region1[:, :, 0:4 * ht // 8, 0: wd] = fmap2[:, :, 0:4 * ht // 8, 0: wd]
    #     region1[:, :, 4 * ht // 8:6 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd]
    #
    #     region2[:, :, 0:5 * ht // 8, 0: wd] = fmap2[:, :, 0:5 * ht // 8, 0: wd]
    #     region2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 8:6 * ht // 8, 0:wd]
    #
    #     region3[:, :, 0:5 * ht // 8, 0: wd] = fmap2[:, :, 0:5 * ht // 8, 0: wd]
    #     region3[:, :, 5 * ht // 8:7 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 5 * ht // 8:7 * ht // 8, 0:wd]
    #
    #     region4[:, :, 0:ht // 8, 0: wd] = 0.75 * fmap2[:, :, 0:ht // 8, 0: wd]
    #     region4[:, :, ht // 8:6 * ht // 8, 0:wd] = fmap2[:, :, ht // 8:6 * ht // 8, 0:wd]
    #     region4[:, :, 6 * ht // 8:ht, 0:wd] = 0.75 * fmap2[:, :, 6 * ht // 8:ht, 0:wd]
    #
    #     region5[:, :, 0:2 * ht // 8, 0: wd] = 0.75 * fmap2[:, :, 0:2 * ht // 8, 0: wd]
    #     region5[:, :, 2 * ht // 8:7 * ht // 8, 0:wd] = fmap2[:, :, 2 * ht // 8:7 * ht // 8, 0:wd]
    #     region5[:, :, 7 * ht // 8:ht, 0:wd] = 0.75 * fmap2[:, :, 7 * ht // 8:ht, 0:wd]
    #
    #     region6[:, :, ht // 8:3 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, ht // 8:3 * ht // 8, 0:wd]
    #     region6[:, :, 3 * ht // 8:ht, 0:wd] = fmap2[:, :, 3 * ht // 8:ht, 0:wd]
    #
    #     region7[:, :, ht // 8:3 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, ht // 8:3 * ht // 8, 0:wd]
    #     region7[:, :, 3 * ht // 8:ht, 0:wd] = fmap2[:, :, 3 * ht // 8:ht, 0:wd]
    #
    #     region8[:, :, 2 * ht // 8:4 * ht // 8, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 8:4 * ht // 8, 0:wd]
    #     region8[:, :, 4 * ht // 8:ht, 0:wd] = fmap2[:, :, 4 * ht // 8:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #     region5 = region5.view(batch, dim, ht * wd)
    #     region6 = region6.view(batch, dim, ht * wd)
    #     region7 = region7.view(batch, dim, ht * wd)
    #     region8 = region8.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 8]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 8:2 * ht * wd // 8]
    #     r3_fmap1 = fmap1[:, :, 2 * ht * wd // 8:3 * ht * wd // 8]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 8:4 * ht * wd // 8]
    #     r5_fmap1 = fmap1[:, :, 4 * ht * wd // 8:5 * ht * wd // 8]
    #     r6_fmap1 = fmap1[:, :, 5 * ht * wd // 8:6 * ht * wd // 8]
    #     r7_fmap1 = fmap1[:, :, 6 * ht * wd // 8:7 * ht * wd // 8]
    #     r8_fmap1 = fmap1[:, :, 7 * ht * wd // 8:8 * ht * wd // 8]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #     corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
    #     corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)
    #     corr7 = torch.matmul(r7_fmap1.transpose(1, 2), region7)
    #     corr8 = torch.matmul(r8_fmap1.transpose(1, 2), region8)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6, corr7, corr8), dim=1)
    #
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())



    # def corr(fmap1, fmap2, region1, region2, region3, region4):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # 4分条
    #     # 将相关区域分为强相关和弱相关
    #     region1[:, :, 0:ht // 2, 0:wd] = fmap2[:, :, 0:ht // 2, 0:wd]
    #     region1[:, :, ht // 2:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 2:3 * ht // 4, 0:wd]
    #
    #     region2[:, :, 0:3*ht // 4, 0:wd] = fmap2[:, :, 0:3*ht // 4, 0:wd]
    #
    #     region3[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     region4[:, :, ht // 4:ht // 2, 0:wd] = 0.75 * fmap2[:, :, ht // 4:ht // 2, 0:wd]
    #     region4[:, :, ht // 2:ht, 0:wd] = fmap2[:, :, ht // 2:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 4]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 4:ht * wd // 2]
    #     r3_fmap1 = fmap1[:, :, ht * wd // 2:3 * ht * wd // 4]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 4:ht * wd]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4), dim=1)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())

    #
    # def corr(fmap1, fmap2, block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,
    #          block12, block13, block14, block15, block16):
    #       #sintel 44
    #     batch, dim, ht, wd = fmap1.shape
    #
    #     region1 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region2 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region3 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #     region4 = torch.zeros([batch, dim, ht // 4, wd]).to(device)
    #
    #     region5 = torch.zeros([batch, dim, 2*ht // 4-ht // 4, wd]).to(device)
    #     region6 = torch.zeros([batch, dim, 2*ht // 4-ht // 4, wd]).to(device)
    #     region7 = torch.zeros([batch, dim, 2*ht // 4-ht // 4, wd]).to(device)
    #     region8 = torch.zeros([batch, dim, 2*ht // 4-ht // 4, wd]).to(device)
    #
    #     region9 = torch.zeros([batch, dim, 3*ht // 4-2*ht // 4, wd]).to(device)
    #     region10 = torch.zeros([batch, dim, 3*ht // 4-2*ht // 4, wd]).to(device)
    #     region11 = torch.zeros([batch, dim, 3*ht // 4-2*ht // 4, wd]).to(device)
    #     region12 = torch.zeros([batch, dim, 3*ht // 4-2*ht // 4, wd]).to(device)
    #
    #     region13 = torch.zeros([batch, dim, ht- 3*ht//4, wd]).to(device)
    #     region14 = torch.zeros([batch, dim, ht- 3*ht//4, wd]).to(device)
    #     region15 = torch.zeros([batch, dim, ht- 3*ht//4, wd]).to(device)
    #     region16 = torch.zeros([batch, dim, ht- 3*ht//4, wd]).to(device)
    #
    #
    #     # 第一行
    #     # region对应的是fmap1 而block对应的是fmap2
    #     region1[:, :, :, 0:wd // 4] = fmap1[:, :, 0:ht // 4, 0:wd // 4]
    #     region2[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 0:ht // 4, wd // 4:2 * wd // 4]
    #     region3[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 0:ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region4[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 0:ht // 4, 3 * wd // 4:wd]
    #
    #     region5[:, :, :, 0:wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, 0:wd // 4]
    #     region6[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, wd // 4:2 * wd // 4]
    #     region7[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, ht // 4:2 * ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region8[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, ht // 4:2 * ht // 4, 3 * wd // 4:wd]
    #
    #     region9[:, :, :, 0:wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 0:wd // 4]
    #     region10[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, wd // 4:2 * wd // 4]
    #     region11[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 2 * wd // 4:3 * wd // 4]
    #     region12[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 2 * ht // 4:3 * ht // 4, 3 * wd // 4:wd]
    #
    #     region13[:, :, :, 0:wd // 4] = fmap1[:, :, 3 * ht // 4:ht, 0:wd // 4]
    #     region14[:, :, :, wd // 4:2 * wd // 4] = fmap1[:, :, 3 * ht // 4:ht, wd // 4:2 * wd // 4]
    #     region15[:, :, :, 2 * wd // 4:3 * wd // 4] = fmap1[:, :, 3 * ht // 4:ht, 2 * wd // 4:3 * wd // 4]
    #     region16[:, :, :, 3 * wd // 4:wd] = fmap1[:, :, 3 * ht // 4:ht, 3 * wd // 4:wd]
    #
    #     test1 = region1.view(batch, dim, (ht // 4) * wd)
    #     test2 = region2.view(batch, dim, (ht // 4) * wd)
    #     test3 = region3.view(batch, dim, (ht // 4) * wd)
    #     test4 = region4.view(batch, dim, (ht // 4) * wd)
    #
    #     test5 = region5.view(batch, dim, (2*ht // 4-ht // 4) * wd)
    #     test6 = region6.view(batch, dim, (2*ht // 4-ht // 4) * wd)
    #     test7 = region7.view(batch, dim, (2*ht // 4-ht // 4) * wd)
    #     test8 = region8.view(batch, dim, (2*ht // 4-ht // 4) * wd)
    #
    #     test9 = region9.view(batch, dim, (3*ht // 4-2*ht // 4) * wd)
    #     test10 = region10.view(batch, dim, (3*ht // 4-2*ht // 4) * wd)
    #     test11 = region11.view(batch, dim, (3*ht // 4-2*ht // 4) * wd)
    #     test12 = region12.view(batch, dim, (3*ht // 4-2*ht // 4) * wd)
    #
    #     test13 = region13.view(batch, dim, (ht - 3*ht//4) * wd)
    #     test14 = region14.view(batch, dim, (ht - 3*ht//4) * wd)
    #     test15 = region15.view(batch, dim, (ht - 3*ht//4) * wd)
    #     test16 = region16.view(batch, dim, (ht - 3*ht//4) * wd)
    #
    #     #横向多给一些，比如3/4
    #     block1[:, :, 0:2 * ht // 4, 0:3 * wd // 4] = fmap2[:, :, 0:2 * ht // 4, 0:3 * wd // 4]
    #     block1[:, :, 2 * ht // 4:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0:wd]
    #     block1[:, :, 0:2 * ht // 4, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 3 * wd // 4:wd]
    #
    #     block2[:, :, 0:2 * ht // 4, 0:wd] = fmap2[:, :, 0:2 * ht // 4, 0:wd]
    #     block2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd]
    #     #block2[:, :, 0:2 * ht // 4, 3 * wd // 4: wd] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 3 * wd // 4: wd]
    #
    #     block3[:, :, 0:2 * ht // 4, 0: wd] = fmap2[:, :, 0:2 * ht // 4, 0: wd]
    #     block3[:, :, 2 * ht // 4:3 * ht // 4, 0: wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0: wd]
    #     #block3[:, :, 0:2 * ht // 4, 0: wd // 4] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 0: wd // 4]
    #
    #     block4[:, :, 0:2 * ht // 4, wd // 4: wd] = fmap2[:, :, 0:2 * ht // 4, wd // 4: wd]
    #     block4[:, :, 2 * ht // 4:3 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:3 * ht // 4, 0:wd]
    #     block4[:, :, 0:2 * ht // 4, 0:wd // 4] = 0.75 * fmap2[:, :, 0:2 * ht // 4, 0:wd // 4]
    #
    #     block5[:, :, 0:3 * ht // 4, 0:3 * wd // 4] = fmap2[:, :, 0:3 * ht // 4, 0:3 * wd // 4]
    #     # block5[:, :, 3 * ht // 4:ht, 0:3 * wd // 4] = 0.75 * fmap2[:, :, 3 * ht // 4:ht, 0:3 * wd // 4]
    #     # block5[:, :, 0:3 * ht // 4, 2 * wd // 4:3 * wd // 4] = 0.75 * fmap2[:, :, 0:3 * ht // 4,2 * wd // 4:3 * wd // 4]
    #
    #     block6[:, :, 0:3 * ht // 4, 0:wd] = fmap2[:, :, 0:3 * ht // 4, 0:wd]
    #
    #     block7[:, :, 0:3 * ht // 4, 0:wd] = fmap2[:, :, 0:3 * ht // 4, 0:wd]
    #
    #     block8[:, :, 0:3 * ht // 4, wd // 4:wd] = fmap2[:, :, 0:3 * ht // 4, wd // 4:wd]
    #     # block8[:, :, 3 * ht // 4:ht, wd // 4:wd] = 0.75 * fmap2[:, :, 3 * ht // 4:ht, wd // 4:wd]
    #     # block8[:, :, 0:3 * ht // 4, wd // 4:2 * wd // 4] = 0.75 * fmap2[:, :, 0:3 * ht // 4, wd // 4:2 * wd // 4]
    #
    #     block9[:, :, ht // 4:ht, 0:3 * wd // 4] = fmap2[:, :, ht // 4:ht, 0:3 * wd // 4]
    #     # block9[:, :, 0:ht // 4, 0:3 * wd // 4] = 0.75 * fmap2[:, :, 0:ht // 4, 0:3 * wd // 4]
    #     # block9[:, :, ht // 4:ht, 2 * wd // 4:3 * wd // 4] = 0.75 * fmap2[:, :, ht // 4:ht, 2 * wd // 4:3 * wd // 4]
    #
    #     block10[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     block11[:, :, ht // 4:ht, 0:wd] = fmap2[:, :, ht // 4:ht, 0:wd]
    #
    #     block12[:, :, ht // 4:ht, wd // 4:wd] = fmap2[:, :, ht // 4:ht, wd // 4:wd]
    #     # block12[:, :, 0:ht // 4, wd // 4:wd] = 0.75 * fmap2[:, :, 0:ht // 4, wd // 4:wd]
    #     # block12[:, :, ht // 4:ht, wd // 4:2 * wd // 4] = 0.75 * fmap2[:, :, ht // 4:ht, wd // 4:2 * wd // 4]
    #
    #     block13[:, :, 2 * ht // 4:ht, 0:3 * wd // 4] = fmap2[:, :, 2 * ht // 4:ht, 0:3 * wd // 4]
    #     block13[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     block13[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:ht,3 * wd // 4:wd]
    #
    #     block14[:, :, 2 * ht // 4:ht, 0:wd] = fmap2[:, :, 2 * ht // 4:ht, 0:wd]
    #     block14[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     #block14[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 3 * wd // 4:wd]
    #
    #     block15[:, :, 2 * ht // 4:ht, 0:wd] = fmap2[:, :, 2 * ht // 4:ht, 0:wd]
    #     block15[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     #block15[:, :, 2 * ht // 4:ht, 0:wd // 4] = 0.75 * fmap2[:, :, 2 * ht // 4:ht, 0:wd // 4]
    #
    #     block16[:, :, 2 * ht // 4:ht, wd // 4:wd] = fmap2[:, :, 2 * ht // 4:ht, wd // 4:wd]
    #     block16[:, :, ht // 4:2 * ht // 4, 0:wd] = 0.75 * fmap2[:, :, ht // 4:2 * ht // 4, 0:wd]
    #     block16[:, :, 2 * ht // 4:ht, 0:wd // 4] = 0.75 * fmap2[:, :, 2 * ht // 4:ht,0:wd // 4]
    #     # print(block1 == fmap2)
    #     # print(block2 == block1)
    #     block1_view = block1.view(batch, dim, ht * wd)
    #     block2_view = block2.view(batch, dim, ht * wd)
    #     block3_view = block3.view(batch, dim, ht * wd)
    #     block4_view = block4.view(batch, dim, ht * wd)
    #     block5_view = block5.view(batch, dim, ht * wd)
    #     block6_view = block6.view(batch, dim, ht * wd)
    #     block7_view = block7.view(batch, dim, ht * wd)
    #     block8_view = block8.view(batch, dim, ht * wd)
    #     block9_view = block9.view(batch, dim, ht * wd)
    #     block10_view = block10.view(batch, dim, ht * wd)
    #     block11_view = block11.view(batch, dim, ht * wd)
    #     block12_view = block12.view(batch, dim, ht * wd)
    #     block13_view = block13.view(batch, dim, ht * wd)
    #     block14_view = block14.view(batch, dim, ht * wd)
    #     block15_view = block15.view(batch, dim, ht * wd)
    #     block16_view = block16.view(batch, dim, ht * wd)
    #
    #     corr1 = torch.matmul(test1.transpose(1, 2), block1_view)
    #     corr2 = torch.matmul(test2.transpose(1, 2), block2_view)
    #     corr3 = torch.matmul(test3.transpose(1, 2), block3_view)
    #     corr4 = torch.matmul(test4.transpose(1, 2), block4_view)
    #     corr5 = torch.matmul(test5.transpose(1, 2), block5_view)
    #     corr6 = torch.matmul(test6.transpose(1, 2), block6_view)
    #
    #     corr7 = torch.matmul(test7.transpose(1, 2), block7_view)
    #     corr8 = torch.matmul(test8.transpose(1, 2), block8_view)
    #     corr9 = torch.matmul(test9.transpose(1, 2), block9_view)
    #     corr10 = torch.matmul(test10.transpose(1, 2), block10_view)
    #     corr11 = torch.matmul(test11.transpose(1, 2), block11_view)
    #     corr12 = torch.matmul(test12.transpose(1, 2), block12_view)
    #
    #     corr13 = torch.matmul(test13.transpose(1, 2), block13_view)
    #     corr14 = torch.matmul(test14.transpose(1, 2), block14_view)
    #     corr15 = torch.matmul(test15.transpose(1, 2), block15_view)
    #     corr16 = torch.matmul(test16.transpose(1, 2), block16_view)
    #
    #     corr_region1 = corr1 + corr2 + corr3 + corr4
    #     corr_region2 = corr5 + corr6 + corr7 + corr8
    #     corr_region3 = corr9 + corr10 + corr11 + corr12
    #     corr_region4 = corr13 + corr14 + corr15 + corr16
    #
    #     corr = torch.cat((corr_region1, corr_region2, corr_region3, corr_region4), dim=1)
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())




    # def corr(fmap1, fmap2, region1, region2, region3, region4, region5, region6):
    #     batch, dim, ht, wd = fmap1.shape
    #     fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
    #     # fmap1 = fmap1.transpose(1, 2)
    #     # 6分条
    #
    #     region1[:, :, 0:3 * ht // 6, 0: wd] = fmap2[:, :, 0:3 * ht // 6, 0: wd]
    #     region1[:, :, 3 * ht // 6:5 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 3 * ht // 6:5 * ht // 6, 0:wd]
    #
    #     region2[:, :, 0:4 * ht // 6, 0:wd] = fmap2[:, :, 0:4 * ht // 6, 0: wd]
    #     region2[:, :, 4 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 6:ht, 0: wd]
    #
    #     region3[:, :, 0:5 * ht // 6, 0:wd] = fmap2[:, :, 0:5 * ht // 6, 0:wd]
    #     region3[:, :, 5 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 5 * ht // 6:ht, 0:wd]
    #
    #     region4[:, :, 0:ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:ht // 6, 0:wd]
    #     region4[:, :, ht // 6:ht, 0:wd] = fmap2[:, :, ht // 6:ht, 0:wd]
    #
    #     region5[:, :, 0:2 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:2 * ht // 6, 0:wd]
    #     region5[:, :, 2 * ht // 6:ht, 0:wd] = fmap2[:, :, 2 * ht // 6:ht, 0:wd]
    #
    #     region6[:, :, ht // 6:3 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, ht // 6:3 * ht // 6, 0:wd]
    #     region6[:, :, 3 * ht // 6:ht, 0:wd] = fmap2[:, :, 3 * ht // 6:ht, 0:wd]
    #
    #     region1 = region1.view(batch, dim, ht * wd)
    #     region2 = region2.view(batch, dim, ht * wd)
    #     region3 = region3.view(batch, dim, ht * wd)
    #     region4 = region4.view(batch, dim, ht * wd)
    #     region5 = region5.view(batch, dim, ht * wd)
    #     region6 = region6.view(batch, dim, ht * wd)
    #
    #     r1_fmap1 = fmap1[:, :, 0:ht * wd // 6]
    #     r2_fmap1 = fmap1[:, :, ht * wd // 6:2 * ht * wd // 6]
    #     r3_fmap1 = fmap1[:, :, 2 * ht * wd // 6:3 * ht * wd // 6]
    #     r4_fmap1 = fmap1[:, :, 3 * ht * wd // 6:4 * ht * wd // 6]
    #     r5_fmap1 = fmap1[:, :, 4 * ht * wd // 6:5 * ht * wd // 6]
    #     r6_fmap1 = fmap1[:, :, 5 * ht * wd // 6:ht * wd]
    #
    #     corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
    #     corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
    #     corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
    #     corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
    #     corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
    #     corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)
    #
    #     corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6), dim=1)
    #
    #     corr = corr.view(batch, ht, wd, 1, ht, wd)
    #     return corr / torch.sqrt(torch.tensor(dim).float())






