import torch
import torch.nn.functional as F
from utils.utils import bilinear_sampler, coords_grid
import math

# device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = 'cuda'


class CorrBlock2:
    def __init__(self, fmap1, fmap2,region1, region2, region3, region4, region5, region6, num_levels=4, radius=4,dataset='kitti'):
        self.num_levels = num_levels
        self.radius = radius
        self.corr_pyramid = []
        self.dataset = dataset
        # all pairs correlation
        if self.dataset=='kitti':
            corr = CorrBlock2.corr2_k(fmap1, fmap2, region1, region2, region3, region4, region5, region6)
        else:
            corr = CorrBlock2.corr2_s(fmap1, fmap2, region1, region2, region3, region4, region5, region6)

        batch, h1, w1, dim, h2, w2 = corr.shape
        corr = corr.reshape(batch * h1 * w1, dim, h2, w2)

        self.corr_pyramid.append(corr)
        for i in range(self.num_levels - 1):
            corr = F.avg_pool2d(corr, 2, stride=2)
            self.corr_pyramid.append(corr)

    def __call__(self, coords):
        r = self.radius
        coords = coords.permute(0, 2, 3, 1)
        batch, h1, w1, _ = coords.shape

        out_pyramid = []
        for i in range(self.num_levels):
            corr = self.corr_pyramid[i]
            dx = torch.linspace(-r, r, 2 * r + 1)
            dy = torch.linspace(-r, r, 2 * r + 1)
            delta = torch.stack(torch.meshgrid(dy, dx), axis=-1).to(coords.device)

            centroid_lvl = coords.reshape(batch * h1 * w1, 1, 1, 2) / 2 ** i
            delta_lvl = delta.view(1, 2 * r + 1, 2 * r + 1, 2)
            coords_lvl = centroid_lvl + delta_lvl

            corr = bilinear_sampler(corr, coords_lvl)
            corr = corr.view(batch, h1, w1, -1)
            out_pyramid.append(corr)

        out = torch.cat(out_pyramid, dim=-1)
        return out.permute(0, 3, 1, 2).contiguous().float()



    @staticmethod
    def corr2_k(fmap1, fmap2, region1, region2, region3, region4, region5, region6):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
        # fmap1 = fmap1.transpose(1, 2)
        #  4份窄矩形       阈值设置为5
        # 将相关区域分为强相关和弱相关

        region1[:, :, 0:ht//2, 0:wd] = fmap2[:, :, 0:ht//2, 0:wd]                                           #1.00   [0:3*ht//6]
        region1[:, :, ht//2:4*ht//6, 0:wd] = 0.75 * fmap2[:, :, ht//2:4*ht//6, 0:wd]                        #0.75   [3*ht//6:4*ht//6]
        region1[:, :, 4 * ht // 6:5 * ht // 6, 0:wd] = 0.5 * fmap2[:, :, 4 * ht // 6:5 * ht // 6, 0:wd]     #0.50   [4*ht//6:5*ht//6]

        region2[:, :, 0:ht//2, 0:wd] = fmap2[:, :, 0:ht//2, 0:wd]
        region2[:, :, ht // 2:5 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, ht // 2:5 * ht // 6, 0:wd]

        region3[:, :, 0:ht//6, 0:wd] = 0.75*fmap2[:, :, 0:ht//6, 0:wd]
        region3[:, :, ht // 6:4*ht // 6, 0:wd] = fmap2[:, :, ht // 6:4*ht // 6, 0:wd]
        region3[:, :, 4 * ht // 6:5 * ht // 6, 0:wd] = 0.75*fmap2[:, :, 4 * ht // 6:5 * ht // 6, 0:wd]

        region4[:, :, ht // 6:2*ht // 6, 0:wd] = 0.75 * fmap2[:, :, ht // 6:2*ht // 6, 0:wd]
        region4[:, :, 2*ht // 6:5*ht//6, 0:wd] = fmap2[:, :, 2*ht // 6:5*ht//6, 0:wd]
        region4[:, :, 5*ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 5*ht // 6:ht, 0:wd]

        region5[:, :, ht//2:ht, 0:wd] = fmap2[:, :, ht//2:ht, 0:wd]
        region5[:, :, ht // 6:ht // 2, 0:wd] = 0.75 * fmap2[:, :, ht // 6:ht // 2, 0:wd]

        region6[:, :, ht//2:ht, 0:wd] = fmap2[:, :, ht//2:ht, 0:wd]
        region6[:, :, 2*ht//6:ht//2, 0:wd] = 0.75 * fmap2[:, :, 2*ht//6:ht//2, 0:wd]
        region6[:, :, ht // 6:2*ht//6, 0:wd] = 0.5 * fmap2[:, :, ht // 6:2*ht//6, 0:wd]

        region1 = region1.view(batch, dim, ht * wd)
        region2 = region2.view(batch, dim, ht * wd)
        region3 = region3.view(batch, dim, ht * wd)
        region4 = region4.view(batch, dim, ht * wd)
        region5 = region5.view(batch, dim, ht * wd)
        region6 = region6.view(batch, dim, ht * wd)

        r1_fmap1 = fmap1[:, :, 0:ht * wd // 6]
        r2_fmap1 = fmap1[:, :, ht * wd // 6:2 * ht * wd // 6]
        r3_fmap1 = fmap1[:, :, 2 * ht * wd // 6:3 * ht * wd // 6]
        r4_fmap1 = fmap1[:, :, 3 * ht * wd // 6:4 * ht * wd // 6]
        r5_fmap1 = fmap1[:, :, 4 * ht * wd // 6:5 * ht * wd // 6]
        r6_fmap1 = fmap1[:, :, 5 * ht * wd // 6:ht * wd]

        corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
        corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
        corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
        corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
        corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
        corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)

        corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6), dim=1)

        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())

    @staticmethod
    def corr2_s(fmap1, fmap2, region1, region2, region3, region4, region5, region6):
        batch, dim, ht, wd = fmap1.shape
        fmap1 = fmap1.view(batch, dim, ht * wd).to(device)
        # fmap1 = fmap1.transpose(1, 2)
        #  4份窄矩形       阈值设置为5
        # 将相关区域分为强相关和弱相关

        region1[:, :, 0:3 * ht // 6, 0: wd] = fmap2[:, :, 0:3 * ht // 6, 0: wd]
        region1[:, :, 3 * ht // 6:5 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 3 * ht // 6:5 * ht // 6, 0:wd]

        region2[:, :, 0:4 * ht // 6, 0:wd] = fmap2[:, :, 0:4 * ht // 6, 0: wd]
        region2[:, :, 4 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 4 * ht // 6:ht, 0: wd]

        region3[:, :, 0:5 * ht // 6, 0:wd] = fmap2[:, :, 0:5 * ht // 6, 0:wd]
        region3[:, :, 5 * ht // 6:ht, 0:wd] = 0.75 * fmap2[:, :, 5 * ht // 6:ht, 0:wd]

        region4[:, :, 0:ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:ht // 6, 0:wd]
        region4[:, :, ht // 6:ht, 0:wd] = fmap2[:, :, ht // 6:ht, 0:wd]

        region5[:, :, 0:2 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, 0:2 * ht // 6, 0:wd]
        region5[:, :, 2 * ht // 6:ht, 0:wd] = fmap2[:, :, 2 * ht // 6:ht, 0:wd]

        region6[:, :, ht // 6:3 * ht // 6, 0:wd] = 0.75 * fmap2[:, :, ht // 6:3 * ht // 6, 0:wd]
        region6[:, :, 3 * ht // 6:ht, 0:wd] = fmap2[:, :, 3 * ht // 6:ht, 0:wd]

        region1 = region1.view(batch, dim, ht * wd)
        region2 = region2.view(batch, dim, ht * wd)
        region3 = region3.view(batch, dim, ht * wd)
        region4 = region4.view(batch, dim, ht * wd)
        region5 = region5.view(batch, dim, ht * wd)
        region6 = region6.view(batch, dim, ht * wd)

        r1_fmap1 = fmap1[:, :, 0:ht * wd // 6]
        r2_fmap1 = fmap1[:, :, ht * wd // 6:2 * ht * wd // 6]
        r3_fmap1 = fmap1[:, :, 2 * ht * wd // 6:3 * ht * wd // 6]
        r4_fmap1 = fmap1[:, :, 3 * ht * wd // 6:4 * ht * wd // 6]
        r5_fmap1 = fmap1[:, :, 4 * ht * wd // 6:5 * ht * wd // 6]
        r6_fmap1 = fmap1[:, :, 5 * ht * wd // 6:ht * wd]

        corr1 = torch.matmul(r1_fmap1.transpose(1, 2), region1)
        corr2 = torch.matmul(r2_fmap1.transpose(1, 2), region2)
        corr3 = torch.matmul(r3_fmap1.transpose(1, 2), region3)
        corr4 = torch.matmul(r4_fmap1.transpose(1, 2), region4)
        corr5 = torch.matmul(r5_fmap1.transpose(1, 2), region5)
        corr6 = torch.matmul(r6_fmap1.transpose(1, 2), region6)

        corr = torch.cat((corr1, corr2, corr3, corr4, corr5, corr6), dim=1)

        corr = corr.view(batch, ht, wd, 1, ht, wd)
        return corr / torch.sqrt(torch.tensor(dim).float())












