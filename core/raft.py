import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from update import BasicUpdateBlock, SmallUpdateBlock
from extractor import BasicEncoder, SmallEncoder
from corr import CorrBlock
from corr1 import CorrBlock1
from corr2 import CorrBlock2
from corr3 import CorrBlock3
from corr4 import CorrBlock4
from utils.utils import bilinear_sampler, coords_grid, upflow8

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


class RAFT(nn.Module):
    def __init__(self, args):
        super(RAFT, self).__init__()
        self.args = args
        self.method=args.method
        # if 'dataset' not in args._get_kwargs():
        #     args.dataset = 'kitti'
        # else:
        #     args.dataset = 'sintel'
        if not args.dataset:
            if not args.stage:
                raise ValueError('args.dataset is empty!')
            else:
                self.dataset = args.stage
        self.dataset = args.dataset or args.stage
        if args.small:
            self.hidden_dim = hdim = 96
            self.context_dim = cdim = 64
            args.corr_levels = 4
            args.corr_radius = 3
        
        else:
            self.hidden_dim = hdim = 128
            self.context_dim = cdim = 128
            args.corr_levels = 4
            args.corr_radius = 4

        if 'dropout' not in args._get_kwargs():
            args.dropout = 0

        # feature network, context network, and update block
        if args.small:
            self.fnet = SmallEncoder(output_dim=128, norm_fn='instance', dropout=args.dropout)        
            self.cnet = SmallEncoder(output_dim=hdim+cdim, norm_fn='none', dropout=args.dropout)
            self.update_block = SmallUpdateBlock(self.args, hidden_dim=hdim)

        else:
            self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=args.dropout,method=args.method,dataset=args.dataset)
            self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=args.dropout)
            self.update_block = BasicUpdateBlock(self.args, hidden_dim=hdim)


    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
        """ Estimate optical flow between pair of frames """

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        if self.method =='Original':
            with autocast(enabled=self.args.mixed_precision):
                fmap1, fmap2 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)
        elif self.method == '4split':
            with autocast(enabled=self.args.mixed_precision):
                fmap1, fmap2,region1,region2,region3,region4 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            region1 = region1.float()
            region2 = region2.float()
            region3 = region3.float()
            region4 = region4.float()
            corr_fn = CorrBlock1(fmap1, fmap2,region1,region2,region3,region4,radius=self.args.corr_radius,dataset=self.dataset )
        elif self.method == '6split':
            with autocast(enabled=self.args.mixed_precision):
                fmap1, fmap2, region1, region2, region3, region4,region5,region6 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            region1 = region1.float()
            region2 = region2.float()
            region3 = region3.float()
            region4 = region4.float()
            region5 = region5.float()
            region6 = region6.float()
            corr_fn = CorrBlock2(fmap1, fmap2,region1,region2,region3,region4,region5,region6, radius=self.args.corr_radius,dataset=self.dataset)
        elif self.method == '8split':
            with autocast(enabled=self.args.mixed_precision):
                fmap1, fmap2, region1, region2, region3, region4, region5, region6,region7,region8 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            region1 = region1.float()
            region2 = region2.float()
            region3 = region3.float()
            region4 = region4.float()
            region5 = region5.float()
            region6 = region6.float()
            region7 = region7.float()
            region8 = region8.float()
            corr_fn = CorrBlock3(fmap1, fmap2,region1,region2,region3,region4,region5,region6,region7,region8, radius=self.args.corr_radius,dataset=self.dataset)
        else:
            with autocast(enabled=self.args.mixed_precision):
                fmap1, fmap2,block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,block12, block13, block14, block15, block16 = self.fnet([image1, image2])
            fmap1 = fmap1.float()
            fmap2 = fmap2.float()
            block1 = block1.float()
            block2 = block2.float()
            block3 = block3.float()
            block4 = block4.float()
            block5 = block5.float()
            block6 = block6.float()
            block7 = block7.float()
            block8 = block8.float()
            block9 = block9.float()
            block10 = block10.float()
            block11 = block11.float()
            block12 = block12.float()
            block13 = block13.float()
            block14 = block14.float()
            block15 = block15.float()
            block16 = block16.float()
            corr_fn = CorrBlock4(fmap1, fmap2, block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,block12, block13, block14, block15, block16, radius=self.args.corr_radius,dataset=self.dataset)

            #fmap1, fmap2,region1, region2, region3, region4, region5, region6, region7, region8= self.fnet([image1, image2])
        
        # fmap1 = fmap1.float()
        # fmap2 = fmap2.float()
        # region1 = region1.float()
        # region2 = region2.float()
        # region3 = region3.float()
        # region4 = region4.float()
        # region5 = region5.float()
        # region6 = region6.float()
        # region7 = region7.float()
        # region8 = region8.float()
        # block1 = block1.float()
        # block2 = block2.float()
        # block3 = block3.float()
        # block4 = block4.float()
        # block5 = block5.float()
        # block6 = block6.float()
        # block7 = block7.float()
        # block8 = block8.float()
        # block9 = block9.float()
        # block10 = block10.float()
        # block11 = block11.float()
        # block12 = block12.float()
        # block13 = block13.float()
        # block14 = block14.float()
        # block15 = block15.float()
        # block16 = block16.float()

        #corr_fn = CorrBlock(fmap1, fmap2,region1, region2, region3, region4, region5, region6, region7, region8,radius=self.args.corr_radius)
        # run the context network
        with autocast(enabled=self.args.mixed_precision):
            cnet = self.cnet(image1)
            net, inp = torch.split(cnet, [hdim, cdim], dim=1)
            net = torch.tanh(net)
            inp = torch.relu(inp)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                net, up_mask, delta_flow = self.update_block(net, inp, corr, flow)

            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow

            # upsample predictions
            if up_mask is None:
                flow_up = upflow8(coords1 - coords0)
            else:
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            
            flow_predictions.append(flow_up)

        if test_mode:
            return coords1 - coords0, flow_up
            
        return flow_predictions
