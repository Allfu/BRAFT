import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class ResidualBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(ResidualBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, stride=stride)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes)
            self.norm2 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes)
            self.norm2 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm3 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            if not stride == 1:
                self.norm3 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm3)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)



class BottleneckBlock(nn.Module):
    def __init__(self, in_planes, planes, norm_fn='group', stride=1):
        super(BottleneckBlock, self).__init__()
  
        self.conv1 = nn.Conv2d(in_planes, planes//4, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(planes//4, planes//4, kernel_size=3, padding=1, stride=stride)
        self.conv3 = nn.Conv2d(planes//4, planes, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)

        num_groups = planes // 8

        if norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm2 = nn.GroupNorm(num_groups=num_groups, num_channels=planes//4)
            self.norm3 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
            if not stride == 1:
                self.norm4 = nn.GroupNorm(num_groups=num_groups, num_channels=planes)
        
        elif norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(planes//4)
            self.norm2 = nn.BatchNorm2d(planes//4)
            self.norm3 = nn.BatchNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.BatchNorm2d(planes)
        
        elif norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(planes//4)
            self.norm2 = nn.InstanceNorm2d(planes//4)
            self.norm3 = nn.InstanceNorm2d(planes)
            if not stride == 1:
                self.norm4 = nn.InstanceNorm2d(planes)

        elif norm_fn == 'none':
            self.norm1 = nn.Sequential()
            self.norm2 = nn.Sequential()
            self.norm3 = nn.Sequential()
            if not stride == 1:
                self.norm4 = nn.Sequential()

        if stride == 1:
            self.downsample = None
        
        else:    
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride), self.norm4)


    def forward(self, x):
        y = x
        y = self.relu(self.norm1(self.conv1(y)))
        y = self.relu(self.norm2(self.conv2(y)))
        y = self.relu(self.norm3(self.conv3(y)))

        if self.downsample is not None:
            x = self.downsample(x)

        return self.relu(x+y)

class BasicEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0,method='Original',dataset='kitti'):
        super(BasicEncoder, self).__init__()
        self.norm_fn = norm_fn
        self.method = method
        self.dataset = dataset
        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=64)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(64)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(64)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 64
        self.layer1 = self._make_layer(64,  stride=1)
        self.layer2 = self._make_layer(96, stride=2)
        self.layer3 = self._make_layer(128, stride=2)

        # output convolution
        self.conv2 = nn.Conv2d(128, output_dim, kernel_size=1)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = ResidualBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = ResidualBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
        
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)
        if self.method =='Original':       #Original_RAFT
            if is_list:
                x = torch.split(x, [batch_dim, batch_dim], dim=0)
            return x
        elif self.method =='4split':     #Ours(4split)
            if self.dataset == 'kitti':     #KITTI
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy(0.25 * x[1].detach())
                    region2 = copy.deepcopy(0.5 * x[1].detach())
                    region3 = copy.deepcopy(0.5 * x[1].detach())
                    region4 = copy.deepcopy(0.25 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4
                else:
                    return x
            else:                           #MPI-Sintel
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy(0.25 * x[1].detach())
                    region2 = copy.deepcopy(0.5 * x[1].detach())
                    region3 = copy.deepcopy(0.5 * x[1].detach())
                    region4 = copy.deepcopy(0.25 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4
                else:
                    return x

        elif self.method == '6split':
            if self.dataset == 'kitti':     #KITTI
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy((0.25 * x[1].detach()))  # 0.25->0     #.unsqueeze(0)
                    region2 = copy.deepcopy(0.5 * x[1].detach())  # 0.5->0.25
                    region3 = copy.deepcopy(0.5 * x[1].detach())  # 0.5->0.25
                    region4 = copy.deepcopy(0.5 * x[1].detach())  # 0.25->0
                    region5 = copy.deepcopy(0.5 * x[1].detach())
                    region6 = copy.deepcopy(0.25 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4, region5, region6
                else:
                    return x
            else:                           #MPI-Sintel
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy((0.25 * x[1].detach()))
                    region2 = copy.deepcopy(0.5 * x[1].detach())
                    region3 = copy.deepcopy(0.5 * x[1].detach())
                    region4 = copy.deepcopy(0.5 * x[1].detach())
                    region5 = copy.deepcopy(0.5 * x[1].detach())
                    region6 = copy.deepcopy(0.25 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4, region5, region6
                else:
                    return x

        elif self.method=='8split':
            if self.dataset == 'kitti':  # KITTI
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy(0.5 * x[1].detach())
                    region2 = copy.deepcopy(0.5 * x[1].detach())
                    region3 = copy.deepcopy(0.5 * x[1].detach())
                    region4 = copy.deepcopy(0.5 * x[1].detach())
                    region5 = copy.deepcopy(0.5 * x[1].detach())
                    region6 = copy.deepcopy(0.5 * x[1].detach())
                    region7 = copy.deepcopy(0.5 * x[1].detach())
                    region8 = copy.deepcopy(0.5 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4, region5, region6, region7, region8
                else:
                    return x
            else:                       #MPI-Sintel
                if is_list:
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    region1 = copy.deepcopy(0.5 * x[1].detach())
                    region2 = copy.deepcopy(0.5 * x[1].detach())
                    region3 = copy.deepcopy(0.5 * x[1].detach())
                    region4 = copy.deepcopy(0.5 * x[1].detach())
                    region5 = copy.deepcopy(0.5 * x[1].detach())
                    region6 = copy.deepcopy(0.5 * x[1].detach())
                    region7 = copy.deepcopy(0.5 * x[1].detach())
                    region8 = copy.deepcopy(0.5 * x[1].detach())
                    return x[0], x[1], region1, region2, region3, region4, region5, region6, region7, region8
                else:
                    return x

        else:
            if self.dataset == 'kitti':  # KITTI
                if is_list:
                    #   KITTI 44
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    block1 = copy.deepcopy(0.5 * x[1].detach())
                    block2 = copy.deepcopy(0.5 * x[1].detach())
                    block3 = copy.deepcopy(0.5 * x[1].detach())
                    block4 = copy.deepcopy(0.5 * x[1].detach())

                    block5 = copy.deepcopy(0.85 * x[1].detach())
                    block6 = copy.deepcopy(0.85 * x[1].detach())
                    block7 = copy.deepcopy(0.85 * x[1].detach())
                    block8 = copy.deepcopy(0.85 * x[1].detach())

                    block9 = copy.deepcopy(0.85 * x[1].detach())
                    block10 = copy.deepcopy(0.85 * x[1].detach())
                    block11 = copy.deepcopy(0.85 * x[1].detach())
                    block12 = copy.deepcopy(0.85 * x[1].detach())

                    block13 = copy.deepcopy(0.5 * x[1].detach())
                    block14 = copy.deepcopy(0.5 * x[1].detach())
                    block15 = copy.deepcopy(0.5 * x[1].detach())
                    block16 = copy.deepcopy(0.5 * x[1].detach())
                    return x[0], x[1], block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11, block12, block13, block14, block15, block16
                else:
                    return x
            else:
                if is_list:
                  #   sintel 44
                    x = torch.split(x, [batch_dim, batch_dim], dim=0)
                    block1 = copy.deepcopy(0.25 * x[1].detach())
                    block2 = copy.deepcopy(0.25 * x[1].detach())
                    block3 = copy.deepcopy(0.25 * x[1].detach())
                    block4 = copy.deepcopy(0.25 * x[1].detach())

                    block5 = copy.deepcopy(0.75 * x[1].detach())
                    block6 = copy.deepcopy(0.75 * x[1].detach())
                    block7 = copy.deepcopy(0.75 * x[1].detach())
                    block8 = copy.deepcopy(0.75 * x[1].detach())

                    block9 = copy.deepcopy(0.75 * x[1].detach())
                    block10 = copy.deepcopy(0.75 * x[1].detach())
                    block11 = copy.deepcopy(0.75 * x[1].detach())
                    block12 = copy.deepcopy(0.75 * x[1].detach())

                    block13 = copy.deepcopy(0.25 * x[1].detach())
                    block14 = copy.deepcopy(0.25 * x[1].detach())
                    block15 = copy.deepcopy(0.25 * x[1].detach())
                    block16 = copy.deepcopy(0.25 * x[1].detach())
                    return x[0], x[1], block1, block2, block3, block4, block5, block6, block7, block8, block9, block10, block11,block12, block13, block14, block15, block16
                else:
                    return x




class SmallEncoder(nn.Module):
    def __init__(self, output_dim=128, norm_fn='batch', dropout=0.0):
        super(SmallEncoder, self).__init__()
        self.norm_fn = norm_fn

        if self.norm_fn == 'group':
            self.norm1 = nn.GroupNorm(num_groups=8, num_channels=32)
            
        elif self.norm_fn == 'batch':
            self.norm1 = nn.BatchNorm2d(32)

        elif self.norm_fn == 'instance':
            self.norm1 = nn.InstanceNorm2d(32)

        elif self.norm_fn == 'none':
            self.norm1 = nn.Sequential()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=7, stride=2, padding=3)
        self.relu1 = nn.ReLU(inplace=True)

        self.in_planes = 32
        self.layer1 = self._make_layer(32,  stride=1)
        self.layer2 = self._make_layer(64, stride=2)
        self.layer3 = self._make_layer(96, stride=2)

        self.dropout = None
        if dropout > 0:
            self.dropout = nn.Dropout2d(p=dropout)
        
        self.conv2 = nn.Conv2d(96, output_dim, kernel_size=1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _make_layer(self, dim, stride=1):
        layer1 = BottleneckBlock(self.in_planes, dim, self.norm_fn, stride=stride)
        layer2 = BottleneckBlock(dim, dim, self.norm_fn, stride=1)
        layers = (layer1, layer2)
    
        self.in_planes = dim
        return nn.Sequential(*layers)


    def forward(self, x):

        # if input is list, combine batch dimension
        is_list = isinstance(x, tuple) or isinstance(x, list)
        if is_list:
            batch_dim = x[0].shape[0]
            x = torch.cat(x, dim=0)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.conv2(x)

        if self.training and self.dropout is not None:
            x = self.dropout(x)

        if is_list:
            x = torch.split(x, [batch_dim, batch_dim], dim=0)
        return x

