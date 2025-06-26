import torch
from pyexpat import features

from utils_libs import *
import torchvision.models as models


class client_model(nn.Module):
    def __init__(self, name, args=True):
        super(client_model, self).__init__()
        self.name = name
    

        if self.name == 'cifar10_LeNet':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        if self.name == 'cifar10_LeNet_fusion':
            self.n_cls = 10
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(64+64, 192, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        if self.name == 'cifar100_LeNet':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)

        


        if self.name == 'cifar100_LeNet_fusion':
            self.n_cls = 100
            self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5)
            self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
            self.fc1 = nn.Linear(64 * 5 * 5, 384)
            self.fc2 = nn.Linear(384, 192)
            self.fc3 = nn.Linear(192, self.n_cls)
            self.fuse_conv = nn.Sequential(
                nn.Conv2d(64+64, 192, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )


        if self.name == 'Resnet18':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'
            self.model = resnet18


        if self.name == 'Resnet18_fusion':
            resnet18 = models.resnet18()
            resnet18.fc = nn.Linear(512, 10)

            # Change BN to GN
            resnet18.bn1 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer1[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=64)
            resnet18.layer1[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=64)

            resnet18.layer2[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=128)
            resnet18.layer2[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=128)

            resnet18.layer3[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=256)
            resnet18.layer3[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=256)

            resnet18.layer4[0].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[0].downsample[1] = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn1 = nn.GroupNorm(num_groups=2, num_channels=512)
            resnet18.layer4[1].bn2 = nn.GroupNorm(num_groups=2, num_channels=512)

            assert len(dict(resnet18.named_parameters()).keys()) == len(
                resnet18.state_dict().keys()), 'More BN layers are there...'

            self.conv1 = resnet18.conv1
            self.bn1 = resnet18.bn1
            self.relu = resnet18.relu
            self.maxpool = resnet18.maxpool
            self.layer1 = resnet18.layer1
            self.layer2 = resnet18.layer2
            self.layer3 = resnet18.layer3
            self.layer4 = resnet18.layer4
            self.avgpool = resnet18.avgpool
            self.fc = resnet18.fc

            self.fuse_conv1 = nn.Sequential(
                nn.Conv2d(64+128, 512, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.fuse_conv2 = nn.Sequential(
                nn.Conv2d(128+256, 512, kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )
            self.fuse_conv3 = nn.Sequential(
                nn.Conv2d(256+512,512,kernel_size=1),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten()
            )

        
    def forward(self, x, is_feat=False):
    

        if self.name == 'cifar10_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            f1 = x
            x = self.pool(F.relu(self.conv2(x)))
            f2 = x
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            f3 = x
            x = F.relu(self.fc2(x))
            f4 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2,f3,f4],x
            else:
                return x

        if self.name == 'cifar10_LeNet_fusion':
            x1 = self.pool(F.relu(self.conv1(x)))
            x2 = self.pool(F.relu(self.conv2(x1)))
            x1_resized = F.interpolate(x1, size= x2.shape[2:], mode= 'bilinear')
            fuse_conv = torch.cat([x1_resized, x2], dim=1)
            out1 = self.fuse_conv(fuse_conv)
            x2 = x2.view(-1, 64*5*5)
            x2 = F.relu(self.fc1(x2))
            x2 = F.relu(self.fc2(x2))
            x = self.fc3(x2)
            out1 = self.fc3(out1)
            if is_feat:
                return [out1], x
            else:
                return x
            
        if self.name == 'cifar100_LeNet':
            x = self.pool(F.relu(self.conv1(x)))
            f1 = x
            x = self.pool(F.relu(self.conv2(x)))
            f2 = x
            x = x.view(-1, 64 * 5 * 5)
            x = F.relu(self.fc1(x))
            f3 = x
            x = F.relu(self.fc2(x))
            f4 = x
            x = self.fc3(x)
            if is_feat:
                return [f1,f2,f3,f4],x
            else:
                return x
            

        if self.name == 'cifar100_LeNet_fusion':
            x1 = self.pool(F.relu(self.conv1(x)))
            x2 = self.pool(F.relu(self.conv2(x1)))
            x1_resized = F.interpolate(x1, size= x2.shape[2:], mode= 'bilinear')
            fuse_conv = torch.cat([x1_resized, x2], dim=1)
            out1 = self.fuse_conv(fuse_conv)
            x2 = x2.view(-1, 64*5*5)
            x2 = F.relu(self.fc1(x2))
            x2 = F.relu(self.fc2(x2))
            x = self.fc3(x2)
            out1 = self.fc3(out1)
            if is_feat:
                return [out1], x
            else:
                return x
            

        
        
        if self.name == 'Resnet18':
            x = self.model(x)

       

        if self.name == 'Resnet18_fusion':
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)
            x = self.layer1(x)
            f1 = x
            x = self.layer2(x)
            f2 = x
            x = self.layer3(x)
            f3 = x
            x = self.layer4(x)
            f4 = x
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            f1_resized = F.interpolate(f1,size= f2.shape[2:],mode='bilinear',align_corners=False)
            fuse12 = torch.cat([f1_resized,f2],dim=1)
            out1 = self.fuse_conv1(fuse12)
            out1 = self.fc(out1)
            f2_resized = F.interpolate(f2, size= f3.shape[2:],mode = 'bilinear',align_corners=False)
            fuse23 = torch.cat([f2_resized,f3], dim=1)
            out2 = self.fuse_conv2(fuse23)
            out2 = self.fc(out2)
            f3_resized = F.interpolate(f3,size= f4.shape[2:], mode = 'bilinear', align_corners=False)
            fuse34 = torch.cat([f3_resized, f4], dim=1)
            out3 = self.fuse_conv3(fuse34)
            out3 = self.fc(out3)
            if is_feat:
                return [out1, out2, out3],x
            else:
                return x

        return x

