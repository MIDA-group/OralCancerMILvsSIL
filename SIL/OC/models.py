import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import ResNet, Bottleneck
from torchvision.models.squeezenet import SqueezeNet, Fire


num_classes = 2
class ModelParallelSqueezeNet(SqueezeNet):
    def __init__(self, gpu_number, *args, **kwargs):
        super(ModelParallelSqueezeNet, self).__init__(
            version = "1_1", num_classes = num_classes, *args, **kwargs)
        self.num_classes = num_classes
        self.gpu_number = gpu_number
        
        self.seq1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
            Fire(64, 16, 64, 64),
            Fire(128, 16, 64, 64),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
            
        ).to('cuda:'+self.gpu_number)
        
        self.seq2 = nn.Sequential(
            Fire(128, 32, 128, 128),
            Fire(256, 32, 128, 128),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True)
        ).to('cuda:'+self.gpu_number)

        self.seq3 = nn.Sequential(
            Fire(256, 48, 192, 192),
            Fire(384, 48, 192, 192),
            Fire(384, 64, 256, 256),
            Fire(512, 64, 256, 256)
            
        ).to('cuda:'+self.gpu_number)

        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1).to('cuda:'+self.gpu_number)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5), final_conv, nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        ).to('cuda:'+self.gpu_number)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    torch.nn.init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    torch.nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
                    
    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:'+self.gpu_number))
        x = self.seq3(x.to('cuda:'+self.gpu_number))
        x = self.classifier(x)
        return torch.flatten(x, 1)   
    
    

class ModelParallelResNet18(ResNet):
    def __init__(self, gpu_number, *args, **kwargs):
        super(ModelParallelResNet18, self).__init__(
            Bottleneck, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs)       
        self.gpu_number = gpu_number
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool
            
        ).to('cuda:'+self.gpu_number)
        
        self.seq2 = nn.Sequential(
            self.layer1,
            self.layer2
        ).to('cuda:'+self.gpu_number)

        self.seq3 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        ).to('cuda:'+self.gpu_number)

        self.fc.to('cuda:'+self.gpu_number)
        
    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:'+self.gpu_number))
        x = self.seq3(x.to('cuda:'+self.gpu_number))
        return self.fc(x.view(x.size(0), -1))#x.view(x.size(0), -1) 
    
    

class Lenet(nn.Module):
    def __init__(self, gpu_number):
        super(Lenet, self).__init__()    
        self.gpu_number = gpu_number
        self.L = 500 #self.L = 2048
        self.D = 128 #self.D = 524
        self.K = 1
        
        self.feature_extractor_part1 = nn.Sequential(
           nn.Conv2d(3, 20, kernel_size=5),
           nn.ReLU(),
           nn.MaxPool2d(2, stride=2),
           nn.Conv2d(20, 50, kernel_size=5),
           nn.ReLU(),
           nn.MaxPool2d(2, stride=2)
        ).to('cuda:'+self.gpu_number)

        self.feature_extractor_part2 = nn.Sequential(
           nn.Linear(50 * 17 * 17, self.L),
           nn.ReLU(),
        ).to('cuda:'+self.gpu_number)

        self.fc = nn.Linear(self.L, num_classes).to('cuda:'+self.gpu_number)
        
    def forward(self, x):
        x = self.feature_extractor_part1(x).to('cuda:'+self.gpu_number)
        x = x.view(-1, 50 * 17 * 17)
        x = self.feature_extractor_part2(x).to('cuda:'+self.gpu_number)  # NxL

        return self.fc(x.view(x.size(0), -1))#x.view(x.size(0), -1) 
    
    