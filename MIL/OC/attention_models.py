import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.squeezenet import SqueezeNet, Fire
from torchvision.models.resnet import ResNet, Bottleneck


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
        self.adavpool = nn.AdaptiveAvgPool2d((1, 1))
        final_conv = nn.Conv2d(512, self.num_classes, kernel_size=1).to('cuda:'+self.gpu_number)

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
        x = self.adavpool(x)
        return x.view(x.size(0), -1)


num_classes = 2
class ModelParallelResNet18(ResNet):
    def __init__(self, gpu_number, *args, **kwargs):
        super(ModelParallelResNet18, self).__init__(
            Bottleneck, [2, 2, 2, 2], num_classes=num_classes, *args, **kwargs) 
        self.gpu_number = gpu_number
        self.seq1 = nn.Sequential(
            self.conv1,
            self.bn1,
            self.relu,
            self.maxpool,
            
        ).to('cuda:'+self.gpu_number)
        
        self.seq2 = nn.Sequential(
            self.layer1,
        ).to('cuda:'+self.gpu_number)

        self.seq2_2 = nn.Sequential(
            self.layer2
        ).to('cuda:'+self.gpu_number)

        self.seq3 = nn.Sequential(
            self.layer3,
            self.layer4,
            self.avgpool
        ).to('cuda:'+self.gpu_number)

    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:'+self.gpu_number))
        x = self.seq2_2(x.to('cuda:'+self.gpu_number))
        x = self.seq3(x.to('cuda:'+self.gpu_number))
        return x.view(x.size(0), -1) 

    
class Attention_bags_1GPU(nn.Module):
    def __init__(self, gpu_number, model_architecture):
        super(Attention_bags_1GPU, self).__init__()
        self.gpu_number = gpu_number
        self.model_architecture = model_architecture
        self.K = 1
        if self.model_architecture=='resnet18':
            self.L = 2048 
            self.D = 524 
            self.feature_extractor = ModelParallelResNet18(self.gpu_number)
        elif self.model_architecture=='squeezenet':
            self.L = 512 
            self.D = 128 
            self.feature_extractor = ModelParallelSqueezeNet(self.gpu_number)
        elif self.model_architecture=='lenet':
            self.L = 500
            self.D = 128  
            self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)).to('cuda:'+self.gpu_number)

            self.feature_extractor_part2 = nn.Sequential(
                nn.Linear(50 * 17 * 17, self.L),
                nn.ReLU(),
            ).to('cuda:'+self.gpu_number)
        
        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        ).to('cuda:'+self.gpu_number)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        ).to('cuda:'+self.gpu_number)

    def forward(self, x):
        x = x.squeeze(0)
        if self.model_architecture=='resnet18' or self.model_architecture=='squeezenet':
            H = self.feature_extractor(x).to('cuda:'+self.gpu_number)
        elif self.model_architecture=='lenet':
            H = self.feature_extractor_part1(x).to('cuda:'+self.gpu_number)
            H = H.view(-1, 50 * 17 * 17)
            H = self.feature_extractor_part2(H).to('cuda:'+self.gpu_number)  # NxL
        
        A = self.attention(H.to('cuda:'+self.gpu_number))  # NxK
        A = torch.transpose(A.to('cuda:'+self.gpu_number), 1, 0)  # KxN
        A = F.softmax(A.to('cuda:'+self.gpu_number), dim=1)  # softmax over N

        M = torch.mm(A.to('cuda:'+self.gpu_number), H)  # KxL

        Y_prob = self.classifier(M.to('cuda:'+self.gpu_number))
        Y_hat = torch.ge(Y_prob.to('cuda:'+self.gpu_number), 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()#.data[0]
        
        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A
    
