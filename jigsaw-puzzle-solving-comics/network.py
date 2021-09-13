import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math
from torchvision import models

from efficientnet_pytorch import EfficientNet


###################### Network Selection ######################
def NetworkSelect(opt):
    if opt.arch=='resnet': return ResNet50(opt)
    if opt.arch=="efficient-net": return Efficientnet(opt)




###################### ResNet50 ######################
class ResNet50(nn.Module):
    def __init__(self, opt):
        self.pars = opt

        super(ResNet50, self).__init__()

        if opt.pretrained: print('Getting pretrained weights...')
        self.feature_extraction = models.resnet50(pretrained=False)
        self.feature_extraction.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.merge_tiles_2_pred = nn.Sequential(nn.Linear(opt.num_tiles*self.feature_extraction.fc.in_features, opt.num_classes))#, nn.Softmax(dim=1))
        self.feature_extraction = nn.Sequential(*(list(self.feature_extraction.children())[:-1]))
        self.__initialize_weights()

        print(self.feature_extraction)

    def __initialize_weights(self):
        for idx,module in enumerate(self.modules()):
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                module.weight.data.normal_(0,0.01)
                module.bias.data.zero_()

    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)
        x_list = []
        for i in range(9):
            z = self.feature_extraction(x[i])
            x_list.append(z)

        x = torch.cat(x_list, dim=1)
        #print("X CATTED ",x.shape, "Len x_list ",len(x_list), " xlist[0]", x_list[0].shape, "\n" )
        #print(self.merge_tiles_2_pred)
        x = self.merge_tiles_2_pred(x.squeeze(3).squeeze(2))
        
        #print("X Merge tiles :: ",x.shape)
        return x

    


################################## EFFICIENT NET #########################################
class Efficientnet(nn.Module):
    
    def __init__(self, opt, backbone= 'efficientnet-b5'):
        super(Efficientnet, self).__init__()
        
        self.opt = opt
        
        
        if opt.pretrained:
            self.model = EfficientNet.from_pretrained(backbone, num_classes=1)
        else:
            self.model = EfficientNet.from_name(backbone)
            
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.merge_tiles_2_pred = nn.Sequential(nn.Linear(opt.num_tiles*2048, opt.num_classes))
        
        print("Efficient Net Architecture is being Used")
        
    def forward(self, x):
        B,T,C,H,W = x.size()
        x = x.transpose(0,1)
        x_list = []
        for i in range(9):
            z = self.avg_pool(self.model.extract_features(x[i]))
            x_list.append(z)

        x = torch.cat(x_list, dim=1)
        #print("X CATTED ",x.shape, "Len x_list ",len(x_list), " xlist[0]", x_list[0].shape, "\n" )
        #print(self.merge_tiles_2_pred)
        x = self.merge_tiles_2_pred(x.squeeze(3).squeeze(2))
        
        #print("X Merge tiles :: ",x.shape)
        return x
        