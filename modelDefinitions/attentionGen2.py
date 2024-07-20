import torch.nn as nn
import torch.nn.functional as F
import torch
from torchsummary import summary
from modelDefinitions.basicBlocks import *    

class attentionNet(nn.Module):
    def __init__(self, squeezeFilters = 64, expandFilters = 64, depth = 5):
        super(attentionNet, self).__init__()
        #print("Model 2")
        # Input Block
        self.inputConv = nn.Conv2d(3, squeezeFilters, 3,1,1)

        self.depthAttention1 = RRDB(64)
        self.gate1 = GatedConv2d(64, 64, 3, padding=1)
        self.down1 = nn.Conv2d(64, 128, 3, 2, 1) 
        
        self.depthAttention2 = RRDB(128)
        self.gate2 = GatedConv2d(128, 128, 3, padding=1) 
        self.down2 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.depthAttention3 = RRDB(256)
        self.gate3 = GatedConv2d(256, 256, 3, padding=1) 
        self.down3 = nn.Conv2d(256, 512, 3, 2, 1)
    
        
        self.depthAttention4 = RRDB(512)
        self.convUP1 = nn.Conv2d(512, 256, 3, 1, 1) 
        self.psUpsampling1 = pixelShuffleUpsampling(inputFilters=256, scailingFactor=2)

        self.depthAttention5 = RRDB(256)
        self.convUP2 = nn.Conv2d(256, 128, 3, 1, 1) 
        self.psUpsampling2 = pixelShuffleUpsampling(inputFilters=128, scailingFactor=2)
        
        self.depthAttention6 = RRDB(128)
        self.convUP3 = nn.Conv2d(128, 64, 3, 1, 1) 
        self.psUpsampling3 = pixelShuffleUpsampling(inputFilters=64, scailingFactor=2)

        self.convOut = nn.Conv2d(64,3,1)

        # Weight Initialization
        self._initialize_weights()

    def forward(self, img):

        xInp = F.leaky_relu(self.inputConv(img))

        xSP1 = self.depthAttention1(xInp)
        xGC1 = self.gate1(xSP1)
        xDA1 = self.down1(xSP1)

        xSP2 = self.depthAttention2(xDA1)
        xGC2 = self.gate2(xSP2)
        xDA2 = self.down2(xSP2)
        
        xSP3 = self.depthAttention3(xDA2)
        xGC3 = self.gate3(xSP3)
        xDA3 = self.down3(xSP3)
        
        xSP4 = self.depthAttention4(xDA3)
        xUC1 =  self.convUP1(xSP4)
        xUP1 =  self.psUpsampling1(xUC1) + xGC3

        xSP5 = self.depthAttention5(xUP1)
        xUC2 =  self.convUP2(xSP5)
        xUP2 =  self.psUpsampling2(xUC2) + xGC2
        
        xSP6 = self.depthAttention6(xUP2)
        xUC3 =  self.convUP3(xSP6)
        xUP3 =  self.psUpsampling3(xUC3) + xGC1

        return torch.tanh(self.convOut(xUP3) + img)

    
        
        
        
    
    def _initialize_weights(self):

        self.inputConv.apply(init_weights)
        
        self.depthAttention1.apply(init_weights)
        self.gate1.apply(init_weights)
        self.down1.apply(init_weights)

        self.depthAttention2.apply(init_weights)
        self.gate2.apply(init_weights)
        self.down3.apply(init_weights)

        self.depthAttention3.apply(init_weights)
        self.gate3.apply(init_weights)
        self.down3.apply(init_weights)

        self.depthAttention4.apply(init_weights)
        self.convUP1.apply(init_weights)
        self.psUpsampling1.apply(init_weights)

        self.depthAttention5.apply(init_weights)
        self.convUP2.apply(init_weights)
        self.psUpsampling2.apply(init_weights)

        self.depthAttention6.apply(init_weights)
        self.convUP3.apply(init_weights)
        self.psUpsampling3.apply(init_weights)
         
        self.convOut.apply(init_weights)

#net = attentionNet()
#summary(net, input_size = (3, 128, 128)
#print ("reconstruction network")