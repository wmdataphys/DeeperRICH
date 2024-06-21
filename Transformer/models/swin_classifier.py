import torch
import torch.nn as nn
from models.Swin import SwinTransformer

class Decoder(nn.Module):
    def __init__(self, ):
        super(Decoder, self).__init__()
        self.drop = nn.Dropout(p=0.2)
        self.L3 = nn.Conv2d(384,48,kernel_size=1,stride=1,padding=0)
        self.L2 = nn.Conv2d(192,48,kernel_size=1,stride=1,padding=0)
        self.L1 = nn.Conv2d(96,48,kernel_size=1,stride=1,padding=0)
        self.B3 = nn.BatchNorm2d(48)
        self.B2 = nn.BatchNorm2d(48)
        self.B1 = nn.BatchNorm2d(48)
        self.deconv1 = nn.ConvTranspose2d(48,16,kernel_size=3,stride=2,padding=1,output_padding=(1,1))
        self.convdown1 = nn.Conv2d(64,16,kernel_size=3,stride=1)
        self.deconv2 = nn.ConvTranspose2d(16,16,kernel_size=5,stride=2,padding=0,output_padding=(1,1))
        self.convdown2 = nn.Conv2d(64,8,kernel_size=3,stride=2,padding=1)
        self.act = nn.SELU()
    def forward(self,features):
        T4 = self.act(self.B3(self.L3(features[3])))
        T3 = self.act(self.B2(self.L2(features[2])))
        T2 = self.act(self.B1(self.L1(features[1])))

        d1 = torch.concat([self.deconv1(T4),T3],axis=1)
        S4 = self.drop(self.act(self.convdown1(d1)))
        S43 = self.drop(torch.concat([self.deconv2(S4),T2],axis=1))
        out = self.convdown2(S43).flatten(1)

        return out
        
# I need to refactor such that all of this is dynamic based on embed_dim
class Classifier(nn.Module):
    def __init__(self,patch_size=2,in_chans=2,embed_dim=48,drop_rates=[0.1,0.1,0.1,0.1],num_heads=[3,6,12,24],depths=[2,2,6,2],window_size=7):
        super(Classifier,self).__init__()
        self.swin = SwinTransformer(
                     pretrain_img_size=None,
                     patch_size=patch_size,
                     in_chans=in_chans,
                     embed_dim=embed_dim,
                     depths=depths,
                     num_heads=num_heads,
                     window_size=window_size,
                     mlp_ratio=4.,
                     qkv_bias=True,
                     qk_scale=None,
                     drop_rate=drop_rates[0],
                     attn_drop_rate=drop_rates[1],
                     drop_path_rate=drop_rates[2],
                     norm_layer=nn.LayerNorm,
                     ape=False,
                     patch_norm=True,
                     out_indices=(0, 1, 2, 3),
                     frozen_stages=-1,
                     use_checkpoint=False)
        self.decoder = Decoder()
        self.condition_dim = 3
        self.MLP = nn.Sequential(*[nn.Linear(864+self.condition_dim,128),nn.BatchNorm1d(128),nn.SELU(),nn.Linear(128,1),nn.Sigmoid()])

    def forward(self,x,c):
        x = self.swin(x)
        x = self.decoder(x)
        x = torch.concat([x,c],axis=1)
        y = self.MLP(x).squeeze()
        return y
