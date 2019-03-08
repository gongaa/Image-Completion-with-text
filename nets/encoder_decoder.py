# full assembly of the sub-parts to form the complete net
import torch
import torch.nn as nn
import torch.nn.functional as F

        
class EncoderDecoder(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(EncoderDecoder, self).__init__()
        self.n_classes = n_classes
        self.encoder = nn.Sequential(
            nn.Conv2d(n_channels+n_classes, 64, 5, 1, 2),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1)
        )
        self.conv1= nn.Conv2d(128, 128, 3, 1, 2, dilation=2)
        self.conv2=    nn.Conv2d(128, 128, 3, 1, 4, dilation=4)
        self.conv3=    nn.Conv2d(128, 128, 3, 1, 8, dilation=8)
        self.conv4=    nn.Conv2d(128, 128, 3, 1, 16, dilation=16)
            
        self.conv_new = nn.Conv2d(128, 128, 3, 1, 1)
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.Conv2d(64, self.n_classes, 3, 1, 1)
        )

    def forward(self, img, seg, mask):
        bs,c,h,w = img.size()
        mask = mask.unsqueeze(1)
        seg_out = seg*mask
        x = torch.cat([img, seg_out], dim=1)

        x = self.encoder(x)

        x = self.conv1(x)
    
        x = self.conv2(x)
      
        x = self.conv3(x)
      
        x = self.conv4(x)
      
        x = self.conv_new(x)

        x = self.decoder(x)
        x = x*(1-mask) + seg_out
        # assert x.size() == [bs, self.n_classes, h, w], [ img.size(), x.size() ]
        # try my probability map first
        # x = x.view(bs, self.n_classes, h, w)
        # x = F.softmax(x, 1)
        # x = x.view(bs, self.n_classes, h, w)
        return x