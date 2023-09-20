import torch
import torch.nn as nn


class ConvBNReLU(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, pad):
        super().__init__()
        self.layer = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size, stride, pad),
                nn.BatchNorm2d(ch_out),
                nn.ReLU()
        )

    def forward(self, x):
        return self.layer(x)


class FeatMaskNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.conv1 = ConvBNReLU(5, 16, 3, 1, 1)
        self.conv2 = ConvBNReLU(16, 32, 3, 2, 1)
        self.conv3 = ConvBNReLU(32, 64, 3, 2, 1)
        self.conv4 = ConvBNReLU(64, 128, 3, 2, 1)
        self.conv5 = ConvBNReLU(128, 128, 3, 1, 1)
        self.conv6 = ConvBNReLU(192, 64, 3, 1, 1)
        self.conv7 = ConvBNReLU(96, 32, 3, 1, 1)
        self.conv8 = ConvBNReLU(48, 16, 3, 1, 1)
        self.conv9 = ConvBNReLU(16, 1, 3, 1, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, input_image, input_depth, input_mpi_disparity):

        _, _, h, w = input_image.size()   
        b, s = input_mpi_disparity.size()   
         
        expanded_image = input_image.unsqueeze(1).repeat(1, s, 1, 1, 1)   
         
        expanded_depth = input_depth.unsqueeze(1).repeat(1, s, 1, 1, 1)   
         
        expanded_mpi_disp = input_mpi_disparity[:, :, None, None, None].repeat(1, 1, 1, h, w)   
         
        x = torch.cat([expanded_image, expanded_depth, expanded_mpi_disp], dim=2).reshape(b * s, 5, h, w)   
         
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        c5 = self.conv5(c4)
        u5 = self.upsample(c5)
        c6 = self.conv6(torch.cat([u5, c3], dim=1))
        u6 = self.upsample(c6)
        c7 = self.conv7(torch.cat([u6, c2], dim=1))
        u7 = self.upsample(c7)
        c8 = self.conv8(torch.cat([u7, c1], dim=1))
        c9 = self.conv9(c8)   
        fm = c9.reshape(b, s, h, w)
        fm = torch.softmax(fm, dim=1)

        return fm