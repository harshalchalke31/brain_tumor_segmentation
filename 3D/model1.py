import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_classes=4):
        super(UNet3D, self).__init__()
        
        def conv_block(in_c, out_c, dropout):
            return nn.Sequential(
                nn.Conv3d(in_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout3d(p=dropout),
                nn.Conv3d(out_c, out_c, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        
        self.c1 = conv_block(in_channels, 16, 0.1)
        self.p1 = nn.MaxPool3d(2)

        self.c2 = conv_block(16, 32, 0.1)
        self.p2 = nn.MaxPool3d(2)

        self.c3 = conv_block(32, 64, 0.2)
        self.p3 = nn.MaxPool3d(2)

        self.c4 = conv_block(64, 128, 0.2)
        self.p4 = nn.MaxPool3d(2)

        self.c5 = conv_block(128, 256, 0.3)

        self.up6 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.c6 = conv_block(256, 128, 0.2)

        self.up7 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.c7 = conv_block(128, 64, 0.2)

        self.up8 = nn.ConvTranspose3d(64, 32, kernel_size=2, stride=2)
        self.c8 = conv_block(64, 32, 0.1)

        self.up9 = nn.ConvTranspose3d(32, 16, kernel_size=2, stride=2)
        self.c9 = conv_block(32, 16, 0.1)

        self.out_conv = nn.Conv3d(16, out_classes, kernel_size=1)
        
    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)

        c2 = self.c2(p1)
        p2 = self.p2(c2)

        c3 = self.c3(p2)
        p3 = self.p3(c3)

        c4 = self.c4(p3)
        p4 = self.p4(c4)

        c5 = self.c5(p4)

        u6 = self.up6(c5)
        u6 = torch.cat([u6, c4], dim=1)
        c6 = self.c6(u6)

        u7 = self.up7(c6)
        u7 = torch.cat([u7, c3], dim=1)
        c7 = self.c7(u7)

        u8 = self.up8(c7)
        u8 = torch.cat([u8, c2], dim=1)
        c8 = self.c8(u8)

        u9 = self.up9(c8)
        u9 = torch.cat([u9, c1], dim=1)
        c9 = self.c9(u9)

        out = self.out_conv(c9)
        return out  # use softmax for multi-class segmentation

if __name__=='__main__':
    model = UNet3D(in_channels=3, out_classes=4)
    print(model)

    # Test shape
    x = torch.randn(8, 3, 128, 128, 128)  # B, C, D, H, W
    y = model(x)
    print(y.shape)  # should be [8, 4, 128, 128, 128]
