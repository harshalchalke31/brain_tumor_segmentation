import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------------------------------------------
# 1) Squeeze-and-Excitation (3D)
# ------------------------------------------------------------------
class SELayer3D(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SELayer3D, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, d, h, w = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y

# ------------------------------------------------------------------
# 2) 3D Inverted Residual Block (MobileNetV3 style)
# ------------------------------------------------------------------
class InvertedResidual3D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, use_se, activation):
        super(InvertedResidual3D, self).__init__()
        self.use_res_connect = (stride == (1,1,1) and in_channels == out_channels)
        hidden_dim = int(round(in_channels * expand_ratio))
        
        # Choose activation
        if activation == "RE":
            act_layer = nn.ReLU(inplace=True)
        elif activation == "HS":
            act_layer = nn.Hardswish(inplace=True)
        else:
            raise NotImplementedError(f"Activation {activation} not implemented.")
        
        layers = []
        # (1) Pointwise (Expansion)
        if expand_ratio != 1:
            layers += [
                nn.Conv3d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm3d(hidden_dim),
                act_layer
            ]
        # (2) Depthwise
        layers += [
            nn.Conv3d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                      padding=kernel_size // 2, groups=hidden_dim, bias=False),
            nn.BatchNorm3d(hidden_dim)
        ]
        if use_se:
            layers.append(SELayer3D(hidden_dim))
        layers.append(act_layer)
        # (3) Pointwise (Projection)
        layers += [
            nn.Conv3d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm3d(out_channels)
        ]
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# ------------------------------------------------------------------
# 3) 3D MobileNetV3-Inspired Encoder
# ------------------------------------------------------------------
class MobileNetV3Encoder3D(nn.Module):
    def __init__(self, in_channels=1):
        super(MobileNetV3Encoder3D, self).__init__()
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, 16, kernel_size=(3,3,3), stride=(1,2,2), 
                      padding=1, bias=False),
            nn.BatchNorm3d(16),
            nn.Hardswish(inplace=True)
        )
        
        # Inverted Residual blocks (inspired by MobileNetV3-Small)
        self.block1 = InvertedResidual3D(16, 16, kernel_size=3, stride=(1,1,1),
                                         expand_ratio=1, use_se=True, activation="RE")
        self.block2 = InvertedResidual3D(16, 24, kernel_size=3, stride=(1,2,2),
                                         expand_ratio=4.5, use_se=False, activation="RE")
        self.block3 = InvertedResidual3D(24, 24, kernel_size=3, stride=(1,1,1),
                                         expand_ratio=3.67, use_se=False, activation="RE")
        self.block4 = InvertedResidual3D(24, 40, kernel_size=5, stride=(1,2,2),
                                         expand_ratio=4, use_se=True, activation="HS")
        self.block5 = InvertedResidual3D(40, 40, kernel_size=5, stride=(1,1,1),
                                         expand_ratio=6, use_se=True, activation="HS")
        self.block6 = InvertedResidual3D(40, 80, kernel_size=3, stride=(1,2,2),
                                         expand_ratio=6, use_se=False, activation="HS")
        self.block7 = InvertedResidual3D(80, 80, kernel_size=3, stride=(1,1,1),
                                         expand_ratio=2.5, use_se=False, activation="HS")

    def forward(self, x):
        """ Return list of feature maps for skip connections. """
        features = []
        
        x = self.stem(x)       # [16]
        features.append(x)
        
        x = self.block1(x)     # [16]
        features.append(x)
        
        x = self.block2(x)     # [24]
        features.append(x)
        
        x = self.block3(x)     # [24]
        features.append(x)
        
        x = self.block4(x)     # [40]
        features.append(x)
        
        x = self.block5(x)     # [40]
        features.append(x)
        
        x = self.block6(x)     # [80]
        features.append(x)
        
        x = self.block7(x)     # [80]
        features.append(x)
        
        return features

# ------------------------------------------------------------------
# 4) 3D Decoder Block (UNet-Style)
# ------------------------------------------------------------------
class DecoderBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock3D, self).__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels,
                                     kernel_size=(1,2,2), stride=(1,2,2))
        self.conv = nn.Sequential(
            nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        # Upsample
        x = self.up(x)
        # Adjust shape if needed
        if x.shape != skip.shape:
            diff_d = skip.size(2) - x.size(2)
            diff_h = skip.size(3) - x.size(3)
            diff_w = skip.size(4) - x.size(4)
            x = F.pad(x, [diff_w // 2, diff_w - diff_w // 2,
                          diff_h // 2, diff_h - diff_h // 2,
                          diff_d // 2, diff_d - diff_d // 2])
        # Skip concat
        x = torch.cat([skip, x], dim=1)
        x = self.conv(x)
        return x

# ------------------------------------------------------------------
# 5) Complete 3D MobileNetV3–UNet
# ------------------------------------------------------------------
class MobileNetV3UNet3D(nn.Module):
    def __init__(self, in_channels=3, out_channels=4):
        """
        Args:
            in_channels:  e.g. 3 for multi-channel MRI (Flair/T1ce/T2).
            out_channels: e.g. 4 for multi-class segmentation (0,1,2,3).
        """
        super(MobileNetV3UNet3D, self).__init__()
        self.encoder = MobileNetV3Encoder3D(in_channels)

        # Encoder outputs:
        # feats = [f0(16), f1(16), f2(24), f3(24), f4(40), f5(40), f6(80), f7(80)]
        # We define 5 decoder blocks to eventually reach full resolution:
        self.decoder4 = DecoderBlock3D(in_channels=80, out_channels=80)  # skip with feats[6] -> out=80
        self.decoder3 = DecoderBlock3D(in_channels=80, out_channels=40)  # skip with feats[5] -> out=40
        self.decoder2 = DecoderBlock3D(in_channels=40, out_channels=40)  # skip with feats[4] -> out=40
        self.decoder1 = DecoderBlock3D(in_channels=40, out_channels=24)  # skip with feats[3] -> out=24
        self.decoder0 = DecoderBlock3D(in_channels=24, out_channels=16)  # skip with feats[1] -> out=16

        # Final up + conv => from 16 -> out_channels
        self.final_up = nn.ConvTranspose3d(16, 16, kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.final_conv = nn.Conv3d(16, out_channels, kernel_size=1)

    def forward(self, x):
        # Encode
        feats = self.encoder(x)
        # feats[-1] = f7(80), feats[-2] = f6(80), feats[-3] = f5(40), feats[-4] = f4(40), feats[-5] = f3(24),
        # feats[-6] = f2(24), feats[-7] = f1(16), feats[-8] = f0(16)

        # Bottleneck -> 80 channels
        x = feats[-1]                       # [80]
        x = self.decoder4(x, feats[-2])     # skip=80 -> out=80
        x = self.decoder3(x, feats[-3])     # skip=40 -> out=40
        x = self.decoder2(x, feats[-4])     # skip=40 -> out=40
        x = self.decoder1(x, feats[-5])     # skip=24 -> out=24
        x = self.decoder0(x, feats[-7])     # skip=16 -> out=16

        # Final up + conv
        x = self.final_up(x)      # [16 -> 16], restore final 128×128
        x = self.final_conv(x)    # => [out_channels=4]
        return x


if __name__ == "__main__":
    model = MobileNetV3UNet3D(in_channels=3, out_channels=4).cuda()

    dummy_input = torch.randn(1, 3, 128, 128, 128).cuda()
    out = model(dummy_input)
    print("Output shape:", out.shape)  # Expect: [1, 4, 128, 128, 128]
