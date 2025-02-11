import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v3_large
from torchsummary import summary
# from utils import get_config  # if needed for custom construction (not used here)

# ---------------------------------------------------------------------------
# Squeeze-Excitation Block
# ---------------------------------------------------------------------------
class SEBlock(nn.Module):
    def __init__(self, in_channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(1, in_channels // reduction)
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(in_channels, reduced_channels, bias=True)
        self.fc2 = nn.Linear(reduced_channels, in_channels, bias=True)
        self.relu = nn.ReLU()
        self.hsigmoid = nn.Hardsigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.size()
        pooled = self.global_pool(x).view(batch, channels)
        excitation = self.hsigmoid(self.fc2(self.relu(self.fc1(pooled))))
        excitation = excitation.view(batch, channels, 1, 1)
        return x * excitation

# ---------------------------------------------------------------------------
# Bottleneck Block (Inverted Residual with optional SE)
# ---------------------------------------------------------------------------
class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: int, stride: int,
                 exp_channels: int, use_se: bool = False, nl: nn.Module = nn.ReLU()):
        super().__init__()
        self.use_res_connect = (in_channels == out_channels and stride == 1)
        layers = []
        # Expansion phase
        if exp_channels != in_channels:
            layers.extend([
                nn.Conv2d(in_channels, exp_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(exp_channels),
                nl
            ])
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(exp_channels, exp_channels, kernel, stride, kernel // 2,
                      groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            nl
        ])
        # Optional Squeeze-Excitation
        if use_se:
            layers.append(SEBlock(exp_channels))
        # Projection phase
        layers.extend([
            nn.Conv2d(exp_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.block(x)
        if self.use_res_connect:
            out += x
        return out

# ---------------------------------------------------------------------------
# Convolution Block (Conv + BN + Activation)
# ---------------------------------------------------------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, use_activation: bool = True, use_bn: bool = True):
        super().__init__()
        padding = kernel_size // 2
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=not use_bn)]
        if use_bn:
            layers.append(nn.BatchNorm2d(out_channels))
        if use_activation:
            layers.append(nn.ReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

# ---------------------------------------------------------------------------
# Upsample Block for UNet-style Decoder
# ---------------------------------------------------------------------------
class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        """
        A double-conv block that fuses the upsampled decoder feature with an encoder skip.
        """
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels + skip_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
        x = torch.cat([x, skip], dim=1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

# ---------------------------------------------------------------------------
# MobileNetV3 Encoder
# ---------------------------------------------------------------------------
class MobileNetV3Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, use_backbone: bool = True,
                 config_name: str = "large", backbone_pretrained: bool = True):
        """
        Encoder based on MobileNetV3.
        If use_backbone is True, a pretrained MobileNetV3-large is used.
        """
        super().__init__()
        self.use_backbone = use_backbone
        self.in_channels = in_channels

        if self.use_backbone:
            self.model = mobilenet_v3_large(pretrained=backbone_pretrained)
            self.model.classifier = nn.Identity()
            # These indices mark the layers at which resolution is downsampled.
            self.downsample_indices = [0, 2, 3, 5, len(self.model.features) - 1]
            self.features = self.model.features
        else:
            # Optionally build from scratch via get_config (not covered here)
            self.config = get_config(config_name)
            layers = []
            layers.extend([
                nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.Hardswish()
            ])
            current_idx = 0
            self.downsample_indices = [0]
            input_channels = 16
            for (k, exp_channels, in_c, out_c, use_se, nl, s) in self.config:
                block = Bottleneck(input_channels, out_c, k, s, exp_channels, use_se, nl)
                layers.append(block)
                input_channels = out_c
                current_idx += 1
                if s == 2:
                    self.downsample_indices.append(current_idx)
            layers.extend([
                nn.Conv2d(input_channels, 960, kernel_size=1, bias=False),
                nn.BatchNorm2d(960),
                nn.Hardswish()
            ])
            current_idx += 1
            self.downsample_indices.append(current_idx)
            self.model = nn.Sequential(*layers)
            self.features = self.model

        if 0 not in self.downsample_indices:
            self.downsample_indices = [0] + self.downsample_indices

    def forward(self, x: torch.Tensor):
        feats = []
        out = x
        for idx, layer in enumerate(self.features):
            out = layer(out)
            if idx in self.downsample_indices:
                feats.append(out)
        return tuple(feats)

# ---------------------------------------------------------------------------
# MobileNetV3 UNet (Standard Implementation)
# ---------------------------------------------------------------------------
class MobileNetV3UNet(nn.Module):
    def __init__(self, in_channels: int = 3, out_channels: int = 1,
                 config_name: str = "large", use_backbone: bool = True):
        """
        A UNet-style segmentation model with a MobileNetV3-large encoder.
        Input: 3-channel image.
        Output: segmentation map with out_channels channels.
        """
        super().__init__()
        # Build the encoder (pretrained MobileNetV3-large)
        self.encoder = MobileNetV3Encoder(in_channels=in_channels, use_backbone=use_backbone,
                                          config_name=config_name, backbone_pretrained=True)
        # Dummy pass to get encoder feature channel dimensions
        with torch.no_grad():
            dummy_img = torch.randn(1, in_channels, 112, 112)
            dummy_feats = self.encoder(dummy_img)
        feat_channels = [feat.size(1) for feat in dummy_feats]

        # Build UNet-style decoder (fusing skip connections from encoder)
        decoder_blocks = []
        in_ch = feat_channels[-1]
        for i in range(len(feat_channels) - 2, -1, -1):
            skip_ch = feat_channels[i]
            out_ch = max(16, skip_ch // 2)
            decoder_blocks.append(UpsampleBlock(in_channels=in_ch, skip_channels=skip_ch, out_channels=out_ch))
            in_ch = out_ch
        self.decoder_blocks = nn.ModuleList(decoder_blocks)
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.seg_head = nn.Conv2d(in_ch, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        feats = self.encoder(x)  # Tuple of features from downsample stages
        x_dec = feats[-1]
        skip_idx = len(feats) - 2
        for block in self.decoder_blocks:
            x_dec = block(x_dec, feats[skip_idx])
            skip_idx -= 1
        x_dec = self.final_upsample(x_dec)
        seg_out = self.seg_head(x_dec)
        return seg_out

# ---------------------------------------------------------------------------
# Main Testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MobileNetV3UNet(in_channels=3, out_channels=1, config_name="large", use_backbone=True).to(device)
    dummy_input = torch.randn(1, 3, 256, 256).to(device)
    output = model(dummy_input)
    print(model)
    summary(model, input_size=(3, 256, 256), device=str(device))
    print("Output shape:", output.shape)
