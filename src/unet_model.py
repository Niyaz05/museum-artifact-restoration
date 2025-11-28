# src/unet_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def resnet34_backbone(pretrained=True):
    try:
        weights = torchvision.models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = torchvision.models.resnet34(weights=weights)
    except Exception:
        resnet = torchvision.models.resnet34(pretrained=pretrained)
    return resnet

class DecoderBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        # we keep upsample in the block but will ensure explicit alignment outside if needed
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.act = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)
        return x

class ResNet34UNet(nn.Module):
    """
    ResNet34 UNet (full-output) robust to slight size mismatches by aligning decoder features
    to encoder features before concatenation using bilinear interpolation.
    Input & output tensors are expected in range 0..1.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        resnet = resnet34_backbone(pretrained=pretrained)
        # Encoder
        self.initial = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)  # /4
        self.enc1 = resnet.layer1  # /4 -> 64
        self.enc2 = resnet.layer2  # /8 -> 128
        self.enc3 = resnet.layer3  # /16 -> 256
        self.enc4 = resnet.layer4  # /32 -> 512

        # Center
        self.center = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        # Decoder blocks (concatenate skip)
        # in_ch is (center_out + skip_channels)
        self.dec4 = DecoderBlock(256 + 512, 256)   # -> /16
        self.dec3 = DecoderBlock(256 + 256, 128)   # -> /8
        self.dec2 = DecoderBlock(128 + 128, 64)    # -> /4
        self.dec1 = DecoderBlock(64 + 64, 64)      # -> /2

        # final upsample to restore to original resolution
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Final conv produces full RGB output (Sigmoid -> 0..1)
        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 3, 1),
            nn.Sigmoid()
        )

    def _align_and_cat(self, dec_feat, enc_feat):
        """
        Ensure dec_feat spatial size matches enc_feat. Resize dec_feat if needed and concat.
        """
        if dec_feat.shape[2:] != enc_feat.shape[2:]:
            dec_feat = F.interpolate(dec_feat, size=enc_feat.shape[2:], mode='bilinear', align_corners=False)
        return torch.cat([dec_feat, enc_feat], dim=1)

    def forward(self, x):
        # x: [B,3,H,W] in 0..1 (padded)
        x0 = self.initial(x)   # /4
        e1 = self.enc1(x0)     # /4
        e2 = self.enc2(e1)     # /8
        e3 = self.enc3(e2)     # /16
        e4 = self.enc4(e3)     # /32
        c = self.center(e4)

        # decoder stage 4: align c -> e4 size, concat, decode
        # note: center produces /32-> upsample to /16 inside DecoderBlock
        d4_in = self._align_and_cat(c, e4)
        d4 = self.dec4(d4_in)   # /16

        # stage 3: align d4 to e3, concat
        d3_in = self._align_and_cat(d4, e3)
        d3 = self.dec3(d3_in)   # /8

        # stage 2:
        d2_in = self._align_and_cat(d3, e2)
        d2 = self.dec2(d2_in)   # /4

        # stage 1:
        d1_in = self._align_and_cat(d2, e1)
        d1 = self.dec1(d1_in)   # /2

        d1_up = self.final_upsample(d1)             # /1 (may be off-by-1; final conv handles it)
        out = self.final_conv(d1_up)

        # If out spatially mismatches input (rare), resize to input size
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)

        return out
