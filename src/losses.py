# src/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ------------------------------------------------------------
# PERCEPTUAL LOSS
# ------------------------------------------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg_pretrained = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features.eval()

        self.slice1 = nn.Sequential(*[vgg_pretrained[x] for x in range(2)])      
        self.slice2 = nn.Sequential(*[vgg_pretrained[x] for x in range(2,7)])    
        self.slice3 = nn.Sequential(*[vgg_pretrained[x] for x in range(7,12)])   
        self.slice4 = nn.Sequential(*[vgg_pretrained[x] for x in range(12,21)])  

        for p in self.parameters():
            p.requires_grad = False

        self.register_buffer('mean', torch.tensor([0.485,0.456,0.406]).view(1,3,1,1))
        self.register_buffer('std',  torch.tensor([0.229,0.224,0.225]).view(1,3,1,1))

    def forward(self, x, y):
        loss = 0.0
        device = x.device

        mean = self.mean.to(device)
        std  = self.std.to(device)

        x = (x - mean) / std
        y = (y - mean) / std

        x1 = self.slice1(x); y1 = self.slice1(y); loss += F.l1_loss(x1, y1)
        x2 = self.slice2(x1); y2 = self.slice2(y1); loss += F.l1_loss(x2, y2)
        x3 = self.slice3(x2); y3 = self.slice3(y2); loss += F.l1_loss(x3, y3)
        x4 = self.slice4(x3); y4 = self.slice4(y3); loss += F.l1_loss(x4, y4)

        return loss


# ------------------------------------------------------------
# EDGE LOSS (SOBEL)
# ------------------------------------------------------------
class EdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(2,3)

        self.register_buffer('kx', kx)
        self.register_buffer('ky', ky)

    def forward(self, pred, target):
        pred_gray = 0.2989 * pred[:,0:1] + 0.5870 * pred[:,1:2] + 0.1140 * pred[:,2:3]
        tgt_gray  = 0.2989 * target[:,0:1] + 0.5870 * target[:,1:2] + 0.1140 * target[:,2:3]

        gx_p = F.conv2d(pred_gray, self.kx, padding=1)
        gy_p = F.conv2d(pred_gray, self.ky, padding=1)

        gx_t = F.conv2d(tgt_gray, self.kx, padding=1)
        gy_t = F.conv2d(tgt_gray, self.ky, padding=1)

        g_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-6)
        g_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-6)

        return F.l1_loss(g_p, g_t)


# ------------------------------------------------------------
# TV LOSS
# ------------------------------------------------------------
def tv_loss(x):
    dh = torch.abs(x[:,:,1:,:] - x[:,:,:-1,:]).mean()
    dw = torch.abs(x[:,:,:,1:] - x[:,:,:,:-1]).mean()
    return dh + dw


# ------------------------------------------------------------
# FIXED COMBINED LOSS (NOW FULLY WORKING)
# ------------------------------------------------------------
class CombinedLoss(nn.Module):
    def __init__(self, lam_l1=1.0, lam_perc=1.0, lam_edge=1.0, lam_tv=0.0):
        super().__init__()
        self.lam_l1 = lam_l1
        self.lam_perc = lam_perc
        self.lam_edge = lam_edge
        self.lam_tv = lam_tv

        self.perc_loss = VGGPerceptualLoss()
        self.edge_loss = SobelEdgeLoss()

        self.l1_loss = nn.L1Loss()

    # ------------------------------
    # For refiner: expose separate calls
    # ------------------------------
    def l1(self, pred, target):
        return self.l1_loss(pred, target)

    def perc(self, pred, target):
        return self.perc_loss(pred, target)

    def edge(self, pred, target):
        return self.edge_loss(pred, target)

    def tv(self, pred):
        return tv_loss(pred)

    # ------------------------------
    # Combined UNet loss
    # ------------------------------
    def forward(self, pred, target):
        loss = 0.0

        loss += self.lam_l1   * self.l1(pred, target)
        loss += self.lam_perc * self.perc(pred, target)
        loss += self.lam_edge * self.edge(pred, target)
        if self.lam_tv > 0:
            loss += self.lam_tv * self.tv(pred)

        return loss

# ------------------------------------------------------------
# SOBEL EDGE LOSS (Used by Refiner)
# ------------------------------------------------------------
class SobelEdgeLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Sobel kernels
        kx = torch.tensor([[1, 0, -1],
                           [2, 0, -2],
                           [1, 0, -1]], dtype=torch.float32).view(1,1,3,3)

        ky = torch.tensor([[1, 2, 1],
                           [0, 0, 0],
                           [-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)

        self.register_buffer("kx", kx)
        self.register_buffer("ky", ky)

    def forward(self, pred, target):
        # convert RGB → grayscale
        pred_gray = (
            0.2989*pred[:,0:1] +
            0.5870*pred[:,1:2] +
            0.1140*pred[:,2:3]
        )
        tgt_gray = (
            0.2989*target[:,0:1] +
            0.5870*target[:,1:2] +
            0.1140*target[:,2:3]
        )

        # convolve
        gx_p = F.conv2d(pred_gray, self.kx, padding=1)
        gy_p = F.conv2d(pred_gray, self.ky, padding=1)

        gx_t = F.conv2d(tgt_gray, self.kx, padding=1)
        gy_t = F.conv2d(tgt_gray, self.ky, padding=1)

        grad_p = torch.sqrt(gx_p**2 + gy_p**2 + 1e-6)
        grad_t = torch.sqrt(gx_t**2 + gy_t**2 + 1e-6)

        return F.l1_loss(grad_p, grad_t)
