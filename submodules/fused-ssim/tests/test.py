import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
from fused_ssim import fused_ssim

window = None
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, mask=None):
    channel = img1.size(-3)
    global window
    if window is None:
        window = create_window(window_size, channel)
        if img1.is_cuda:
            window = window.cuda()
    return _ssim(img1, img2, window, window_size, channel, mask=mask)

def conv2d(img, window, padding, groups, mask=None):
    if mask is None:
        return F.conv2d(img, window, padding=padding, groups=groups)
    else:
        masked_window = F.conv2d(mask, window, padding=padding)
        return F.conv2d(img, window, padding=padding, groups=groups) / (masked_window + torch.finfo(torch.float32).eps)

def _ssim(img1, img2, window, window_size, channel, mask=None):
    if mask is not None:
        mask = mask[None, ...]
        img1 = img1 * mask
        img2 = img2 * mask

    mu1 = conv2d(img1, window, window_size // 2, channel, mask=mask)
    mu2 = conv2d(img2, window, window_size // 2, channel, mask=mask)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = conv2d(img1 * img1, window, window_size // 2, channel, mask=mask) - mu1_sq
    sigma2_sq = conv2d(img2 * img2, window, window_size // 2, channel, mask=mask) - mu2_sq
    sigma12 = conv2d(img1 * img2, window, window_size // 2, channel, mask=mask) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


if __name__ == "__main__":
    img1 = torch.rand(3, 100, 100, requires_grad=True, device=torch.device('cuda'))
    img2 = torch.tensor(img1.tolist(), requires_grad=True, device=torch.device('cuda'))
    gt = torch.rand(3, 100, 100).cuda()
    mask = torch.randint(0, 4, (100, 100)).float().cuda()

    collage_mask_partial = torch.where(mask == 1, 1., 0.)
    original_ssim_map = ssim(img1, gt, mask=collage_mask_partial)
    fused_ssim_map = fused_ssim(img2, gt, collage_mask_partial)
    (1-original_ssim_map).mean().backward()
    (1-fused_ssim_map).mean().backward()

    print(original_ssim_map)
    print(fused_ssim_map)
    print(img1.grad)
    print(img2.grad)


    # mask = torch.randint(0, 2, (100, 100)).float().cuda()
    # sl = fused_ssim(img1, img2, mask)
    # (1-sl).mean().backward()
    # print(sl)
    # print(mask)
    # print(img1.grad[0])