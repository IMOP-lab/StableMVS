import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from models.module import *
import time
from torch.autograd import Variable
import numpy as np
from math import exp
import matplotlib.pyplot as plt


## the fourth version of loss_photo, loss_ssim, loss_perceptual, loss_jdacs
## the loss for train_sample_with_jdacs


class vggNet(nn.Module):
    def __init__(self, pretrained=True):
        super(vggNet, self).__init__()
        self.net = models.vgg16(pretrained=True).features.eval()

    def forward(self, x):

        out = []
        for i in range(len(self.net)):
            # x = self.net[i](x)
            x = self.net[i](x)
            # if i in [3, 8, 15, 22, 29]:#提取1,1/2,1/4,1/8,1/16
            if i in [3, 8, 15]:  # 提取1，1/2，1/4的特征图
                # print(self.net[i])
                out.append(x)
        return out


# object function for nmf
def approximation_error(V, W, H, square_root=True):
    # Frobenius Norm
    return torch.norm(V - torch.mm(W, H))


def multiplicative_update_step(V, W, H, update_h=None, VH=None, HH=None):
    # update operation for W
    if VH is None:
        assert HH is None
        Ht = torch.t(H)  # [k, m] --> [m, k]
        VH = torch.mm(V, Ht)  # [n, m] x [m, k] --> [n, k]
        HH = torch.mm(H, Ht)  # [k, m] x [m, k] --> [k, k]

    WHH = torch.mm(W, HH) # [n, k] x [k, k] --> [n, k]
    WHH[WHH == 0] = 1e-7
    W *= VH / WHH

    if update_h:
        # update operation for H (after updating W)
        Wt = torch.t(W)  # [n, k] --> [k, n]
        WV = torch.mm(Wt, V)  # [k, n] x [n, m] --> [k, m]
        WWH = torch.mm(torch.mm(Wt, W), H)  #
        WWH[WWH == 0] = 1e-7
        H *= WV / WWH
        VH, HH = None, None

    return W, H, VH, HH


def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, cuda=True, verbose=False):
    if verbose:
        start_time = time.time()

    # scale = math.sqrt(V.mean() / k)
    scale = torch.sqrt(V.mean() / k)

    if random_seed is not None:
        if cuda:
            current_random_seed = torch.cuda.initial_seed()
            torch.cuda.manual_seed(random_seed)
        else:
            current_random_seed = torch.initial_seed()
            torch.manual_seed(random_seed)

    if W is None:
        if cuda:
            W = torch.cuda.FloatTensor(V.size(0), k).normal_()
        else:
            W = torch.randn(V.size(0), k)
        W *= scale  # [n, k]

    update_H = True
    if H is None:
        if cuda:
            H = torch.cuda.FloatTensor(k, V.size(1)).normal_()
        else:
            H = torch.randn(k, V.size(1))
        H *= scale  # [k, m]
    else:
        update_H = False

    if random_seed is not None:
        if cuda:
            torch.cuda.manual_seed(current_random_seed)
        else:
            torch.manual_seed(current_random_seed)

    W = torch.abs(W)
    H = torch.abs(H)

    error_at_init = approximation_error(V, W, H, square_root=True)
    previous_error = error_at_init

    VH = None
    HH = None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(V, W, H, update_h=update_H, VH=VH, HH=HH)
        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H, square_root=True)
            if (previous_error - error) / error_at_init < tol:
                break
            previous_error = error

    if verbose:
        print('Exited after {} iterations. Total time: {} seconds'.format(n_iter+1, time.time()-start_time))
    return W, H


class SegDFF(nn.Module):
    def __init__(self, K, max_iter=50):
        super(SegDFF, self).__init__()
        self.K = K
        self.max_iter = max_iter
        self.net = models.vgg19(pretrained=True)
        del self.net.features._modules['36'] # delete redundant layers to save memory

    def forward(self, imgs):
        # imgs: [batch_size, num_views, 3, height, width]
        batch_size = imgs.size(0)
        heatmaps = []
        for b in range(batch_size):
            imgs_b = imgs[b]
            with torch.no_grad():
                # h, w = imgs_b.size(2), imgs_b.size(3)
                imgs_b = F.interpolate(imgs_b, size=(192, 384), mode='bilinear', align_corners=False)  # 224 244
                features = self.net.features(imgs_b)
                flat_features = features.permute(0, 2, 3, 1).contiguous().view(-1, features.size(1))
                W, _ = NMF(flat_features, self.K, random_seed=1, cuda=True, max_iter=self.max_iter, verbose=False)
                # print(torch.isnan(W))
                isnan = torch.sum(torch.isnan(W).float())
                while isnan > 0:
                    # 注：NMF有时求解会失败，W矩阵全部会nan值，在反向传播时是无效的。一旦出现求解失败的情况，就重新初始化随机参数进行求解。
                    print('nan detected. trying to resolve the nmf.')
                    W, _ = NMF(flat_features, self.K, random_seed=random.randint(0, 255), cuda=True, max_iter=self.max_iter, verbose=False)
                    isnan = torch.sum(torch.isnan(W).float())
                heatmap = W.view(features.size(0), features.size(2), features.size(3), self.K)
                # heatmap = F.softmax(heatmap, dim=-1)
                # heatmap = F.interpolate(heatmap, size=(h, w), mode='bilinear', align_corners=False)
                # heatmap = torch.argmax(heatmap, dim=3)  # [num_views, height, width]
                heatmaps.append(heatmap)
        heatmaps = torch.stack(heatmaps, dim=0)  # [batch_size, num_views, K, height, width]
        heatmaps.requires_grad = False
        return heatmaps



############## unsupervised loss  #####################

def _bilinear_sample(im, x, y, name='bilinear_sampler'):
    """Perform bilinear sampling on im given list of x, y coordinates.
    Implements the differentiable sampling mechanism with bilinear kernel
    in https://arxiv.org/abs/1506.02025.
    x,y are tensors specifying normalized coordinates [-1, 1] to be sampled on im.
    For example, (-1, -1) in (x, y) corresponds to pixel location (0, 0) in im,
    and (1, 1) in (x, y) corresponds to the bottom right pixel in im.
    Args:
        im: Batch of images with shape [B, h, w, channels].
        x: Tensor of normalized x coordinates in [-1, 1], with shape [B, h, w, 1].
        y: Tensor of normalized y coordinates in [-1, 1], with shape [B, h, w, 1].
        name: Name scope for ops.
    Returns:
        Sampled image with shape [B, h, w, channels].
        Principled mask with shape [B, h, w, 1], dtype:float32.  A value of 1.0
        in the mask indicates that the corresponding coordinate in the sampled
        image is valid.
      """
    x = x.reshape(-1)  # [batch_size * height * width]
    y = y.reshape(-1)  # [batch_size * height * width]

    # Constants.
    batch_size, height, width, channels = im.shape

    x, y = x.float(), y.float()
    max_y = int(height - 1)
    max_x = int(width - 1)

    # Scale indices from [-1, 1] to [0, width - 1] or [0, height - 1].
    x = (x + 1.0) * (width - 1.0) / 2.0
    y = (y + 1.0) * (height - 1.0) / 2.0

    # Compute the coordinates of the 4 pixels to sample from.
    x0 = torch.floor(x).int()
    x1 = x0 + 1
    y0 = torch.floor(y).int()
    y1 = y0 + 1

    mask = (x0 >= 0) & (x1 <= max_x) & (y0 >= 0) & (y0 <= max_y)
    mask = mask.float()

    x0 = torch.clamp(x0, 0, max_x)
    x1 = torch.clamp(x1, 0, max_x)
    y0 = torch.clamp(y0, 0, max_y)
    y1 = torch.clamp(y1, 0, max_y)
    dim2 = width
    dim1 = width * height

    # Create base index.
    base = torch.arange(batch_size) * dim1
    base = base.reshape(-1, 1)
    base = base.repeat(1, height * width)
    base = base.reshape(-1)  # [batch_size * height * width]
    base = base.long().to('cuda')

    base_y0 = base + y0.long() * dim2
    base_y1 = base + y1.long() * dim2
    idx_a = base_y0 + x0.long()
    idx_b = base_y1 + x0.long()
    idx_c = base_y0 + x1.long()
    idx_d = base_y1 + x1.long()

    # Use indices to lookup pixels in the flat image and restore channels dim.
    im_flat = im.reshape(-1, channels).float()  # [batch_size * height * width, channels]
    # pixel_a = tf.gather(im_flat, idx_a)
    # pixel_b = tf.gather(im_flat, idx_b)
    # pixel_c = tf.gather(im_flat, idx_c)
    # pixel_d = tf.gather(im_flat, idx_d)
    pixel_a = im_flat[idx_a]
    pixel_b = im_flat[idx_b]
    pixel_c = im_flat[idx_c]
    pixel_d = im_flat[idx_d]

    wa = (x1.float() - x) * (y1.float() - y)
    wb = (x1.float() - x) * (1.0 - (y1.float() - y))
    wc = (1.0 - (x1.float() - x)) * (y1.float() - y)
    wd = (1.0 - (x1.float() - x)) * (1.0 - (y1.float() - y))
    wa, wb, wc, wd = wa.unsqueeze(1), wb.unsqueeze(1), wc.unsqueeze(1), wd.unsqueeze(1)

    output = wa * pixel_a + wb * pixel_b + wc * pixel_c + wd * pixel_d
    output = output.reshape(batch_size, height, width, channels)
    mask = mask.reshape(batch_size, height, width, 1)
    return output, mask


def _spatial_transformer(img, coords):
    """A wrapper over binlinear_sampler(), taking absolute coords as input."""
    img = img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
    # img: [B, H, W, C]
    # img_height = img.shape[1]
    # img_width = img.shape[2]
    px = coords[:, :, :, :1]  # [batch_size, height, width, 1]
    py = coords[:, :, :, 1:]  # [batch_size, height, width, 1]
    # Normalize coordinates to [-1, 1] to send to _bilinear_sampler.
    # px = px / (img_width - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    # py = py / (img_height - 1) * 2.0 - 1.0  # [batch_size, height, width, 1]
    output_img, mask = _bilinear_sample(img, px, py)  # [B, H, W, C]
    output_img = output_img.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
    mask = mask.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
    return output_img, mask


def warping_with_depth(src_fea, ref_depth, src_proj, ref_proj):
    # src_fea: [B, C, H, W]
    # ref_depth: [B, H, W]
    # src_proj: [B, 4, 4]
    # ref_proj: [B, 4, 4]
    # out: [B, C, H, W]

    batch, channels = src_fea.shape[0], src_fea.shape[1]

    batchsize, height, width = ref_depth.shape[0], ref_depth.shape[1], ref_depth.shape[2]

    # with torch.no_grad():
    proj = torch.matmul(src_proj, torch.inverse(ref_proj))  # Tcw
    rot = proj[:, :3, :3]  # [B,3,3]
    trans = proj[:, :3, 3:4]  # [B,3,1]

    y, x = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=src_fea.device),
                           torch.arange(0, width, dtype=torch.float32, device=src_fea.device)])
    y, x = y.contiguous(), x.contiguous()
    y, x = y.view(height * width), x.view(height * width)

    xyz = torch.stack((x, y, torch.ones_like(x)))  # [3, H*W]
    xyz = torch.unsqueeze(xyz, 0).repeat(batch, 1, 1)  # [B, 3, H*W]
    rot_xyz = torch.matmul(rot, xyz)  # [B, 3, H*W]
    rot_depth_xyz = rot_xyz * ref_depth.view(batch, 1, -1)  # [B, 3, H*W]
    proj_xyz = rot_depth_xyz + trans.view(batch, 3, 1)  # [B, 3, H*W]
    proj_xy = proj_xyz[:, :2, :] / proj_xyz[:, 2:3, :]  # [B, 2, H*W]
    proj_x_normalized = proj_xy[:, 0, :] / ((width - 1) / 2) - 1
    proj_y_normalized = proj_xy[:, 1, :] / ((height - 1) / 2) - 1
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=2)  # [B, H*W, 2]
    grid = proj_xy

    # warped_src_fea, mask = _spatial_transformer(src_fea, grid.view(batch, height, width, 2))

    warped_src_fea = F.grid_sample(src_fea, grid.view(batch, height, width, 2), mode='bilinear', padding_mode='zeros')
    warped_src_fea = warped_src_fea.view(batch, channels, height, width)

    mask = torch.sum(warped_src_fea, dim=1).unsqueeze(1) != 0
    mask = mask.repeat(1, channels, 1, 1)  # [B, 3, H, W]

    return warped_src_fea, mask


def gradient_x_depth(depth):
    return depth[:, :, :-1, :] - depth[:, :, 1:, :]


def gradient_y_depth(depth):
    return depth[:, :, :, :-1] - depth[:, :, :, 1:]


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :, :, :-1] - img[:, :, :, 1:]


def gradient(pred):
    D_dy = torch.zeros(pred.shape, device='cuda')  # 全0矩阵
    D_dx = torch.zeros(pred.shape, device='cuda')  # 全0矩阵
    D_dy[:, :, :, 1:] = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    D_dx[:, :, 1:, :] = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy


"""
def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]

def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy
"""


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y, mask):
        # print('mask: {}'.format(mask.shape))
        # print('x: {}'.format(x.shape))
        # print('y: {}'.format(y.shape))
        # x = x.permute(0, 3, 1, 2)  # [B, H, W, C] --> [B, C, H, W]
        # y = y.permute(0, 3, 1, 2)
        # mask = mask.permute(0, 3, 1, 2)

        # x = self.refl(x)
        # y = self.refl(y)
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask)
        output = SSIM_mask * torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)
        # return output.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        return output


def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x_depth(depth)
    depth_dy = gradient_y_depth(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dx), 1, keepdim=True)))
    weights_y = torch.exp(-(lambda_wt * torch.mean(torch.abs(image_dy), 1, keepdim=True)))
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))


def compute_reconstr_loss(warped, ref, mask, simple=True):
    if simple:
        t = torch.abs(warped * mask - ref * mask)
        photo_loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        return photo_loss
    else:
        alpha = 0.5
        ref_dx, ref_dy = gradient(ref * mask)
        warped_dx, warped_dy = gradient(warped * mask)

        grad_loss_x = torch.abs(warped_dx - ref_dx)
        grad_loss_y = torch.abs(warped_dy - ref_dy)
        grad_loss = grad_loss_x + grad_loss_y
        t = torch.abs(warped * mask - ref * mask)
        photo_loss = torch.where(t < 1, 0.5 * t ** 2, t - 0.5)
        # photo_loss = torch.mean(photo_loss, dim=1, keepdim=True)

        """photo_loss = F.smooth_l1_loss(warped * mask, ref * mask, reduction='mean')
        grad_loss = F.smooth_l1_loss(warped_dx, ref_dx, reduction='mean') + \
                     F.smooth_l1_loss(warped_dy, ref_dy, reduction='mean')"""

        return (1 - alpha) * photo_loss + alpha * grad_loss


class UnSupLoss(nn.Module):
    def __init__(self):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()
        self.w_perceptual = [4, 1, 0.5]

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0
        self.perceptual_loss = 0
        # self.w_loss = [2,0.5,0.0, 2]
        # self.w_loss = [12, 6, 0.0, 0.1]
        self.w_loss = [12, 6, 0.1, 0.0]
        # loss_sum = 8 * loss_photo + 2 * loss_ssim + 0.067 * loss_s
        # loss_sum = 1.0 * loss_photo + 2 * loss_ssim + 0.02 * loss_s
        self.w_re = self.w_loss[0]
        self.w_ss = self.w_loss[1]
        self.w_sm = self.w_loss[2]
        self.w_per = self.w_loss[3]

    def forward(self, depth_est, imgs, proj_matrices, outputs_feature, stage_idx, mask_photometric=None):

        # imgs: [B, N, 3, H, W]
        # depth_est: [B, H, W]
        # src_proj: [B, 4, 4]
        # ref_proj: [B, 4, 4]
        # out: Loss
        imgs = torch.unbind(imgs, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)  # 返回切片
        assert len(imgs) == len(proj_matrices), "Different number of images and projection matrices"

        batchsize, height, width = depth_est.shape[0], depth_est.shape[1], depth_est.shape[2]

        ref_img, src_imgs = imgs[0], imgs[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        ref_vgg_feature, src_vgg_feature = outputs_feature[0], outputs_feature[1:]
        # print(outputs_feature[0][3-stage_idx].shape) # B*64*384*768, B*128*192*384, B*256*96*192
        # print(outputs_feature[1][3-stage_idx].shape) # B*64*384*768, B*128*192*384, B*256*96*192
        # print(outputs_feature[2][3-stage_idx].shape) # B*64*384*768, B*128*192*384, B*256*96*192

        # ref_color = ref_img[:, :, 1::4, 1::4]  # B*C*128*160
        ref_color = F.interpolate(ref_img, [height, width], mode='bilinear', align_corners=False)  # False

        self.reconstr_loss = 0
        self.ssim_loss = 0
        self.smooth_loss = 0
        self.perceptual_loss = 0

        warped_img_list = []
        mask_list = []
        reprojection_losses = []

        ##smooth loss##
        self.smooth_loss += depth_smoothness(depth_est.unsqueeze(dim=1), ref_color, 1.0)

        # Loss photo & Loss ssim & loss_perpectual
        for view in range(len(src_imgs)):
            view_img = F.interpolate(src_imgs[view], [height, width], mode='bilinear',
                                     align_corners=False)  # False # [B, C, H, W]
            src_proj = src_projs[view]
            warped_img, mask = warping_with_depth(view_img, depth_est, src_proj, ref_proj)  # [B, C, H, W]
            mask = mask.float()

            warped_img_list.append(warped_img)
            mask_list.append(mask)

            # Loss photo
            reconstr_loss = compute_reconstr_loss(warped_img, ref_color, mask, simple=False)  # [B, C, H, W]
            # print(reconstr_loss.shape)
            valid_mask = 1 - mask  # replace all 0 values with INF  # [B, C, H, W]
            # print(valid_mask.shape)
            reprojection_losses.append(reconstr_loss + 1e4 * valid_mask)  # shape?
            # print(reconstr_loss)

            # SSIM loss
            if view < 3:
                # l_ssim = self.ssim(ref_img, warped_img, mask)
                # self.ssim_loss += torch.mean(l_ssim[mask>0.5])  # torch.mean 是否合适
                self.ssim_loss += torch.mean(self.ssim(ref_color, warped_img, mask))  # torch.mean 是否合适

            # Loss perpectual
            sampled_feature_src, mask_perpectual = warping_with_depth(src_vgg_feature[view][3 - stage_idx], depth_est,
                                                                      src_proj, ref_proj)
            # mask_perpectual = mask_perpectual.float()

            if mask_photometric:
                mask_perpectual = mask_perpectual * mask_photometric

            if F.smooth_l1_loss(ref_vgg_feature[3 - stage_idx][mask_perpectual],
                                sampled_feature_src[mask_perpectual]).nelement() == 0:
                self.perceptual_loss += torch.tensor(0.)
            else:
                self.perceptual_loss += F.smooth_l1_loss(ref_vgg_feature[3 - stage_idx][mask_perpectual],
                                                         sampled_feature_src[mask_perpectual]) * self.w_perceptual[
                                            3 - stage_idx]

        # top-k operates along the last dimension, so swap the axes accordingly
        reprojection_volume = torch.stack(reprojection_losses).permute(1, 2, 3, 4,
                                                                       0)  # shape [K, B, C, H, W] -> [B, C, H, W, K]
        # print(reprojection_volume.shape)
        # by default, it'll return top-k largest entries, hence sorted=False to get smallest entries
        top_vals, top_inds = torch.topk(torch.neg(reprojection_volume), k=2,
                                        sorted=False)  # [B, C, H, W, K]  k=3  in original paper
        top_vals = torch.neg(top_vals)  # [B, C, H, W, K]
        top_mask = top_vals < (1e4 * torch.ones_like(top_vals, device='cuda'))  # [B, C, H, W, K]
        # print(top_vals.shape)
        # print(top_mask.shape)
        top_mask = top_mask.float()
        top_vals = torch.mul(top_vals, top_mask)  # [B, C, H, W, K]

        # top_vals = torch.sum(top_vals, dim=-1) # torch.mean 是否合适
        # top_mask = torch.sum(top_mask, dim=-1) # torch.mean 是否合适
        # self.reconstr_loss = torch.mean(top_vals[top_mask>0.5])) # torch.mean 是否合适
        self.reconstr_loss = torch.mean(torch.sum(top_vals, dim=-1))  # torch.mean 是否合适

        # total weight sum loss
        self.loss_sum = self.w_re * self.reconstr_loss + self.w_ss * self.ssim_loss + self.w_sm * self.smooth_loss + self.w_per * self.perceptual_loss
        # print(loss_sum, loss_photo, loss_ssim, loss_s)

        return self.loss_sum



def compute_seg_loss(warped_seg, ref_seg, mask):
    ref_seg = ref_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
    warped_seg = warped_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
    mask = mask.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]

    # mask = mask.repeat(1, 1, 1, warped_seg.size(3))
    # print('mask: {}'.format(mask.shape))
    # print('warped_seg: {}'.format(warped_seg.shape))
    # print('ref_seg: {}'.format(ref_seg.shape))
    warped_seg_filtered = warped_seg[mask > 0.5]
    ref_seg_filtered = ref_seg[mask > 0.5]
    # print('warped_seg_filtered: {}'.format(warped_seg_filtered.shape))
    # print('ref_seg_filtered: {}'.format(ref_seg_filtered.shape))
    warped_seg_filtered_flatten = warped_seg_filtered.contiguous().view(-1, warped_seg.size(3))  # [B * H * W, C]
    ref_seg_filtered_flatten = ref_seg_filtered.contiguous().view(-1, ref_seg.size(3))  # [B * H * W, C]
    ref_seg_filtered_flatten = torch.argmax(ref_seg_filtered_flatten, dim=1) # [B, H, W]
    loss = F.cross_entropy(warped_seg_filtered_flatten, ref_seg_filtered_flatten, size_average=True)
    return loss



class UnSupSegLoss(nn.Module):
    def __init__(self):
        super(UnSupSegLoss, self).__init__()
        self.seg_model = SegDFF(K=4, max_iter=50)

    def pre_seg(self, imgs):
        # print('imgs: {}'.format(imgs.shape)) # except [batch_size, num_views, C, height, width]
        seg_maps = self.seg_model(imgs)  # # [batch_size, num_views, height, width, K] [1, 5, 14, 14, 4]
        # print(seg_maps.shape)

        seg_maps = seg_maps.permute(0, 1, 4, 2, 3)  #  [batch_size, num_views, K, height, width]

        return seg_maps

    def forward(self, depth_est, seg_maps, proj_matrices):
        """# print('imgs: {}'.format(imgs.shape)) # except [batch_size, num_views, C, height, width]
        seg_maps = self.seg_model(imgs)  # # [batch_size, num_views, height, width, K] [1, 5, 14, 14, 4]
        # print(seg_maps.shape)

        seg_maps = seg_maps.permute(0, 1, 4, 2, 3)  #  [batch_size, num_views, K, height, width]
        # print('seg_maps: {}'.format(seg_maps.shape))"""

        """plt.figure(0)
        de0 = seg_maps[0,0,0:3,:,:].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(de0, cmap='gray_r')
        plt.figure(1)
        de1 = seg_maps[0,1,0:3,:,:].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(de1, cmap='gray_r')
        plt.figure(2)
        de2 = seg_maps[0,2,0:3,:,:].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(de2, cmap='gray_r')
        plt.figure(3)
        de3 = seg_maps[0,3,0:3,:,:].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(de3, cmap='gray_r')
        plt.figure(4)
        de4 = seg_maps[0,4,0:3,:,:].permute(1, 2, 0).detach().cpu().numpy()
        plt.imshow(de4, cmap='gray_r')
        plt.show()"""

        seg_maps = torch.unbind(seg_maps, 1)
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(seg_maps) == len(proj_matrices), "Different number of images and projection matrices"

        batchsize, height, width = depth_est.shape[0], depth_est.shape[1], depth_est.shape[2]
        # height, width = depth.size(1), depth.size(2)

        ref_seg, src_seg = seg_maps[0], seg_maps[1:]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:]
        ref_seg = F.interpolate(ref_seg, size=(height, width), mode='bilinear') # [B, K, H, W]

        warped_seg_list = []
        mask_list = []
        reprojection_losses = []
        view_segs = []

        for view in range(len(src_seg)):
            view_seg = src_seg[view]
            view_seg = F.interpolate(view_seg, size=(height, width), mode='bilinear') # [B, K, H, W]
            src_proj = src_projs[view]
            warped_seg, mask = warping_with_depth(view_seg, depth_est, src_proj, ref_proj)  # [B, K, H, W]
            mask = mask.float()

            # warped_seg: [B, H, W, C]
            # mask: [B, H, W]
            reprojection_losses.append(compute_seg_loss(warped_seg, ref_seg, mask))

            """plt.figure(5)
            de5 = view_seg[0, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(de5, cmap='gray_r')
            plt.figure(6)
            de6= warped_seg[0, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(de6, cmap='gray_r')
            plt.figure(7)
            de7 = mask[0, 0:3, :, :].permute(1, 2, 0).detach().cpu().numpy()
            plt.imshow(de7, cmap='gray_r')
            plt.show()"""

            view_segs.append(view_seg)
            warped_seg_list.append(warped_seg)
            mask_list.append(mask)

        # print(reprojection_losses)
        reproj_seg_loss = sum(reprojection_losses) * 1.0
        #print(reproj_seg_loss)

        # self.ref_seg = ref_seg
        view_segs = torch.stack(view_segs, dim=1) # [B, N, K, H, W]

        ref_seg = ref_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        view_segs = view_segs.permute(0, 1, 3, 4, 2)  # [B, N, C, H, W] --> [B, N, H, W, C]

        return reproj_seg_loss, ref_seg, view_segs


class UnSupSegLossAcc(nn.Module):
    def __init__(self, args):
        super(UnSupSegLossAcc, self).__init__()


    def forward(self, seg_maps, cams, depth):
        seg_maps = torch.unbind(seg_maps, 1)
        cams = torch.unbind(cams, 1)
        height, width = depth.size(1), depth.size(2)
        num_views = len(seg_maps)

        ref_seg = seg_maps[0]
        ref_seg = F.interpolate(ref_seg, size=(height, width), mode='bilinear')
        # ref_seg = ref_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
        ref_cam = cams[0]

        warped_seg_list = []
        mask_list = []
        reprojection_losses = []
        view_segs = []
        for view in range(1, num_views):
            view_seg = seg_maps[view]
            view_seg = F.interpolate(view_seg, size=(height, width), mode='bilinear')
            # view_seg = view_seg.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            view_cam = cams[view]
            view_segs.append(view_seg)
            # print(torch.isnan(view_seg))
            warped_seg, mask = warping_with_depth(view_seg, depth, view_cam, ref_cam)  # [B, C, H, W]
            mask = mask.float()
            # print(torch.isnan(warped_seg))
            warped_seg_list.append(warped_seg)
            mask_list.append(mask)
            # warped_seg: [B, H, W, C]
            # mask: [B, H, W]
            # print('warped_seg: {} ref_seg: {} mask: {}'.format(warped_seg.shape, ref_seg.shape, mask.shape))
            # print('warped_seg: {}'.format(warped_seg))
            # print('mask: {}'.format(mask))

            reprojection_losses.append(compute_seg_loss(warped_seg, ref_seg, mask))
        # print(reprojection_losses)
        reproj_seg_loss = sum(reprojection_losses) * 1.0

        # self.ref_seg = ref_seg
        view_segs = torch.stack(view_segs, dim=1)

        return reproj_seg_loss, ref_seg, view_segs


def aug_loss(depth_est, depth_gt, mask):
    mask = mask > 0.5
    return F.smooth_l1_loss(depth_est[mask], depth_gt[mask], size_average=True)


def cas_loss_unsup(inputs, images, images_seg, depth_gt_ms, proj_matrices_ms, outputs_feature, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)
    unsup_single_loss = UnSupLoss().to('cuda')
    unsupseg_single_loss = UnSupSegLoss().to('cuda')
    seg_maps = unsupseg_single_loss.pre_seg(images_seg)


    """total_loss_sum = torch.tensor(0.0, dtype=torch.float32, device=proj_matrices_ms["stage1"].device, requires_grad=False)
    total_loss_s = torch.tensor(0.0, dtype=torch.float32, device=proj_matrices_ms["stage1"].device,
                                    requires_grad=False)
    total_loss_photo = torch.tensor(0.0, dtype=torch.float32, device=proj_matrices_ms["stage1"].device,
                                    requires_grad=False)
    total_loss_ssim = torch.tensor(0.0, dtype=torch.float32, device=proj_matrices_ms["stage1"].device,
                                    requires_grad=False)"""

    total_loss_sum = 0
    total_loss_s = 0
    total_loss_photo = 0
    total_loss_ssim = 0
    total_loss_perceptual = 0
    total_loss_seg = 0
    # print(src_vgg_feature[0][0].shape) 4*64*512*640
    # print(src_vgg_feature[0][1].shape)  4*128*256*320
    # print(src_vgg_feature[0][2].shape)  4*256*128*160

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:

        depth_est_stage = stage_inputs["depth"]
        # depth_gt_stage = depth_gt_ms[stage_key]
        mask_stage = F.interpolate(stage_inputs["photometric_confidence"].unsqueeze(1), scale_factor=[1, 1],
                                   mode='bilinear', align_corners=False)
        mask_stage = mask_stage > 0.0
        stage_idx = int(stage_key.replace("stage", ""))
        proj_matrices_stage = proj_matrices_ms["stage{}".format(stage_idx)]

        # outputs_feature_stage = outputs_feature[:][3-stage_idx]
        # print(outputs_feature[0][3-stage_idx].shape) # B*64*384*768, B*128*192*384, B*256*96*192

        standard_loss = unsup_single_loss(depth_est_stage, images, proj_matrices_stage, outputs_feature, stage_idx,
                                     mask_photometric=None)

        segment_loss, ref_seg, view_segs = unsupseg_single_loss(depth_est_stage, seg_maps, proj_matrices_stage)
        segment_loss = torch.mean(segment_loss)
        loss_sum = standard_loss + segment_loss * 0.01  # 0.01
        # loss_sum = standard_loss

        # print("sum:{:.3f}  loos_s: {:.3f}  loss_photo: {:.3f}  loss_ssim: {:.3f} loss_perceptual:{:.3f} " .format(depth_loss.detach().cpu().numpy(), loss_s.detach().cpu().numpy(), loss_photo.detach().cpu().numpy(), loss_ssim.detach().cpu().numpy(),loss_perceptual.detach().cpu().numpy() ))
        if depth_loss_weights is not None:
            total_loss_sum += depth_loss_weights[stage_idx - 1] * loss_sum
            total_loss_s += depth_loss_weights[stage_idx - 1] * unsup_single_loss.smooth_loss
            total_loss_photo += depth_loss_weights[stage_idx - 1] * unsup_single_loss.reconstr_loss
            total_loss_ssim += depth_loss_weights[stage_idx - 1] * unsup_single_loss.ssim_loss
            total_loss_perceptual += depth_loss_weights[stage_idx - 1] * unsup_single_loss.perceptual_loss
            total_loss_seg += depth_loss_weights[stage_idx - 1] * segment_loss
        else:
            total_loss_sum += 1.0 * loss_sum
            total_loss_s += 1.0 * unsup_single_loss.smooth_loss
            total_loss_photo += 1.0 * unsup_single_loss.reconstr_loss
            total_loss_ssim += 1.0 * unsup_single_loss.ssim_loss
            total_loss_perceptual += 1.0 * unsup_single_loss.perceptual_loss
            total_loss_seg += 1.0 * segment_loss
        # print(total_loss_sum, depth_loss, total_loss_s, total_loss_photo, total_loss_ssim)


    return total_loss_sum, loss_sum, total_loss_s, total_loss_photo, total_loss_ssim, total_loss_perceptual, total_loss_seg, ref_seg, view_segs


def cas_loss_gt(inputs, depth_gt_ms, mask_ms, **kwargs):
    depth_loss_weights = kwargs.get("dlossw", None)

    total_loss = torch.tensor(0.0, dtype=torch.float32, device=mask_ms["stage1"].device, requires_grad=False)

    for (stage_inputs, stage_key) in [(inputs[k], k) for k in inputs.keys() if "stage" in k]:
        depth_est = stage_inputs["depth"]
        depth_gt = depth_gt_ms[stage_key]
        mask = mask_ms[stage_key]
        mask = mask > 0.5

        depth_loss = F.smooth_l1_loss(depth_est[mask], depth_gt[mask], reduction='mean')

        if depth_loss_weights is not None:
            stage_idx = int(stage_key.replace("stage", "")) - 1
            total_loss += depth_loss_weights[stage_idx] * depth_loss
        else:
            total_loss += 1.0 * depth_loss

    return total_loss, depth_loss




def compute_3dpts_batch(pts, intrinsics):
    pts_shape = pts.shape  # 4*128*160
    batchsize = pts_shape[0]
    height = pts_shape[1]
    width = pts_shape[2]

    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, dtype=torch.float32, device=pts.device),
                                   torch.arange(0, width, dtype=torch.float32, device=pts.device)])

    y_ref, x_ref = y_ref.contiguous(), x_ref.contiguous()
    y_ref, x_ref = y_ref.view(height * width), x_ref.view(height * width)

    xyz_ref = torch.matmul(torch.inverse(intrinsics),
                           torch.stack((x_ref, y_ref, torch.ones_like(x_ref))).unsqueeze(0) * pts.view(batchsize,
                                                                                                       -1).unsqueeze(1))

    xyz_ref = xyz_ref.view(batchsize, 3, height, width)
    xyz_ref = xyz_ref.permute(0, 2, 3, 1)

    return xyz_ref


def compute_normal_by_depth(depth_est, ref_intrinsics, nei):
    ## mask is used to filter the background with infinite depth
    # mask = tf.greater(depth_map, tf.zeros(depth_map.get_shape().as_list())) #我这里好像不存在depth<0的点

    # kitti_shape = depth_map.get_shape().as_list()
    depth_est_shape = depth_est.shape  # 4*128*160
    batchsize = depth_est_shape[0]
    height = depth_est_shape[1]
    width = depth_est_shape[2]

    pts_3d_map = compute_3dpts_batch(depth_est, ref_intrinsics)  # 4*128*160*3
    pts_3d_map = pts_3d_map.contiguous()

    ## shift the 3d pts map by nei along 8 directions
    pts_3d_map_ctr = pts_3d_map[:, nei:-nei, nei:-nei, :]
    pts_3d_map_x0 = pts_3d_map[:, nei:-nei, 0:-(2 * nei), :]
    pts_3d_map_y0 = pts_3d_map[:, 0:-(2 * nei), nei:-nei, :]
    pts_3d_map_x1 = pts_3d_map[:, nei:-nei, 2 * nei:, :]
    pts_3d_map_y1 = pts_3d_map[:, 2 * nei:, nei:-nei, :]
    pts_3d_map_x0y0 = pts_3d_map[:, 0:-(2 * nei), 0:-(2 * nei), :]
    pts_3d_map_x0y1 = pts_3d_map[:, 2 * nei:, 0:-(2 * nei), :]
    pts_3d_map_x1y0 = pts_3d_map[:, 0:-(2 * nei), 2 * nei:, :]
    pts_3d_map_x1y1 = pts_3d_map[:, 2 * nei:, 2 * nei:, :]

    ## generate difference between the central pixel and one of 8 neighboring pixels
    diff_x0 = pts_3d_map_ctr - pts_3d_map_x0  # 因为是求向量，所以不用除以相邻两点之间的距离
    diff_x1 = pts_3d_map_ctr - pts_3d_map_x1
    diff_y0 = pts_3d_map_y0 - pts_3d_map_ctr
    diff_y1 = pts_3d_map_y1 - pts_3d_map_ctr
    diff_x0y0 = pts_3d_map_x0y0 - pts_3d_map_ctr
    diff_x0y1 = pts_3d_map_ctr - pts_3d_map_x0y1
    diff_x1y0 = pts_3d_map_x1y0 - pts_3d_map_ctr
    diff_x1y1 = pts_3d_map_ctr - pts_3d_map_x1y1

    ## flatten the diff to a #pixle by 3 matrix
    # pix_num = kitti_shape[0] * (kitti_shape[1]-2*nei) * (kitti_shape[2]-2*nei)
    pix_num = batchsize * (height - 2 * nei) * (width - 2 * nei)
    # print(pix_num)
    # print(diff_x0.shape)
    diff_x0 = diff_x0.view(pix_num, 3)
    diff_y0 = diff_y0.view(pix_num, 3)
    diff_x1 = diff_x1.view(pix_num, 3)
    diff_y1 = diff_y1.view(pix_num, 3)
    diff_x0y0 = diff_x0y0.view(pix_num, 3)
    diff_x0y1 = diff_x0y1.view(pix_num, 3)
    diff_x1y0 = diff_x1y0.view(pix_num, 3)
    diff_x1y1 = diff_x1y1.view(pix_num, 3)

    ## calculate normal by cross product of two vectors
    normals0 = F.normalize(torch.cross(diff_x1, diff_y1))  # * tf.tile(normals0_mask[:, None], [1,3]) tf.tile=.repeat
    normals1 = F.normalize(torch.cross(diff_x0, diff_y0))  # * tf.tile(normals1_mask[:, None], [1,3])
    normals2 = F.normalize(torch.cross(diff_x0y1, diff_x0y0))  # * tf.tile(normals2_mask[:, None], [1,3])
    normals3 = F.normalize(torch.cross(diff_x1y0, diff_x1y1))  # * tf.tile(normals3_mask[:, None], [1,3])

    normal_vector = normals0 + normals1 + normals2 + normals3
    # normal_vector = tf.reduce_sum(tf.concat([[normals0], [normals1], [normals2], [normals3]], 0),0)
    # normal_vector = F.normalize(normals0)
    normal_vector = F.normalize(normal_vector)
    # normal_map = tf.reshape(tf.squeeze(normal_vector), [kitti_shape[0]]+[kitti_shape[1]-2*nei]+[kitti_shape[2]-2*nei]+[3])
    normal_map = normal_vector.view(batchsize, height - 2 * nei, width - 2 * nei, 3)

    # 对于depth小于0的点，不计算normal
    # normal_map *= tf.tile(tf.expand_dims(tf.cast(mask[:, nei:-nei, nei:-nei], tf.float32), -1), [1,1,1,3])

    # normal_map = tf.pad(normal_map, [[0,0], [nei, nei], [nei, nei], [0,0]] ,"CONSTANT")
    normal_map = F.pad(normal_map, (0, 0, nei, nei, nei, nei), "constant", 0)

    # print(normal_map.shape) #4*128*160*3
    # print(normal_map[0,:,:,0])

    return normal_map


def compute_depth_by_normal(depth_map, normal_map, intrinsics, tgt_image, nei=1):
    depth_init = depth_map.clone()

    d2n_nei = 1  # normal_depth转化的时候的空边
    depth_map = depth_map[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei)]
    normal_map = normal_map[:, d2n_nei:-(d2n_nei), d2n_nei:-(d2n_nei), :]

    # depth_dims = depth_map.get_shape().as_list()
    depth_map_shape = depth_map.shape
    batchsize = depth_map_shape[0]  # 4
    height = depth_map_shape[1]  # 126
    width = depth_map_shape[2]  # 158

    # x_coor = tf.range(nei, depth_dims[2]+nei)
    # y_coor = tf.range(nei, depth_dims[1]+nei)
    # x_ctr, y_ctr = tf.meshgrid(x_coor, y_coor)
    y_ctr, x_ctr = torch.meshgrid(
        [torch.arange(d2n_nei, height + d2n_nei, dtype=torch.float32, device=normal_map.device),
         torch.arange(d2n_nei, width + d2n_nei, dtype=torch.float32, device=normal_map.device)])
    y_ctr, x_ctr = y_ctr.contiguous(), x_ctr.contiguous()

    # x_ctr = tf.cast(x_ctr, tf.float32)
    # y_ctr = tf.cast(y_ctr, tf.float32)
    # x_ctr_tile = tf.tile(tf.expand_dims(x_ctr, 0), [depth_dims[0], 1, 1])
    # y_ctr_tile = tf.tile(tf.expand_dims(y_ctr, 0), [depth_dims[0], 1, 1])
    x_ctr_tile = x_ctr.unsqueeze(0).repeat(batchsize, 1, 1)  # B*height*width
    y_ctr_tile = y_ctr.unsqueeze(0).repeat(batchsize, 1, 1)

    x0 = x_ctr_tile - d2n_nei
    y0 = y_ctr_tile - d2n_nei
    x1 = x_ctr_tile + d2n_nei
    y1 = y_ctr_tile + d2n_nei
    normal_x = normal_map[:, :, :, 0]
    normal_y = normal_map[:, :, :, 1]
    normal_z = normal_map[:, :, :, 2]

    # fx, fy, cx, cy = intrinsics[:,0], intrinsics[:,1], intrinsics[:,2], intrinsics[:,3]
    fx, fy, cx, cy = intrinsics[:, 0, 0], intrinsics[:, 1, 1], intrinsics[:, 0, 2], intrinsics[:, 1, 2]
    cx_tile = cx.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
    cy_tile = cy.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
    fx_tile = fx.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)
    fy_tile = fy.unsqueeze(-1).unsqueeze(-1).repeat(1, height, width)

    numerator = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0 = (x0 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_y0 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1 = (x1 - cx_tile) / fx_tile * normal_x + (y_ctr_tile - cy_tile) / fy_tile * normal_y + normal_z
    denominator_y1 = (x_ctr_tile - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0y0 = (x0 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x0y1 = (x0 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1y0 = (x1 - cx_tile) / fx_tile * normal_x + (y0 - cy_tile) / fy_tile * normal_y + normal_z
    denominator_x1y1 = (x1 - cx_tile) / fx_tile * normal_x + (y1 - cy_tile) / fy_tile * normal_y + normal_z

    mask_x0 = denominator_x0 == 0
    denominator_x0 = denominator_x0 + 1e-3 * mask_x0.float()
    mask_y0 = denominator_y0 == 0
    denominator_y0 = denominator_y0 + 1e-3 * mask_y0.float()
    mask_x1 = denominator_x1 == 0
    denominator_x1 = denominator_x1 + 1e-3 * mask_x1.float()
    mask_y1 = denominator_y1 == 0
    denominator_y1 = denominator_y1 + 1e-3 * mask_y1.float()
    mask_x0y0 = denominator_x0y0 == 0
    denominator_x0y0 = denominator_x0y0 + 1e-3 * mask_x0y0.float()
    mask_x0y1 = denominator_x0y1 == 0
    denominator_x0y1 = denominator_x0y1 + 1e-3 * mask_x0y1.float()
    mask_x1y0 = denominator_x1y0 == 0
    denominator_x1y0 = denominator_x1y0 + 1e-3 * mask_x1y0.float()
    mask_x1y1 = denominator_x1y1 == 0
    denominator_x1y1 = denominator_x1y1 + 1e-3 * mask_x1y1.float()

    depth_map_x0 = numerator / denominator_x0 * depth_map
    depth_map_y0 = numerator / denominator_y0 * depth_map
    depth_map_x1 = numerator / denominator_y0 * depth_map
    depth_map_y1 = numerator / denominator_y0 * depth_map
    depth_map_x0y0 = numerator / denominator_x0y0 * depth_map
    depth_map_x0y1 = numerator / denominator_x0y1 * depth_map
    depth_map_x1y0 = numerator / denominator_x1y0 * depth_map
    depth_map_x1y1 = numerator / denominator_x1y1 * depth_map

    # print(depth_map_x0.shape) #4*126*158

    depth_x0 = depth_init
    depth_x0[:, d2n_nei:-(d2n_nei), :-(2 * d2n_nei)] = depth_map_x0
    depth_y0 = depth_init
    depth_y0[:, 0:-(2 * d2n_nei), d2n_nei:-(d2n_nei)] = depth_map_y0
    depth_x1 = depth_init
    depth_x1[:, d2n_nei:-(d2n_nei), 2 * d2n_nei:] = depth_map_x1
    depth_y1 = depth_init
    depth_y1[:, 2 * d2n_nei:, d2n_nei:-(d2n_nei)] = depth_map_y1
    depth_x0y0 = depth_init
    depth_x0y0[:, 0:-(2 * d2n_nei), 0:-(2 * d2n_nei)] = depth_map_x0y0
    depth_x1y0 = depth_init
    depth_x1y0[:, 0:-(2 * d2n_nei), 2 * d2n_nei:] = depth_map_x1y0
    depth_x0y1 = depth_init
    depth_x0y1[:, 2 * d2n_nei:, 0:-(2 * d2n_nei)] = depth_map_x0y1
    depth_x1y1 = depth_init
    depth_x1y1[:, 2 * d2n_nei:, 2 * d2n_nei:] = depth_map_x1y1

    # --------------------计算权重--------------------------
    tgt_image = tgt_image.permute(0, 2, 3, 1)
    tgt_image = tgt_image.contiguous()  # 4*128*160*3

    # print(depth_map_x0.shape)  #4*124*156
    # normal_map = F.pad(normal_map,(0,0,nei,nei,nei,nei),"constant", 0)

    img_grad_x0 = tgt_image[:, d2n_nei:-d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    # print(img_grad_x0.shape) #4*126*158*3
    img_grad_x0 = F.pad(img_grad_x0, (0, 0, 0, 2 * d2n_nei, d2n_nei, d2n_nei), "constant", 1e-3)
    img_grad_y0 = tgt_image[:, :-2 * d2n_nei, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_y0 = F.pad(img_grad_y0, (0, 0, d2n_nei, d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x1 = tgt_image[:, d2n_nei:-d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1 = F.pad(img_grad_x1, (0, 0, 2 * d2n_nei, 0, d2n_nei, d2n_nei), "constant", 1e-3)
    img_grad_y1 = tgt_image[:, 2 * d2n_nei:, d2n_nei:-d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_y1 = F.pad(img_grad_y1, (0, 0, d2n_nei, d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)

    img_grad_x0y0 = tgt_image[:, :-2 * d2n_nei, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x0y0 = F.pad(img_grad_x0y0, (0, 0, 0, 2 * d2n_nei, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x1y0 = tgt_image[:, :-2 * d2n_nei, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1y0 = F.pad(img_grad_x1y0, (0, 0, 2 * d2n_nei, 0, 0, 2 * d2n_nei), "constant", 1e-3)
    img_grad_x0y1 = tgt_image[:, 2 * d2n_nei:, :-2 * d2n_nei, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x0y1 = F.pad(img_grad_x0y1, (0, 0, 0, 2 * d2n_nei, 2 * d2n_nei, 0), "constant", 1e-3)
    img_grad_x1y1 = tgt_image[:, 2 * d2n_nei:, 2 * d2n_nei:, :] - tgt_image[:, d2n_nei:-d2n_nei, d2n_nei:-d2n_nei, :]
    img_grad_x1y1 = F.pad(img_grad_x1y1, (0, 0, 2 * d2n_nei, 0, 2 * d2n_nei, 0), "constant", 1e-3)

    # print(img_grad_x0.shape) #4*128*160*3

    alpha = 0.1
    weights_x0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0), 3))
    weights_y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y0), 3))
    weights_x1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1), 3))
    weights_y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_y1), 3))

    weights_x0y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y0), 3))
    weights_x1y0 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y0), 3))
    weights_x0y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x0y1), 3))
    weights_x1y1 = torch.exp(-1 * alpha * torch.mean(torch.abs(img_grad_x1y1), 3))

    # print(weights_x0.shape)    #4*128*160
    weights_sum = torch.sum(torch.stack(
        (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1), 0), 0)

    # print(weights.shape) 4*128*160
    weights = torch.stack(
        (weights_x0, weights_y0, weights_x1, weights_y1, weights_x0y0, weights_x1y0, weights_x0y1, weights_x1y1),
        0) / weights_sum
    depth_map_avg = torch.sum(
        torch.stack((depth_x0, depth_y0, depth_x1, depth_y1, depth_x0y0, depth_x1y0, depth_x0y1, depth_x1y1),
                    0) * weights, 0)

    return depth_map_avg


# torch
# @staticmethod
@torch.no_grad()
def reproject_with_depth_torch(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src,
                               extrinsics_src):
    """
    # project the reference point cloud into the source view, then project back
    :param depth_ref: [B, H, W]
    :param intrinsics_ref: [B, 3, 3]
    :param extrinsics_ref: [B, 4, 4]
    :param depth_src: [B, H, W]
    :param intrinsics_src: [B, 3, 3]
    :param extrinsics_src: [B, 4, 4]
    :return:
    """
    # width, height = depth_ref.shape[1], depth_ref.shape[0]
    batch, height, width = depth_ref.shape[0], depth_ref.shape[1], depth_ref.shape[2]
    # step1. project reference pixels to the source view
    # reference view x, y
    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, device=depth_ref.device, dtype=depth_ref.dtype),
                                   torch.arange(0, width, device=depth_ref.device, dtype=depth_ref.dtype)])

    x_ref, y_ref = x_ref.reshape([-1]), y_ref.reshape([-1])  # [H*W], [H*W]

    # reference 3D space
    xyz_ref = torch.matmul(torch.inverse(intrinsics_ref),
                           torch.unsqueeze(torch.stack((x_ref, y_ref, torch.ones_like(x_ref, dtype=x_ref.dtype).cuda()),
                                                       dim=0), 0).repeat(batch, 1, 1) * depth_ref.view(batch, 1,
                                                                                                       -1))  # [3, H*W] ->[B, 3, H*W] * [B, 1, H*W]

    # source 3D space
    xyz_src = torch.matmul(torch.matmul(extrinsics_src, torch.inverse(extrinsics_ref)),
                           torch.cat((xyz_ref,
                                      torch.ones_like(x_ref, dtype=xyz_ref.dtype).cuda().unsqueeze(0).repeat(batch, 1,
                                                                                                             1)),
                                     dim=1))[:, :3,
              :]  # [B, 4, 4] * {[B, 3, H*W] + [B, 1, H*W] -> [B, 4, H*W]} = [B, 4, H*W] ->[B, 3, H*W]

    # source view x, y
    K_xyz_src = torch.matmul(intrinsics_src, xyz_src)  # [B, 3, 3] * [B, 3, H*W]
    xy_src = K_xyz_src[:, :2, :] / K_xyz_src[:, 2:3, :]  # [B, 2, H*W]

    # step2. reproject the source view points with source view depth estimation
    # find the depth estimation of the source view
    x_src = xy_src[:, 0, :].reshape([batch, height, width]).float()  # [B, H, W]
    y_src = xy_src[:, 1, :].reshape([batch, height, width]).float()  # [B, H, W]

    proj_x_normalized = x_src / ((width - 1) / 2) - 1  # [B, H, W]
    proj_y_normalized = y_src / ((height - 1) / 2) - 1  # [B, H, W]
    proj_xy = torch.stack((proj_x_normalized, proj_y_normalized), dim=3)  # [B, H, W, 2]
    grid = proj_xy.float()  # [B, H, W, 2]

    current_version = torch.__version__
    current_version = [int(c) for c in current_version.split(".")]
    default_change_version = [1, 2, 0]
    change_or_not = False

    for i, j in zip(current_version, default_change_version):
        if i > j:
            change_or_not = True
            break

    if change_or_not:
        sampled_depth_src = F.grid_sample(depth_src.float().view(batch, 1, height, width),
                                          grid.view(batch, height, width, 2), mode='bilinear',
                                          padding_mode='zeros', align_corners=True)
    else:
        sampled_depth_src = F.grid_sample(depth_src.float().view(batch, 1, height, width),
                                          grid.view(batch, height, width, 2), mode='bilinear',
                                          padding_mode='zeros')  # [B, 1, H, W]

    # mask = sampled_depth_src > 0

    # source 3D space
    # NOTE that we should use sampled source-view depth_here to project back
    xyz_src = torch.matmul(torch.inverse(intrinsics_src),
                           torch.cat((xy_src, torch.ones_like(x_ref, dtype=xy_src.dtype).cuda().unsqueeze(
                               0).repeat(batch, 1, 1)), dim=1) * sampled_depth_src.view(batch, 1, -1))
    # [B, 3, 3] * {[B, 2, H*W] + [B, 1, H*W] = [B, 3, H*W] * [B, 1, H*W]} = [B, 3, H*W]

    # reference 3D space
    xyz_reprojected = torch.matmul(torch.matmul(extrinsics_ref, torch.inverse(extrinsics_src)), torch.cat(
        (xyz_src, torch.ones_like(x_ref, dtype=x_ref.dtype).cuda().unsqueeze(0).repeat(batch, 1, 1)), dim=1))[:, :3, :]
    # [B, 4, 4] * { [B, 3, H*W] + [B, 1, H*W] -> [B, 4, H*W] } = [B, 4, H*W] -> [B, 3, H*W]

    # source view x, y, depth
    depth_reprojected = xyz_reprojected[:, 2, :].reshape([batch, height, width]).float()  # [B, 1, H*W] -> [B, H, W]
    K_xyz_reprojected = torch.matmul(intrinsics_ref, xyz_reprojected)  # [B, 3, 3] -> [B, 3, H*W]
    xy_reprojected = K_xyz_reprojected[:, :2, :] / K_xyz_reprojected[:, 2:3,
                                                   :]  # [B, 2, H*W] / [B, 1, H*W] = [B, 2, H*W]
    x_reprojected = xy_reprojected[:, 0, :].reshape([batch, height, width]).float()  # [B, 1, H*W] -> [B, H, W]
    y_reprojected = xy_reprojected[:, 1, :].reshape([batch, height, width]).float()  # [B, 1, H*W] -> [B, H, W]

    return depth_reprojected, x_reprojected, y_reprojected, x_src, y_src  # [B, H, W]


# torch
@torch.no_grad()
def check_geometric_consistency_torch(depth_ref, intrinsics_ref, extrinsics_ref, depth_src, intrinsics_src,
                                      extrinsics_src):
    """
    # check the geometric consistency between the reference image and its source images
    :param depth_ref:
    :param intrinsics_ref:
    :param extrinsics_ref:
    :param depth_src:
    :param intrinsics_src:
    :param extrinsics_src:
    :return:
    """
    # width, height = depth_ref.shape[1], depth_ref.shape[0]
    batch, height, width = depth_ref.shape[0], depth_ref.shape[1], depth_ref.shape[2]
    y_ref, x_ref = torch.meshgrid([torch.arange(0, height, device=depth_ref.device, dtype=depth_ref.dtype),
                                   torch.arange(0, width, device=depth_ref.device,
                                                dtype=depth_ref.dtype)])  # [H, W]. [H, W]
    y_ref = y_ref.repeat(batch, 1, 1)  # [B, H, W]
    x_ref = x_ref.repeat(batch, 1, 1)  # [B, H, W]

    depth_reprojected, x2d_reprojected, y2d_reprojected, x2d_src, y2d_src = reproject_with_depth_torch(
        depth_ref,
        intrinsics_ref,
        extrinsics_ref,
        depth_src,
        intrinsics_src,
        extrinsics_src)
    # check |p_reproj-p_1| < 1
    dist = torch.sqrt((x2d_reprojected - x_ref) ** 2 + (y2d_reprojected - y_ref) ** 2)  # [B, H, W]

    # check |d_reproj-d_1| / d_1 < 0.01
    depth_diff = torch.abs(depth_reprojected - depth_ref)  # [B, H, W]
    relative_depth_diff = depth_diff / depth_ref  # [B, H, W]

    """plt.figure(2)
    de3 = dist[0, :, :].detach().cpu().numpy()
    plt.imshow(de3, cmap='gray_r')

    plt.figure(3)
    de3 = relative_depth_diff[0, :, :].detach().cpu().numpy()
    plt.imshow(de3, cmap='gray_r')
    plt.show()"""

    mask = (dist < 0.5) & (relative_depth_diff < 0.001)  # [B, H, W]
    depth_reprojected[~mask] = 0

    return mask, depth_reprojected, x2d_src, y2d_src  # [B, H, W]

    # need GPU


