import numpy as np
import torchvision.utils as vutils
import torch
import torch.nn.functional as F
import os
import re
import errno
import sys


# print arguments
def print_args(args):
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


# torch.no_grad warpper for functions
def make_nograd_func(func):
    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


# convert a function into recursive style to handle nested dict/list/tuple variables
def make_recursive_func(func):
    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars):
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars):
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars):
    if isinstance(vars, torch.Tensor):
        return vars.cuda()
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


def mkdir_p(path):
    try:
        os.makedirs(path)
    except OSError as exc:  # Python ≥ 2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def dict2cuda(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2cuda(v)
        elif isinstance(v, torch.Tensor):
            v = v.cuda()
        new_dic[k] = v
    return new_dic


def dict2numpy(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2numpy(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().numpy().copy()
        new_dic[k] = v
    return new_dic


def dict2float(data: dict):
    new_dic = {}
    for k, v in data.items():
        if isinstance(v, dict):
            v = dict2float(v)
        elif isinstance(v, torch.Tensor):
            v = v.detach().cpu().item()
        new_dic[k] = v
    return new_dic


def save_scalars(logger, mode, scalar_dict, global_step):
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger, mode, images_dict, global_step):
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = '{}/{}'.format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = '{}/{}_{}'.format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


def save_cameras(cam, path):
    cam_txt = open(path, 'w+')

    cam_txt.write('extrinsic\n')
    for i in range(4):
        for j in range(4):
            cam_txt.write(str(cam[0, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.write('\n')

    cam_txt.write('intrinsic\n')
    for i in range(3):
        for j in range(3):
            cam_txt.write(str(cam[1, i, j]) + ' ')
        cam_txt.write('\n')
    cam_txt.close()


def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def write_pfm(file, image, scale=1):
    file = open(file, 'wb')
    color = None
    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3: # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1: # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image_string = image.tostring()
    file.write(image_string)
    file.close()


class DictAverageMeter(object):
    def __init__(self):
        self.data = {}
        self.count = 0

    def update(self, new_input):
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self):
        return {k: v / self.count for k, v in self.data.items()}


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image(metric_func):
    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


# a wrapper to compute metrics for each image individually
def compute_metrics_for_each_image2(metric_func):
    def wrapper(depth_est, depth_gt, interval, mask, *args):
        batch_size = depth_gt.shape[0]
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], interval, mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(depth_est, depth_gt, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors < thres
    return torch.mean(err_mask.float())


@make_nograd_func
@compute_metrics_for_each_image2
def Inter_metrics(depth_est, depth_gt, interval, mask, thres):
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs((depth_est - depth_gt))
    errors = errors / interval
    err_mask = errors < thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est, depth_gt, mask, depth_threshold):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    # print(depth_est.shape)
    diff = (depth_est - depth_gt).abs()
    mask2 = (diff < depth_threshold)
    result = diff[mask2]
    return torch.mean(result)


# RMSE（均方根误差）阈值版本
@make_nograd_func
@compute_metrics_for_each_image
def RMSE_metrics(depth_est, depth_gt, mask, depth_threshold):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = depth_est - depth_gt  # 计算差值（保留正负）
    mask2 = (diff.abs() < depth_threshold)  # 筛选误差小于阈值的区域
    squared_diff = diff[mask2] ** 2  # 平方误差
    return torch.sqrt(torch.mean(squared_diff))  # 均值的平方根


# Relative MAE（相对平均绝对误差）
# @make_nograd_func
# @compute_metrics_for_each_image
# def RelativeMAE_metrics(depth_est, depth_gt, mask, depth_threshold,range):
#     depth_est, depth_gt = depth_est[mask], depth_gt[mask]
#     diff = (depth_est - depth_gt).abs()
#     mask2 = (diff < depth_threshold)
#     print(depth_gt[mask2],range)

#     rel_diff = diff[mask2] / (depth_gt[mask2] + 1e-6)  # 添加小偏移量避免除零
#     return torch.mean(rel_diff)

@make_nograd_func
@compute_metrics_for_each_image
def RelativeMAE_metrics(depth_est, depth_gt, mask, depth_threshold):
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    diff = (depth_est - depth_gt).abs()
    mask2 = (diff < depth_threshold)

    # 用深度范围（极差）归一化
    depth_range = depth_gt[mask2].max() - depth_gt[mask2].min()  # 计算极差并避免除零
    rel_diff = diff[mask2] / depth_range
    # print(depth_range)

    return torch.mean(rel_diff)

import torch
from torch.nn.functional import conv2d

# 禁用梯度计算
def make_nograd_func(func):
    def wrapper(*args, **kwargs):
        with torch.no_grad():
            return func(*args, **kwargs)
    return wrapper

# 对每个图像分别计算指标
def compute_metrics_for_each_image(func):
    def wrapper(depth_est, depth_gt, mask, *args, **kwargs):
        # 如果深度图是 [H, W]（无批次），直接计算
        if depth_est.dim() == 2 and depth_gt.dim() == 2:
            # 确保掩码是 [H, W]
            if mask.dim() == 3 and mask.shape[0] == 1:
                mask = mask.squeeze(0)  # 从 [1, H, W] 转换为 [H, W]
            elif mask.dim() != 2:
                raise ValueError(f"Expected mask of shape [H, W] or [1, H, W], got {mask.shape}")
            return func(depth_est, depth_gt, mask, *args, **kwargs)
        # 如果深度图是 [B, H, W]（批处理），逐个计算
        elif depth_est.dim() == 3 and depth_gt.dim() == 3:
            assert mask.dim() == 3, f"Expected mask of shape [B, H, W], got {mask.shape}"
            batch_size = depth_est.shape[0]
            assert mask.shape[0] == batch_size, "Batch size mismatch between depth and mask"
            results = []
            for i in range(batch_size):
                result = func(depth_est[i], depth_gt[i], mask[i], *args, **kwargs)
                results.append(result)
            return torch.stack(results).mean()  # 返回批次平均值
        else:
            raise ValueError(f"Unsupported input shapes: depth_est {depth_est.shape}, depth_gt {depth_gt.shape}")
    return wrapper

@make_nograd_func
@compute_metrics_for_each_image
def EdgeAccuracy_metrics1(depth_est, depth_gt, mask, depth_threshold, edge_threshold=0.1):
    """
    计算边缘区域的准确性（Edge Accuracy）。
    
    参数：
        depth_est: 估计深度图 [H, W]
        depth_gt: 地面真相深度图 [H, W]
        mask: 有效区域掩码 [H, W]
        depth_threshold: 误差阈值
        edge_threshold: 边缘检测的梯度阈值（默认0.1，可调整）
    
    返回：
        边缘区域内误差小于 depth_threshold 的平均绝对误差
    """
    # 确保输入是二维张量 [H, W]
    # print(depth_est.shape,depth_gt.shape)

    assert depth_est.dim() == 2 and depth_gt.dim() == 2, "Expected [H, W] input for each image"
    assert mask.dim() == 2, f"Expected [H, W] mask, got {mask.shape}"
    
    # 确保 mask 是 bool 类型
    mask = mask.bool()
    
    # 计算地面真相深度图的梯度，用于边缘检测
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    # 将 Sobel 算子移动到与输入相同的设备
    sobel_x = sobel_x.to(depth_gt.device)
    sobel_y = sobel_y.to(depth_gt.device)
    
    # 将深度图转换为 [1, 1, H, W] 格式以进行卷积
    depth_gt_2d = depth_gt.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 计算梯度
    grad_x = conv2d(depth_gt_2d, sobel_x, padding=1)  # 水平梯度 [1, 1, H, W]
    grad_y = conv2d(depth_gt_2d, sobel_y, padding=1)  # 垂直梯度 [1, 1, H, W]
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()  # 梯度幅度 [H, W]
    
    # 边缘掩码：梯度大于阈值的区域视为边缘
    edge_mask = (grad_magnitude > edge_threshold)  # [H, W]
    
    # 结合原始 mask 和 edge_mask
    final_mask = mask & edge_mask  # 有效区域且为边缘的掩码 [H, W]
    
    # 如果没有边缘区域，返回 0
    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=depth_est.device)
    
    # 计算整个深度图的绝对误差
    diff = (depth_est - depth_gt).abs()  # [H, W]
    diff_masked = diff[final_mask]  # 提取边缘区域的误差，展平为一维 [M]
    
    # 筛选误差小于 depth_threshold 的部分
    mask2 = (diff_masked < depth_threshold)
    result = diff_masked[mask2]
    
    # 返回边缘区域的平均绝对误差
    return torch.mean(result) if result.numel() > 0 else torch.tensor(0.0, device=depth_est.device)


import torch
from torch.nn.functional import conv2d
import matplotlib.pyplot as plt

# 假设这些装饰器已定义
@make_nograd_func
@compute_metrics_for_each_image
def EdgeAccuracy_metrics(depth_est, depth_gt, mask, edge_threshold, depth_threshold, output_path="edge_visualization.png"):
    # 输入检查
    # print("depth_est shape:", depth_est.shape, "depth_gt shape:", depth_gt.shape)
    assert depth_est.dim() == 2 and depth_gt.dim() == 2, "Expected [H, W] input for each image"
    assert mask.dim() == 2, f"Expected [H, W] mask, got {mask.shape}"
    
    # 确保 mask 是 bool 类型
    mask = mask.bool()
    
    # 计算地面真相深度图的梯度，用于边缘检测
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32).view(1, 1, 3, 3)
    
    sobel_x = sobel_x.to(depth_gt.device)
    sobel_y = sobel_y.to(depth_gt.device)
    
    depth_gt_2d = depth_gt.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    
    # 计算梯度
    grad_x = conv2d(depth_gt_2d, sobel_x, padding=1)
    grad_y = conv2d(depth_gt_2d, sobel_y, padding=1)
    grad_magnitude = torch.sqrt(grad_x**2 + grad_y**2).squeeze()  # [H, W]
    
    # 边缘掩码
    edge_mask = (grad_magnitude > edge_threshold)  # [H, W]
    
    # # 调试：检查 edge_mask 的值
    # print("Edge mask unique values:", edge_mask.unique())
    # print("Number of edge pixels (True):", edge_mask.sum().item())
    
    # 结合原始 mask 和 edge_mask
    final_mask = mask & edge_mask
    
    # # 计算边缘区域占比
    # total_valid_pixels = mask.sum().item()  # 原始 mask 中有效像素数
    # edge_pixels = edge_mask.sum().item()    # 边缘像素数（基于 edge_mask）
    # final_edge_pixels = final_mask.sum().item()  # 结合 mask 后的边缘像素数
    
    # if total_valid_pixels > 0:
    #     edge_ratio = edge_pixels / total_valid_pixels  # 边缘区域占总有效区域的比例（基于 edge_mask）
    #     final_edge_ratio = final_edge_pixels / total_valid_pixels  # 结合 mask 后的边缘占比
    #     print(f"Edge area ratio (edge_mask): {edge_ratio:.4f} ({edge_pixels}/{total_valid_pixels})")
    #     print(f"Final edge area ratio (final_mask): {final_edge_ratio:.4f} ({final_edge_pixels}/{total_valid_pixels})")
    # else:
    #     print("No valid pixels in mask, cannot compute edge ratio.")
    
    # # 可视化边缘并保存
    # plt.figure(figsize=(15, 5))
    
    # # 子图1：原始深度图
    # plt.subplot(1, 3, 1)
    # plt.imshow(depth_gt.cpu().numpy(), cmap='gray')
    # plt.title("Ground Truth Depth")
    # plt.axis('off')
    
    # # 子图2：梯度幅度
    # plt.subplot(1, 3, 2)
    # plt.imshow(grad_magnitude.cpu().numpy(), cmap='hot')
    # plt.title("Gradient Magnitude")
    # plt.axis('off')
    
    # # 子图3：检测到的边缘，明确指定颜色
    # plt.subplot(1, 3, 3)
    # edge_mask_vis = edge_mask.cpu().numpy().astype(float)
    # plt.imshow(edge_mask_vis, cmap='gray', vmin=0, vmax=1)  # 0=黑色，1=白色
    # plt.title("Detected Edges (edge_mask)")
    # plt.axis('off')
    
    # plt.tight_layout()
    # plt.savefig(output_path, dpi=300, bbox_inches='tight')
    # plt.close()
    
    # print(f"Visualization saved to {output_path}")
    # print("In the 'Detected Edges' plot: White = Edges, Black = Non-Edges")
    
    # 原有误差计算逻辑
    if final_mask.sum() == 0:
        return torch.tensor(0.0, device=depth_est.device)
    
    diff = (depth_est - depth_gt).abs()
    diff_masked = diff[final_mask]
    mask2 = (diff_masked < depth_threshold)
    result = diff_masked[mask2]
    
    return torch.mean(result) if result.numel() > 0 else torch.tensor(0.0, device=depth_est.device)


