import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np
import time
from tensorboardX import SummaryWriter
from datasets import find_dataset_def
from models import *
from utils import *
import gc
import sys
import datetime
import matplotlib.pyplot as plt
from datasets.data_io import read_pfm, save_pfm, write_cam
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='A PyTorch Implementation of casmvsnet and casrednet')
parser.add_argument('--mode', default='test', help='train or test', choices=['train', 'test' ,'draw'])
# parser.add_argument('--model', default='casmvsnet', help='select model', choices=['casmvsnet, casrednet1, ada_mvs, ucsnet, casrmvsnet'])
parser.add_argument('--model', default='casmvsnet', help='select model')

parser.add_argument('--supervised', type=bool, default=True, help='whether to train with ground truth.')

parser.add_argument('--set_name', default='whu_omvs', help='give the dataset name')
parser.add_argument('--dataset', default='cas_whuomvs', help='select dataset')
# parser.add_argument('--dataset', default='cas_total_rscv', help='select dataset')

# dataset and trained model path
parser.add_argument('--trainpath', default='/opt/data/private/WHU-OMVS/train', help='train datapath')
parser.add_argument('--testpath', default='/opt/data/private/WHU-OMVS/test', help='test datapath')
parser.add_argument('--loadckpt', default=None, help='load a specific checkpoint')
# parser.add_argument('--loadckpt', default='./checkpoints/adamvs/casvis2d_meitan_oblique_5_13/model_000019_meitan_ob_0.1339_13.ckpt', help='load a specific checkpoint')
parser.add_argument('--logdir', default='./checkpoints/try1', help='the directory to save checkpoints/logs')

# input parameters
parser.add_argument('--normalize', type=str, default='mean', help='methods of center_image, mean[mean var] or standard[0-1].') # attention: CasMVSNet [0-255];; CasREDNet [mean var]
parser.add_argument('--view_num', type=int, default=5, help='Number of images (1 ref image and view_num - 1 view images).')
parser.add_argument('--max_w', type=int, default=768, help='Maximum image width')
parser.add_argument('--max_h', type=int, default=384, help='Maximum image height')
parser.add_argument('--resize_scale', type=float, default=1, help='output scale for depth and image (W and H)')
parser.add_argument('--sample_scale', type=float, default=1, help='Downsample scale for building cost volume (W and H)')
parser.add_argument('--interval_scale', type=float, default=1, help='the number of depth values')

# Cascade parameters
parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')  # 2021-04-20
parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
parser.add_argument('--min_interval', type=float, default=0.1, help='min_interval in the bottom stage')
parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
parser.add_argument('--dlossw', type=str, default="0.5,1.0,2.0", help='depth loss weight for different stage')
parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')

# network architecture
parser.add_argument('--epochs', type=int, default=50, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,12,14:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=0.0, help='weight decay')
parser.add_argument('--summary_freq', type=int, default=50, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')
parser.add_argument('--batch_size', type=int, default=1, help='train batch size')
parser.add_argument('--adaptive_scaling', type=bool, default=True, help='Let image size to fit the network, including scaling and cropping')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

# parse arguments and check
args = parser.parse_args()
if args.resume:
    assert args.mode == "train"
    assert args.loadckpt is None
if args.testpath is None:
    args.testpath = args.trainpath
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create logger for mode "train" and "testall"
if args.mode == "train":
    if not os.path.isdir(args.logdir):
        os.mkdir(args.logdir)

    current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    print("current time", current_time_str)

    print("creating new summary file")
    logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)



# dataset, dataloader
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.set_name, "train", args.view_num, args.normalize, args)
test_dataset = MVSDataset(args.testpath, args.set_name, "test", args.view_num, args.normalize, args)
# train_dataset = MVSDataset(args.trainpath, "train", args.view_num, args.normalize, args)
# test_dataset = MVSDataset(args.testpath,  "test", args.view_num, args.normalize, args)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=1, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=1, drop_last=False)

# model, optimizer
# build model
model = None
# CascadeREDNet, cas_mvsnet_loss, Infer_CascadeREDNet = find_model_def(args.model)
if args.model == 'casmvsnet':
    from models.cas_mvsnet import CascadeMVSNet, cas_mvsnet_loss
    model = CascadeMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss

elif args.model == 'casrmvsnet':
    from models.casrmvsnet import CascadeRMVSNet, cas_mvsnet_loss, Infer_CascadeRMVSNet
    model = CascadeRMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss  # CascadeRMVSNet

elif args.model == 'casrednet1':
    from models.casrednet1 import CascadeREDNet, cas_mvsnet_loss, Infer_CascadeREDNet
    model = CascadeREDNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss  # CascadeRMVSNet

elif args.model == 'ucsnet':
    from models.ucsnet import UCSNet, cas_mvsnet_loss
    model = UCSNet(lamb=1.5, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss  # CascadeRMVSNet

elif args.model == 'adamvs':
    from models.adamvs import AdaMVSNet, cas_mvs_vis_loss
    model = AdaMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_intervals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvs_vis_loss  # CascadeRMVSNet



elif args.model == 'casmvsnet_zbf':
    from models.cas_mvsnet_zbf_uncs import CascadeMVSNet, cas_mvsnet_loss
    model = CascadeMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss

elif args.model == 'dino':
    from models.cas_mvsnet_zbf_DINO import CascadeMVSNet, cas_mvsnet_loss
    model = CascadeMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss
    # model_loss_old = cas_mvsnet_loss_old


    # for param in model.feature.parameters(): param.requires_grad = True

    for name, param in model.diffusion.named_parameters():
        param.requires_grad = False

    for name, param in model.feature.named_parameters():
        # 打印参数的名字和内容
        # print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")

        # 判断名字中是否包含 'attention_block'
        if 'diffusion' in name:
            # 如果包含 'attention_block'，冻结该参数
            param.requires_grad = False
            # print(f"Frozen parameter: {name}")

        else:
            # 否则，保持该参数的梯度计算
            param.requires_grad = True
            # print(f"Unfrozen parameter: {name}")
    # for param in model.cost_regularization.parameters(): param.requires_grad = False

    # # 遍历 model.cost_regularization 下的所有参数
    for name, param in model.cost_regularization.named_parameters():
        # 打印参数的名字和内容
        # print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")

        # 判断名字中是否包含 'attention_block'
        if 'diffusion' in name:
            # 如果包含 'attention_block'，冻结该参数
            param.requires_grad = False
            # print(f"Frozen parameter: {name}")
        else:
            # 否则，保持该参数的梯度计算
            param.requires_grad = True
            # print(f"Unfrozen parameter: {name}")

    # for param in model.parameters(): param.requires_grad = False

    for param in model.DepthNet.parameters(): param.requires_grad = True
    for param in model.refine_network.parameters(): param.requires_grad = True


    for name, param in model.named_parameters():
        print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        # param.requires_grad = False


    # print(model)



elif args.model == 'casmvsnet_zbf_dep':
    from models.cas_mvsnet_zbf_depany import CascadeMVSNet, cas_mvsnet_loss
    model = CascadeMVSNet(ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                          depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                          share_cr=args.share_cr,
                          cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch])
    model_loss = cas_mvsnet_loss
# else:
#     raise Exception("{}? Not implemented yet!".format(args.model))


if not args.supervised:
    from losses.unsup_loss1 import cas_loss_unsup, vggNet
    model_feature = vggNet()
    cas_loss_unsup = cas_loss_unsup

if args.mode in ["train", "test"]:
    model = nn.DataParallel(model)
    if not args.supervised:
        model_feature = nn.DataParallel(model_feature)


model.cuda()
# optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)
optimizer = optim.RMSprop(model.parameters(), lr=args.lr, alpha=0.9, weight_decay=args.wd)

# load parameters
start_epoch = 0
if (args.mode == "train" and args.resume) or (args.mode == "test" and not args.loadckpt):
    saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
    saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # use the latest checkpoint file
    loadckpt = os.path.join(args.logdir, saved_models[-1])
    print("resuming", loadckpt)
    state_dict = torch.load(loadckpt)
    model.load_state_dict(state_dict['model'], strict=False)


    # optimizer.load_state_dict(state_dict['optimizer'])

    start_epoch = state_dict['epoch'] + 1


    #
    # # 加载预训练权重
    # state_dict = \
    # torch.load('/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/Dino_edge——refine3/model_000012_0.1332.ckpt')[
    #     'model']
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    #
    # refine_network_dict = {k.replace('refine_network.', ''): v for k, v in state_dict.items() if k.startswith('refine_network.')}
    #
    # # model.refine_network.load_state_dict(refine_network_dict)
    # # model.module.refine_network.load_state_dict(refine_network_dict)
    #
    # # print("refine_network 部分权重加载成功")2
    # for key, value in refine_network_dict.items():
    #     print(f"refine_network -> Key: {key}, Value Shape: {value.shape}")
    #
    # # for param in model.module.refine_network.parameters(): param.requires_grad = False



elif args.loadckpt:
    # load checkpoint file specified by args.loadckpt
    print("loading model {}".format(args.loadckpt))
    state_dict = torch.load(args.loadckpt)

    model_state_dict = model.state_dict()
    filtered_state_dict = {k: v for k, v in state_dict['model'].items() if
                           k in model_state_dict and model_state_dict[k].shape == v.shape}
    model.load_state_dict(filtered_state_dict, strict=False)


        # pretrained_diffusion = torch.load('/opt/data/private/2025code/MVS_lowLT/MVS_lowLT/model_000000_207.4482.ckpt',
        #                map_location='cuda')["model"]


        # model_diffusion = self.diffusion.state_dict()

        # pretrained_diffusion = {k.replace('module.diffusion.', ''): v for k, v in pretrained_diffusion.items() if k.startswith("module.diffusion")}
        # print(".........")
        # # print(pretrained_diffusion)
        # for name, param in self.diffusion.named_parameters():
        #     print(f"Parameter name: {name}, requires_grad: {param.requires_grad}")
        #     # param.requires_grad = False
        # model_diffusion.update(pretrained_diffusion)
        # self.diffusion.load_state_dict(model_diffusion, strict=True)

        # self.diffusion.eval()


    # 加载预训练权重
    state_dict = \
    torch.load('/opt/data/private/2025code/MVS_lowLT/1_5.7wt_/model_000010_33.9323.ckpt')[
        'model']
    state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    refine_network_dict = {k.replace('diffusion.', ''): v for k, v in state_dict.items() if k.startswith('diffusion.')}

    # model.refine_network.load_state_dict(refine_network_dict)
    model.module.diffusion.load_state_dict(refine_network_dict, strict=True)


    # model.load_state_dict(state_dict['model'], strict=False)
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))






    # #
    # # 加载预训练权重
    # state_dict = \
    # torch.load('/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/Dino_edge——refine3/model_000012_0.1332.ckpt')[
    #     'model']
    # state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    # refine_network_dict = {k.replace('refine_network.', ''): v for k, v in state_dict.items() if k.startswith('refine_network.')}
    
    # model.module.refine_network.load_state_dict(refine_network_dict)



    # model.load_state_dict(state_dict['model'], strict=False)
print("start at epoch {}".format(start_epoch))
print('Number of model parameters: {}'.format(sum([p.data.nelement() for p in model.parameters()])))






# main function
def train():
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])

    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma,last_epoch=start_epoch - 1)

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma)

    for epoch_idx in range(start_epoch, args.epochs):
        print('Epoch {}:'.format(epoch_idx))
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx

        # training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                # save_images(logger, 'train', image_outputs, global_step)
            if batch_idx%1000==0 or batch_idx%10==0 and batch_idx<100:
                print(
                    'Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}, train_result = {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                         len(TrainImgLoader), loss,
                                                                                         time.time() - start_time, scalar_outputs))
            del scalar_outputs, image_outputs
        torch.save({'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}, "{}/model_{:0>6}.ckpt".format(args.logdir, epoch_idx))

        # testing
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            # loss, scalar_outputs, image_outputs, saved_outputs, saved_index = test_sample(sample, detailed_summary=do_summary)
            loss, scalar_outputs, image_outputs, saved_outputs = test_sample(sample, detailed_summary=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            if batch_idx % 100 == 0:
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}, {}'.format(epoch_idx, args.epochs, batch_idx,
                                                                                         len(TestImgLoader), loss,
                                                                                         time.time() - start_time, scalar_outputs))
            # del scalar_outputs, image_outputs, saved_outputs, saved_index
            del scalar_outputs, image_outputs, saved_outputs
        save_scalars(logger, 'fulltest', avg_test_scalars.mean(), global_step)
        print("avg_test_scalars:", avg_test_scalars.mean())
        # abs_depth_error = avg_test_scalars.mean()["abs_depth_acc"]#源代码报错？
        abs_depth_error = avg_test_scalars.mean()["abs_depth_error"]

        # saved record to txt
        train_record = open(args.logdir + '/train_record.txt', "a+")
        train_record.write(str(epoch_idx) + ' ' + str(avg_test_scalars.mean()) + '\n')
        train_record.close()

        # checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()},
                "{}/model_{:0>6}_{:.4f}.ckpt".format(args.logdir, epoch_idx, abs_depth_error))

        # gc.collect()alars:", avg_test_scalars.mean())
        # gc.collect()


def test():
    # create output folder
    output_folder = os.path.join(args.testpath, 'depths_{}'.format(args.model))
    if not os.path.isdir(output_folder):
        os.mkdir(output_folder)
    list_index = []
    # f = open(output_folder + '/index.txt', "w")



    avg_test_scalars = DictAverageMeter()
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        # loss, scalar_outputs, image_outputs, saved_outputs, saved_index = test_sample(sample, detailed_summary=True)
        loss, scalar_outputs, image_outputs, saved_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        scalar_outputs = {k: float("{0:.6f}".format(v)) for k, v in scalar_outputs.items()}
        print("Iter {}/{}, time = {:3f}, test results = {}".format(batch_idx, len(TestImgLoader), time.time() - start_time, scalar_outputs))

        # save results
        # depth_est = np.float32(np.squeeze(tensor2numpy(image_outputs["depth_est"])))
        # prob = np.float32(np.squeeze(tensor2numpy(image_outputs["photometric_confidence"])))
        # ref_image = np.squeeze(tensor2numpy(saved_outputs["outimage"]))
        # ref_cam = np.squeeze(tensor2numpy((saved_outputs["outcam"])))
        # # out_location = np.squeeze(saved_outputs["outcam"])

        # ##  aerial dataset
        # vid = saved_outputs["out_view"][0]
        # name = saved_outputs["out_name"][0]
        #
        # # paths
        # output_folder2 = output_folder + ('/%s/' % vid)
        # if not os.path.exists(output_folder2+'/color/'):
        #     os.mkdir(output_folder2)
        #     os.mkdir(output_folder2+'/color/')
        #
        # init_depth_map_path = output_folder2 + ('%s_init.pfm' % name)
        # prob_map_path = output_folder2 + ('%s_prob.pfm' % name)
        # out_ref_image_path = output_folder2 + ('%s.jpg' % name)
        # out_ref_cam_path = output_folder2 + ('%s.txt' % name)
        #
        # if vid not in list_index:
        #     # if (list_index.index(out_index)==-1):
        #     ref_cam[0] = str(args.max_w)
        #     ref_cam[1] = str(args.max_h)
        #     list_index.append(vid)
        #     for word in ref_cam:
        #         f.write(str(word) + ' ')
        #     f.write('\n')
        #
        #
        # # save output
        # save_pfm(init_depth_map_path, depth_est)
        # save_pfm(prob_map_path, prob)
        # plt.imsave(out_ref_image_path, ref_image, format='jpg')
        # write_cam(out_ref_cam_path, ref_cam, ref_cam)
        #
        # size1 = len(depth_est)
        # size2 = len(depth_est[1])
        # e = np.ones((size1, size2), dtype=np.float32)
        # out_init_depth_image = e * 36000 - depth_est
        # plt.imsave(output_folder2 + ('/color/%s_init.png' % name), out_init_depth_image, format='png')
        # plt.imsave(output_folder2 + ('/color/%s_prob.png' % name), prob, format='png')
        #
        # del scalar_outputs, image_outputs, saved_outputs

    print("final, time = {:3f}, test results = {}".format(time.time() - start_time, avg_test_scalars.mean()))


def train_sample(sample, detailed_summary=False):
    model.train()
    optimizer.zero_grad()

    sample_cuda = tocuda(sample)
    # depth_gt_ms = sample_cuda["depth"]
    depth_gt_ms = sample_cuda["depth"]

    mask_ms = sample_cuda["mask"]
    depth_interval = sample_cuda["depth_interval"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], )
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"], test=False)

    depth_est = outputs["depth"]
    # depth_est = outputs["refined_depth"]




    # depth_est = outputs["prob_volume"]


    # print(depth_est.shape)
    # return {"depth": depth, "photometric_confidence": photometric_confidence, 'variance': exp_variance,
    #         'prob_volume': prob_volume, 'depth_values': depth_values}
    # loss = 0.0
    depth_loss = 0.0
    loss_s = 0.0
    loss_photo = 0.0
    loss_ssim = 0.0
    loss_perceptual = 0.0

    if args.supervised:
        loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms,
                                      dlossw=[float(e) for e in args.dlossw.split(",") if e])
        # print(loss)
    # else:
    #     with torch.no_grad():
    #         # print("Begin VGG16 extract feature")
    #         imgs_features = sample_cuda["imgs"].clone()
    #         imgs_features = torch.unbind(imgs_features, 1)  # nivews个4*3*512*640

    #         mean = torch.tensor([[[0.485]], [[0.456]], [[0.406]]], device='cuda')
    #         std = torch.tensor([[[0.229]], [[0.224]], [[0.225]]], device='cuda')
    #         imgs_features = [(img - mean) / std for img in imgs_features]

    #         outputs_feature = [model_feature(img) for img in imgs_features]
    #     loss, depth_loss, loss_s, loss_photo, loss_ssim, loss_perceptual, warped_mask = cas_loss_unsup(outputs, sample_cuda["imgs"], depth_gt_ms,
    #                                                                      sample_cuda["proj_matrices"], outputs_feature,
    #                                                                      dlossw=[float(e) for e in
    #                                                                              args.dlossw.split(",") if e])

    # print(loss.requires_grad)  # 应该为 True
    # print(loss.grad_fn)        # 不应该为 None
    loss.backward()
    optimizer.step()

    scalar_outputs = {"loss": loss, "depth_loss": depth_loss, "loss_s": loss_s, "loss_photo": loss_photo, "loss_ssim": loss_ssim, "loss_perceptual": loss_perceptual}
    image_outputs = {"depth_est": depth_est, "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0]}
    # if detailed_summary:
        # image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask
        # scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5, float(depth_interval * 100.0))
        # scalar_outputs["thres1interval_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, float(depth_interval * 1.0))
        # scalar_outputs["thres6interval_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, float(depth_interval * 6.0))
        # scalar_outputs["thres3interval_error"] = Inter_metrics(depth_est, depth_gt, depth_interval, mask > 0.5, 3)

    # 在调用.backward()之前



    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@make_nograd_func
def test_sample(sample, detailed_summary=True):
    model.eval()

    sample_cuda = tocuda(sample)
    depth_gt_ms = sample_cuda["depth"]
    mask_ms = sample_cuda["mask"]
    # depth_interval = sample_cuda["depth_interval"]

    num_stage = len([int(nd) for nd in args.ndepths.split(",") if nd])
    depth_gt = depth_gt_ms["stage{}".format(num_stage)]
    mask = mask_ms["stage{}".format(num_stage)]

    # depth_values = sample_cuda["depth_values"]
    # depth_min = float(depth_values[0, 0].cpu().numpy())
    # depth_max = float(depth_values[0, -2].cpu().numpy())
    # range = depth_max - depth_min


    # outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"])
    outputs = model(sample_cuda["imgs"], sample_cuda["proj_matrices"], sample_cuda["depth_values"],test=True)



    # depth_est = outputs["depth"]
    depth_est = outputs["refined_depth"]
    # print(depth_est1.shape)
    # print(depth_est.shape)

    # photometric_confidence = outputs["photometric_confidence"]

    loss, depth_loss = model_loss(outputs, depth_gt_ms, mask_ms, dlossw=[float(e) for e in args.dlossw.split(",") if e])

    scalar_outputs = {"loss": loss, "depth_loss": depth_loss}
    image_outputs = {"depth_est": depth_est,
                     # "photometric_confidence": photometric_confidence,
                     "depth_gt": depth_gt,
                     "ref_img": sample["imgs"][:, 0],
                     "mask": mask}
    saved_outputs = {"outimage": sample["outimage"],
                     "outcam": sample["outcam"],
                     "out_view": sample["out_view"],
                     "out_name": sample["out_name"]}

    if detailed_summary:
        image_outputs["errormap"] = (depth_est - depth_gt).abs() * mask

    scalar_outputs["abs_depth_error"] = AbsDepthError_metrics(depth_est, depth_gt, mask > 0.5,
                                                              float(args.min_interval * 100.0))

    scalar_outputs["thres1interval_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, float(args.min_interval * 1.0))

    scalar_outputs["thres6interval_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, float(args.min_interval * 6.0))

    scalar_outputs["thres3interval_error"] = Inter_metrics(depth_est, depth_gt, args.min_interval, mask > 0.5, 3)
    scalar_outputs["thres10interval_error"] = Thres_metrics(depth_est, depth_gt, mask > 0.5, float(args.min_interval * 10.0))

    scalar_outputs["RMSE_metrics"] = RMSE_metrics(depth_est, depth_gt, mask > 0.5, float(args.min_interval * 100.0))
    scalar_outputs["RelativeMAE_metrics"] = RelativeMAE_metrics(depth_est, depth_gt, mask > 0.5, float(args.min_interval * 100.0))

    scalar_outputs["EdgeAccuracy_metrics"] = EdgeAccuracy_metrics(depth_est, depth_gt, mask > 0.5,   edge_threshold=1, depth_threshold = float(args.min_interval * 100.0))
    # scalar_outputs["thres10interval_error"] = Thres_metrics(depth_est, depth_gt, mask < 0.5, float(args.min_interval * 10.0))



    #return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, saved_outputs, sample["out_index"]

    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs, saved_outputs



def profile():
    warmup_iter = 5
    iter_dataloader = iter(TestImgLoader)

    @make_nograd_func
    def do_iteration():
        torch.cuda.synchronize()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        test_sample(next(iter_dataloader), detailed_summary=True)
        torch.cuda.synchronize()
        end_time = time.perf_counter()
        return end_time - start_time

    for i in range(warmup_iter):
        t = do_iteration()
        print('WarpUp Iter {}, time = {:.4f}'.format(i, t))

    with torch.autograd.profiler.profile(enabled=True, use_cuda=True) as prof:
        for i in range(5):
            t = do_iteration()
            print('Profile Iter {}, time = {:.4f}'.format(i, t))
            time.sleep(0.02)

    if prof is not None:
        # print(prof)
        trace_fn = 'chrome-trace.bin'
        prof.export_chrome_trace(trace_fn)
        print("chrome trace file is written to: ", trace_fn)


@make_nograd_func
def draw():
    model.eval()
    # from datasets.preprocess import *
    # from datasets.data_io import *

    from PIL import Image
    img = Image.open("/home/zbf/16t/e/邹bf数据/1/001_011.png")

    img_array = np.array(img)
    img = img_array.astype(np.float32)
    var = np.var(img, axis=(0, 1), keepdims=True)
    mean = np.mean(img, axis=(0, 1), keepdims=True)
    np_img = (img - mean) / (np.sqrt(var) + 0.00000001)
    np_img = torch.from_numpy(np_img)
    np_img = np_img.permute(2, 0, 1).unsqueeze(0).to("cuda")
    outputs = model.draw(np_img)
    return outputs


if __name__ == '__main__':
    if args.mode == "train":
        train()
    elif args.mode == "test":
        test()
    elif args.mode == "profile":
        profile()
    elif args.mode == "draw":
        draw()