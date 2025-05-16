import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose

from .dinov2 import DINOv2
from .util.blocks import FeatureFusionBlock, _make_scratch
from .util.transform import Resize, NormalizeImage, PrepareForNet


def _make_fusion_block(features, use_bn, size=None):
    return FeatureFusionBlock(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        size=size,
    )


class ConvBlock(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_feature),
            nn.ReLU(True)
        )
    
    def forward(self, x):
        return self.conv_block(x)


class DPTHead(nn.Module):
    def __init__(
        self, 
        in_channels, 


        use_bn=False,
        feat=64,

        out=[64, 128, 256, 256],

            use_clstoken=False
    ):
        super(DPTHead, self).__init__()
        
        self.use_clstoken = use_clstoken
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out[0],
                out_channels=out[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out[1],
                out_channels=out[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out[3],
                out_channels=out[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        if use_clstoken:
            self.readout_projects = nn.ModuleList()
            for _ in range(len(self.projects)):
                self.readout_projects.append(
                    nn.Sequential(
                        nn.Linear(2 * in_channels, in_channels),
                        nn.GELU()))
        
        self.scratch = _make_scratch(
            out,
            feat,
            groups=1,
            expand=False,
        )


        self.scratch.stem_transpose = None
        
        self.scratch.refinenet1 = _make_fusion_block(feat, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(feat, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(feat, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(feat, use_bn)
        
        head_features_1 = feat
        head_features_2 = 32
        
        # self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        # self.scratch.output_conv2 = nn.Sequential(
        #     nn.Conv2d(head_features_1 // 2, head_features_2, kernel_size=3, stride=1, padding=1),
        #     nn.ReLU(True),
        #     nn.Conv2d(head_features_2, 1, kernel_size=1, stride=1, padding=0),
        #     nn.ReLU(True),
        #     nn.Identity(),
        # )
    
    def forward(self, out_features, patch_h, patch_w):

        out = []
        for i, x in enumerate(out_features):
            if self.use_clstoken:
                x, cls_token = x[0], x[1]
                readout = cls_token.unsqueeze(1).expand_as(x)
                x = self.readout_projects[i](torch.cat((x, readout), -1))
            else:
                x = x[0]
            
            x = x.permute(0, 2, 1).reshape((x.shape[0], x.shape[-1], patch_h, patch_w))
            
            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            
            out.append(x)
        # layer_3 = {Tensor: (1, 384, 37, 37)}
        # layer_4 = {Tensor: (1, 768, 19, 19)}
        # layer_2 = {Tensor: (1, 192, 74, 74)}
        # layer_1 = {Tensor: (1, 96, 148, 148)}
        layer_1, layer_2, layer_3, layer_4 = out
        
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        # {Tensor: (1, 128, 148, 148)}
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        # {Tensor: (1, 128, 74, 74)}
        layer_3_rn = self.scratch.layer3_rn(layer_3)
     # {Tensor: (1, 128, 37, 37)}
        layer_4_rn = self.scratch.layer4_rn(layer_4)
        # {Tensor: (1, 128, 19, 19)}


        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])

        # {Tensor: (1, 128, 37, 37)}
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        # {Tensor: (1, 128, 74, 74)}
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        # {Tensor: (1, 128, 148, 148)}
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)
        # path_1{Tensor: (1, 128, 296, 296)}

        return path_3, path_2, path_1


class DepthAnythingV2(nn.Module):
    def __init__(
        self, 
        encoder='vitl', 
        # features=256,
        # out_channels=[256, 512, 1024, 1024],
        features=1,
        use_bn=False,
        out_channels=[96, 192, 384, 768],
        use_clstoken=False
    ):
        super(DepthAnythingV2, self).__init__()
        
        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        
        self.encoder = encoder
        self.pretrained = DINOv2(model_name=encoder)
        
        # self.depth_head = DPTHead(self.pretrained.embed_dim, features, use_bn, out_channels=out_channels, use_clstoken=use_clstoken)
        self.depth_head = DPTHead(self.pretrained.embed_dim, use_bn, out=out_channels, use_clstoken=use_clstoken)

        # for name, param in self.depth_head.named_parameters():
        #     param.requires_grad = True
        #     print(f"{name} requires_grad: {param.requires_grad}")

    def forward(self, x):

        #x 1,3,518,518
        patch_h, patch_w = x.shape[-2] // 14, x.shape[-1] // 14  #37
        with torch.no_grad():
            features = self.pretrained.get_intermediate_layers(x, self.intermediate_layer_idx[self.encoder], return_class_token=True)

        with torch.no_grad():
            path_3, path_2, path_1 = self.depth_head(features, patch_h, patch_w)

        # path_3, path_2, path_1 = self.depth_head(features, patch_h, patch_w)

        return path_3, path_2, path_1

    
    # @torch.no_grad()
    def infer_image(self, raw_image, input_size=518):

        image, (h, w) = self.image2tensor(raw_image, input_size)


        # print(raw_image.shape)# (1, 8, 384,768)
        # depth = self.forward(image)
        # depth = F.interpolate(depth[:, None], (h, w), mode="bilinear", align_corners=True)#[0, 0]

        # image = F.interpolate(raw_image, size=(input_size, input_size), mode='bilinear', align_corners=False)


        path_3, path_2, path_1 = self.forward(image)
        path_1 = F.interpolate(path_1, (h, w), mode="nearest")#[0, 0]
        path_2 = F.interpolate(path_2, (int(h/2), int(w/2)), mode="nearest")#[0, 0]
        path_3 = F.interpolate(path_3, (int(h/4), int(w/4)), mode="nearest")#[0, 0]




        #         intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0) # (1, 32, 384,768)

        # depth1 = F.interpolate(path_1, (h, w), mode="bilinear", align_corners=True)#[0, 0]
        # depth2 = F.interpolate(path_2, (int(h/2), int(w/2)), mode="bilinear", align_corners=True)#[0, 0]
        # depth3 = F.interpolate(path_3, (int(h/4), int(w/4)), mode="bilinear", align_corners=True)#[0, 0]




        return path_3, path_2, path_1#.cpu().numpy()

        # depth = F.interpolate(depth, (h, w), mode="bilinear", align_corners=True)#[0, 0]
        # print(depth.shape)

        # return depth#.cpu().numpy()
    
    def image2tensor(self, raw_image, input_size=518):        
        # transform = Compose([
        #     # Resize(
        #     #     width=input_size,
        #     #     height=input_size,
        #     #     resize_target=False,
        #     #     keep_aspect_ratio=True,
        #     #     ensure_multiple_of=14,
        #     #     resize_method='lower_bound',
        #     #     image_interpolation_method=cv2.INTER_CUBIC,
        #     # ),
        #     NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #     # PrepareForNet(),
        # ])
        
        h, w = raw_image.shape[2:]
        
        # image = cv2.cvtColor(raw_image, cv2.COLOR_BGR2RGB) / 255.0
        # # print(image.shape)
        # raw_image = transform({'image': raw_image})['image']
        # print(image.shape)

        resized_image = F.interpolate(raw_image, size=(input_size, input_size), mode='bilinear', align_corners=False)

        # image = torch.from_numpy(image).unsqueeze(0)
        # print(image.shape)

        # DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        # image = image.to(DEVICE)
        
        return resized_image, (h, w)
