from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from datasets.data_io import *
from datasets.preprocess import *
from imageio import imread, imsave, imwrite


# the WHU dataset preprocessed by Jin Liu (only for training and testing)
class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, normalize, args, **kwargs):
        super(MVSDataset, self).__init__()
        self.data_folder = data_folder
        self.mode = mode
        self.args = args
        self.view_num = view_num
        self.normalize = normalize
        # self.ndepths = ndepths
        self.interval_scale = args.interval_scale
        self.counter = 0
        assert self.mode in ["train", "val", "test"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        # sample_list = gen_train_mvs_list(self.data_folder, self.view_num, self.args.fext)
        sample_list = gen_test_mvs_list(self.data_folder, self.view_num, ".pfm")

        return sample_list

    def __len__(self):
        return len(self.sample_list)

    def tr_read_cam_whu(self, file, interval_scale=1):
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Twc

        # if cam ori is XrightYup
        O = np.eye(3, dtype=np.float32)
        O[1, 1] = -1
        O[2, 2] = -1
        R = extrinsics[0:3, 0:3]
        R2 = np.matmul(R, O)
        extrinsics[0:3, 0:3] = R2

        extrinsics = np.linalg.inv(extrinsics)  # Tcw
        cam[0, :, :] = extrinsics

        for i in range(0, 13):
            pera[0][i] = words[17 + i]

        f = pera[0][0]
        # x0 = pera[0][1] - pera[0][7] - (pera[0][10] * pera[0][11])  # munchen
        # y0 = pera[0][2] - pera[0][8] - (pera[0][9] * pera[0][12])
        x0 = pera[0][1] # whu
        y0 = pera[0][2]
        cam[1][0][0] = f
        cam[1][1][1] = f
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        # depth range
        cam[1][3][0] = np.float32(pera[0][3])  # start
        cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
        cam[1][3][3] = np.float32(pera[0][4])  # end

        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32
        cam[1][3][2] = acturald

        """if acturald > ndepths:
            scale = acturald / np.float32(ndepths)
            cam[1][3][1] = cam[1][3][1] * scale
            acturald = ndepths"""

        location = words[23:30]

        return cam, location

    def tr_read_cam_munchen(self, file, interval_scale=1):
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Twc

        # if cam ori is XrightYup
        O = np.eye(3, dtype=np.float32)
        O[1, 1] = -1
        O[2, 2] = -1
        R = extrinsics[0:3, 0:3]
        R2 = np.matmul(R, O)
        extrinsics[0:3, 0:3] = R2

        extrinsics = np.linalg.inv(extrinsics)  # Tcw
        cam[0, :, :] = extrinsics

        for i in range(0, 13):
            pera[0][i] = words[17 + i]

        f = pera[0][0]
        x0 = pera[0][1] - pera[0][8] - (pera[0][10] * pera[0][11]) / 2  # Munchen
        y0 = pera[0][2] - (pera[0][9] * pera[0][12])
        # x0 = pera[0][1]  # whu
        # y0 = pera[0][2]
        cam[1][0][0] = f
        cam[1][1][1] = f
        cam[1][0][2] = x0
        cam[1][1][2] = y0
        cam[1][2][2] = 1

        # depth range
        # cam[1][3][0] = np.float32(pera[0][6]) - 1  # start
        # cam[1][3][1] = np.float32(pera[0][4] * interval_scale)  # interval
        # cam[1][3][3] = np.float32(pera[0][7])  # end

        cam[1][3][0] = np.float32(pera[0][3])  # start
        cam[1][3][1] = np.float32(pera[0][5] * interval_scale)  # interval
        cam[1][3][3] = np.float32(pera[0][4])  # end
        # print(pera)

        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32
        cam[1][3][2] = acturald

        """if acturald > ndepths:
            scale = acturald / np.float32(ndepths)
            cam[1][3][1] = cam[1][3][1] * scale
            acturald = ndepths
        # cam[1][3][2] = acturald
        cam[1][3][2] = ndepths"""

        location = words[23:30]

        return cam, location


    def read_img(self, filename):
        img = Image.open(filename)
        return img

    def read_depth(self, filename):
        # read pfm depth file
        # depimg = imread(filename)
        # depth_image = (np.float32(depimg) / 64.0)  # WHU MVS dataset
        depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)
        return np.array(depth_image)

    def center_image(self, img, mode='mean'):
        """ normalize image input """
        # attention: CasMVSNet [mean var];; CasREDNet [0-255]
        if mode == 'standard':
            np_img = np.array(img, dtype=np.float32) / 255.

        elif mode == 'mean':
            img_array = np.array(img)
            img = img_array.astype(np.float32)
            # img = img.astype(np.float32)
            var = np.var(img, axis=(0, 1), keepdims=True)
            mean = np.mean(img, axis=(0, 1), keepdims=True)
            np_img = (img - mean) / (np.sqrt(var) + 0.00000001)

        else:
            raise Exception("{}? Not implemented yet!".format(mode))

        return np_img


    def __getitem__(self, idx):
        data = self.sample_list[idx]
        ###### read input data ######
        outimage = None
        outcam = None
        outlocation = None

        centered_images = []
        proj_matrices = []

        #depth
        depth_image = self.read_depth(os.path.join(data[2 * self.view_num]))

        for view in range(self.view_num):
            # Images
            if self.mode == "train" and self.args.supervised:
                image = image_augment(self.read_img(data[2 * view]))
            else:
                image = self.read_img(data[2 * view])
            image = np.array(image)

            # Cameras
            cam, location = self.tr_read_cam_munchen(data[2 * view + 1], self.interval_scale)
            location.append(str(self.args.resize_scale))

            if view == 0:
                # determine a proper scale to resize input
                scaled_image, scaled_cam, scaled_depth = scale_input(image, cam, depth_image=depth_image, scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_cam, croped_depth = crop_input(scaled_image, scaled_cam, depth_image=scaled_depth, max_h=self.args.max_h, max_w=self.args.max_w, resize_scale=self.args.resize_scale)
                outimage = croped_image
                outcam = croped_cam
                outlocation = location
                depth_min = croped_cam[1][3][0]

                depth_max = croped_cam[1][3][3]
                # print(croped_cam[1][3][0],croped_cam[1][3][1],croped_cam[1][3][2],croped_cam[1][3][3])
                # depth_interval = croped_cam[1][3][1]
                # new_ndepths = croped_cam[1][3][2]
            else:
                # determine a proper scale to resize input
                scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
                # crop to fit network
                croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h, max_w=self.args.max_w, resize_scale=self.args.resize_scale)

            # scale cameras for building cost volume
            scaled_cam = scale_camera(croped_cam, scale=self.args.sample_scale)
            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = scaled_cam[0, :, :]
            intrinsics = scaled_cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(self.center_image(croped_image, mode=self.normalize))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # Depth
        # print(new_ndepths)
        depth_values = np.array([depth_min, depth_max], dtype=np.float32)
        # print(depth_values)
        mask = np.float32((croped_depth >= depth_min) * 1.0) * np.float32((croped_depth <= depth_max) * 1.0)

        h, w = croped_depth.shape
        depth_ms = {
            "stage1": cv2.resize(croped_depth, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(croped_depth, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": croped_depth,
        }
        mask_ms = {
            "stage1": cv2.resize(mask, (w // 4, h // 4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(mask, (w // 2, h // 2), interpolation=cv2.INTER_NEAREST),
            "stage3": mask,
        }
        # ms proj_mats
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, :2, :] = proj_matrices[:, :2, :] / 4

        proj_matrices_ms = {
            "stage1": stage3_pjmats,
            "stage2": stage2_pjmats,
            "stage3": proj_matrices
        }

        name = os.path.splitext(os.path.basename(data[0]))[0]
        vid = os.path.dirname(data[0]).split("/")[-2]

        return {"imgs": centered_images,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "outimage": outimage,
                "outcam": outcam,
                "outlocation": outlocation,
                "out_name": name,
                "out_view": vid}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    pass
