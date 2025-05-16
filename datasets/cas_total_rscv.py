from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image, ImageEnhance, ImageOps, ImageFile
from datasets.data_io import *
from datasets.preprocess import *
from imageio import imread, imsave, imwrite


# all of existed dataset preprocessed by Jin Liu (only for training and testing)
class MVSDataset(Dataset):
    def __init__(self, data_folder, mode, view_num, normalize, args, **kwargs):
        super(MVSDataset, self).__init__()
        self.all_data_folder = data_folder
        self.mode = mode
        self.args = args
        self.view_num = view_num
        self.normalize = normalize
        self.ndepths = args.ndepths
        self.interval_scale = args.interval_scale
        self.counter = 0
        assert self.mode in ["train", "val", "test"]
        self.sample_list = self.build_list()
        self.sample_num = len(self.sample_list)

    def build_list(self):
        # Prepare all training samples
        """ generate data paths for whu dataset """

        data_folder = self.all_data_folder
        sample_list = self.whu_list(data_folder, "munchen_100RS", gt_fext='.pfm')
        # total_sample_list += sample_list

        return sample_list

        # total_sample_list = []
        #
        # # meitan/tianjin/munchen sets
        # all_data_set_path = self.all_data_folder + '/index.txt'
        # all_data_set = open(all_data_set_path).read().split()
        #
        # for sub_set in all_data_set:
        #     data_folder = self.all_data_folder + '/{}/{}'.format(sub_set, self.mode)

            # if sub_set == 'munchen_100RS':
            #     sample_list = self.whu_list(data_folder, sub_set, gt_fext='.pfm')
            # elif sub_set == 'dtu':
            #     sample_list = self.dtu_list(data_folder, sub_set, gt_fext='.pfm')
            # elif sub_set == 'BlendedMVS':
            #     sample_list = self.BlendedMVS_list(data_folder, sub_set, gt_fext='.pfm')
            # elif sub_set == 'meitan_oblique':
            #     sample_list = self.ObliqueWhu_list(data_folder, sub_set, gt_fext='.exr')
            # else:
            #     sample_list = self.whu_list(data_folder, sub_set, gt_fext='.png')

        #     total_sample_list += sample_list
        #
        # return total_sample_list


    def whu_list(self, data_folder, sat_name, gt_fext='.png'):
        sample_list = []

        # image index
        train_cluster_path = data_folder + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        # pair
        view_pair_path = data_folder + '/pair.txt'
        ref_indexs = []
        src_indexs = []
        with open(view_pair_path) as f:
            cluster_num = int(f.readline().rstrip())
            for idx in range(cluster_num):
                ref_index = int(f.readline().rstrip())
                src_index = [int(x) for x in f.readline().rstrip().split()][1:]
                ref_indexs.append(ref_index)
                src_indexs.append(src_index)

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(data_folder, ('Images/%s' % i)).replace("\\", "/")
            cam_folder = os.path.join(data_folder, ('Cams/%s' % i)).replace("\\", "/")
            depth_folder = os.path.join(data_folder, ('Depths/%s' % i)).replace("\\", "/")

            # for each view
            for ref_ind, view_inds in zip(ref_indexs, src_indexs):  # 0-4
                image_folder2 = os.path.join(image_folder, ('%d' % ref_ind)).replace("\\", "/")
                image_files = sorted(os.listdir(image_folder2))

                view_cnts = min(self.view_num, len(view_inds) + 1)

                for j in range(0, int(np.size(image_files))):
                    paths = []
                    portion = os.path.splitext(image_files[j])
                    newcamname = portion[0] + '.txt'
                    newdepthname = portion[0] + gt_fext

                    # ref image
                    ref_image_path = os.path.join(os.path.join(image_folder, ('%d' % ref_ind)),
                                                  image_files[j]).replace("\\", "/")
                    ref_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % ref_ind)), newcamname).replace(
                        "\\", "/")

                    paths.append(ref_image_path)
                    paths.append(ref_cam_path)

                    # view images
                    for view in range(view_cnts - 1):
                        # print(image_folder)
                        view_ind = view_inds[view]  # selected view image
                        view_image_path = os.path.join(os.path.join(image_folder, ('%d' % view_ind)),
                                                       image_files[j]).replace("\\", "/")

                        view_cam_path = os.path.join(os.path.join(cam_folder, ('%d' % view_ind)),
                                                     newcamname).replace("\\", "/")
                        paths.append(view_image_path)
                        paths.append(view_cam_path)

                    # depth path
                    depth_image_path = os.path.join(os.path.join(depth_folder, ('%d' % ref_ind)),
                                                    newdepthname).replace("\\", "/")
                    paths.append(depth_image_path)
                    sample_list.append((sat_name, view_cnts, paths))

        return sample_list

    def dtu_list(self, data_folder, set_name, gt_fext='.pfm'):
        sample_list = []

        # image index
        train_cluster_path = data_folder + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        # pair
        view_pair_path = data_folder + '/pair.txt'
        metas = []

        with open(view_pair_path) as f:
            cluster_num = int(f.readline().rstrip())
            for idx in range(cluster_num):
                ref_index = int(f.readline().rstrip())
                src_indexs = [int(x) for x in f.readline().rstrip().split()[1::2]]
                for light_idx in range(7):
                    metas.append((light_idx, ref_index, src_indexs))

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(data_folder, ('Rectified/%s_train' % i)).replace("\\", "/")
            cam_folder = os.path.join(data_folder, 'Cameras').replace("\\", "/")
            depth_folder = os.path.join(data_folder, ('Depths/%s' % i)).replace("\\", "/")

            for idx in range(len(metas)):
                paths = []
                light_idx, ref_index, src_indexs = metas[idx]
                view_cnts = min(self.view_num, len(src_indexs) + 1)
                src_view_ids = src_indexs[:view_cnts - 1]

                ref_image_path = os.path.join(image_folder,
                                              'rect_{:0>3}_{}_r5000.png'.format(ref_index + 1, light_idx)).replace("\\",
                                                                                                                   "/")
                ref_cam_path = os.path.join(cam_folder, '{:0>8}_cam.txt'.format(ref_index)).replace("\\", "/")
                depth_image_path = os.path.join(depth_folder, 'depth_map_{:0>4}.pfm'.format(ref_index)).replace("\\",
                                                                                                                "/")

                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                for i, vid in enumerate(src_view_ids):
                    view_image_path = os.path.join(image_folder,
                                                   'rect_{:0>3}_{}_r5000.png'.format(vid + 1, light_idx)).replace("\\",
                                                                                                                  "/")
                    view_cam_path = os.path.join(cam_folder, '{:0>8}_cam.txt'.format(vid)).replace("\\", "/")
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                paths.append(depth_image_path)
                sample_list.append((set_name, view_cnts, paths))

        return sample_list

    def BlendedMVS_list(self, data_folder, set_name, gt_fext='.pfm'):
        sample_list = []

        # image index
        train_cluster_path = data_folder + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(data_folder, ('%s/blended_images' % i)).replace("\\", "/")
            cam_folder = os.path.join(data_folder, '%s/cams' % i).replace("\\", "/")
            depth_folder = os.path.join(data_folder, ('%s/rendered_depth_maps' % i)).replace("\\", "/")
            view_pair_path = os.path.join(data_folder, '%s/cams/pair.txt' % i).replace("\\", "/")

            metas = []
            with open(view_pair_path) as f:
                cluster_num = int(f.readline().rstrip())
                for idx in range(cluster_num):
                    ref_index = int(f.readline().rstrip())
                    src_indexs = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((ref_index, src_indexs))

            for idx in range(len(metas)):
                paths = []
                ref_index, src_indexs = metas[idx]
                view_cnts = min(self.view_num, len(src_indexs) + 1)
                src_view_ids = src_indexs[:view_cnts - 1]

                ref_image_path = os.path.join(image_folder, '{:0>8}.jpg'.format(ref_index)).replace("\\", "/")
                ref_cam_path = os.path.join(cam_folder, '{:0>8}_cam.txt'.format(ref_index)).replace("\\", "/")
                depth_image_path = os.path.join(depth_folder, '{:0>8}.pfm'.format(ref_index)).replace("\\", "/")
                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                for i, vid in enumerate(src_view_ids):
                    view_image_path = os.path.join(image_folder, '{:0>8}.jpg'.format(vid)).replace("\\", "/")
                    view_cam_path = os.path.join(cam_folder, '{:0>8}_cam.txt'.format(vid)).replace("\\", "/")
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                paths.append(depth_image_path)
                sample_list.append((set_name, view_cnts, paths))

        return sample_list

    def ObliqueWhu_list(self, data_folder, set_name, gt_fext='.exr'):
        sample_list = []

        # image index
        train_cluster_path = data_folder + '/index.txt'
        data_cluster = open(train_cluster_path).read().split()

        def read_name_list(path):
            names_list = {}
            cluster_list = open(path).read().split()
            total_num = int(cluster_list[0])

            for i in range(total_num):
                index = int(cluster_list[i * 3 + 1])  # index
                name = cluster_list[i * 3 + 2]
                names_list[index] = name
            return names_list

        # for each data scene
        for i in data_cluster:
            image_folder = os.path.join(data_folder, ('%s/images' % i)).replace("\\", "/")
            cam_folder = os.path.join(data_folder, '%s/cams' % i).replace("\\", "/")
            depth_folder = os.path.join(data_folder, ('%s/depths' % i)).replace("\\", "/")
            view_pair_path = os.path.join(data_folder, '%s/info/viewpair.txt' % i).replace("\\", "/")
            map_index_path = os.path.join(data_folder, '%s/info/image_path.txt' % i).replace("\\", "/")
            map_index_dict = read_name_list(map_index_path)

            metas = []
            with open(view_pair_path) as f:
                cluster_num = int(f.readline().rstrip())
                for idx in range(cluster_num):
                    ref_index = int(f.readline().rstrip())
                    src_indexs = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    metas.append((ref_index, src_indexs))

            for idx in range(len(metas)):
                paths = []
                ref_index, src_indexs = metas[idx]
                view_cnts = min(self.view_num, len(src_indexs) + 1)
                src_view_ids = src_indexs[:view_cnts - 1]

                ref_name = map_index_dict[ref_index]
                ref_image_path = os.path.join(image_folder, '{}.png'.format(ref_name)).replace("\\", "/")
                ref_cam_path = os.path.join(cam_folder, '{}.txt'.format(ref_name)).replace("\\", "/")
                depth_image_path = os.path.join(depth_folder, ref_name+gt_fext).replace("\\", "/")
                paths.append(ref_image_path)
                paths.append(ref_cam_path)

                for i, vid in enumerate(src_view_ids):
                    src_name = map_index_dict[vid]
                    view_image_path = os.path.join(image_folder, '{}.png'.format(src_name)).replace("\\", "/")
                    view_cam_path = os.path.join(cam_folder, '{}.txt'.format(src_name)).replace("\\", "/")
                    paths.append(view_image_path)
                    paths.append(view_cam_path)

                paths.append(depth_image_path)
                sample_list.append((set_name, view_cnts, paths))

        return sample_list


    def __len__(self):
        return len(self.sample_list)


    def CalRotationMatrix(self, phi, omega, kappa):
        #  CalRotationMatrix with phi,omega,kappa, in rad
        # if given angle, omega_inrad = omega_inangle*2*math.pi/360
        tem = []

        m00 = math.cos(phi) * math.cos(kappa)
        tem.append(m00)
        m01 = math.cos(omega) * math.sin(kappa) + math.sin(omega) * math.sin(phi) * math.cos(kappa)
        tem.append(m01)
        m02 = math.sin(omega) * math.sin(kappa) - math.cos(omega) * math.sin(phi) * math.cos(kappa)
        tem.append(m02)

        m10 = -1*math.cos(phi) * math.sin(kappa)
        tem.append(m10)
        m11 = math.cos(omega) * math.cos(kappa) - math.sin(omega) * math.sin(phi) * math.sin(kappa)
        tem.append(m11)
        m12 = math.sin(omega) * math.cos(kappa) + math.cos(omega) * math.sin(phi) * math.sin(kappa)
        tem.append(m12)

        m20 = math.sin(phi)
        tem.append(m20)
        m21 = -1 * math.sin(omega) * math.cos(phi)
        tem.append(m21)
        m22 = math.cos(omega) * math.cos(phi)
        tem.append(m22)
        rotation_matrix = np.array(tem)


        return rotation_matrix.reshape(3,3)


    def cam_noise(self, extrinsics):

        R = extrinsics[0:3, 0:3]
        rand_omega = (-1 + 2 * np.random.random()) / 1.0 * 2 * math.pi / 360.0
        rand_phi = (-1 + 2 * np.random.random()) / 1.0 * 2 * math.pi / 360.0
        rand_kappa = (-1 + 2 * np.random.random()) / 1.0 * 2 * math.pi / 360.0
        rand_R = self.CalRotationMatrix(rand_phi, rand_omega, rand_kappa)
        new_R = R * rand_R
        extrinsics[0:3, 0:3] = new_R

        return extrinsics


    def tr_read_whu_cam(self, file, interval_scale=1):
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

        # extrinsics = self.cam_noise(extrinsics)

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
        x0 = pera[0][1] # whu
        y0 = pera[0][2]

        # K Photogrammetry system XrightYup
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

        location = words[23:30]
        return cam, location

    def tr_read_dtu_cam(self, file, interval_scale=1):
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        intrinsics = np.zeros((3, 3), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Tcw

        cam[0, :, :] = extrinsics

        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                intrinsics[i][j] = words[intrinsic_index]

        cam[1, 0:3, 0:3] = intrinsics

        # depth range
        cam[1][3][0] = np.float32(words[27])  # start
        cam[1][3][1] = np.float32(words[28] * interval_scale)  # interval
        cam[1][3][3] = np.float32(cam[1][3][0] + cam[1][3][1] * 192)  # end

        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32

        # if acturald > self.ndepths:
        #     scale = acturald / np.float32(self.ndepths)
        #     cam[1][3][1] = cam[1][3][1] * scale
        #     acturald = self.ndepths

        cam[1][3][2] = acturald

        location = [0, 0, 0, 0]
        return cam, location

    def tr_read_blendedmvs_cam(self, file, interval_scale=1):
        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        intrinsics = np.zeros((3, 3), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        pera = np.zeros((1, 13), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 1
                extrinsics[i][j] = words[extrinsic_index]  # Tcw

        cam[0, :, :] = extrinsics

        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                intrinsics[i][j] = words[intrinsic_index]

        cam[1, 0:3, 0:3] = intrinsics

        # depth range
        cam[1][3][0] = np.float32(words[27])  # start
        cam[1][3][1] = np.float32(words[28]) * np.float32(interval_scale)  # interval
        cam[1][3][2] = np.float32(words[29]) / np.float32(interval_scale)  # sample number
        cam[1][3][3] = np.float32(words[30])  # end

        location = [0, 0, 0, 0]
        return cam, location

    def tr_read_obliquewhu_cam(self, file, interval_scale=1):

        # read camera txt file
        cam = np.zeros((2, 4, 4), dtype=np.float32)
        intrinsics = np.zeros((3, 3), dtype=np.float32)
        extrinsics = np.zeros((4, 4), dtype=np.float32)
        words = open(file).read().split()
        # read extrinsic
        for i in range(0, 4):
            for j in range(0, 4):
                extrinsic_index = 4 * i + j + 2
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

        for i in range(0, 3):
            for j in range(0, 3):
                intrinsic_index = 3 * i + j + 18
                intrinsics[i][j] = words[intrinsic_index]
        cam[1, 0:3, 0:3] = intrinsics

        # depth range
        cam[1][3][0] = np.float32(words[27]) # start
        cam[1][3][3] = np.float32(words[28])# end
        cam[1][3][1] = np.float32(words[29]) * interval_scale # interval
        # cam[1][3][1] = np.float32(0.8) * interval_scale  # interval
        # cam[1][3][1] = np.float32(0.2)  # interval
        acturald = int((cam[1][3][3] - cam[1][3][0]) / cam[1][3][1] / 32 + 1) * 32
        cam[1][3][2] = acturald


        location = [0, 0, 0, 0]
        return cam, location

    def read_img(self, filename):
        img = Image.open(filename)
        return img

    def read_depth(self, filename, set_name):

        fext = os.path.splitext(filename)[-1]
        if fext =='.png':
            depimg = imread(filename)# read png depth file
            depth_image = (np.float32(depimg) / 64.0)  # WHU MVS dataset and Tianjin MVS dataset
        elif fext == '.pfm':
            depth_image = np.array(read_pfm(filename)[0], dtype=np.float32)  # munchen dataset   # read pfm depth file
        elif fext == '.exr':
            depth_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
            mask_path = filename.replace("depths", "masks")
            mask_path = mask_path.replace(".exr", ".png")
            mask_image = np.array(cv2.imread(mask_path, cv2.COLOR_BGR2GRAY)) / 255.
            mask_image = mask_image < 0.5
            depth_image[mask_image] = 0
        else:
            raise Exception("{}? Not implemented yet!".format(fext))

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

        sat_name, view_cnts, paths = self.sample_list[idx]
        ###### read input data ######
        outimage = None
        outcam = None
        outlocation = None

        centered_images = []
        proj_matrices = []

        #depth
        depth_image = self.read_depth(os.path.join(paths[2 * view_cnts]), sat_name)
        # print(paths[2 * view_cnts])

        for view in range(view_cnts):
            # Images
            if self.mode == "train" and self.args.supervised:
                image = image_augment(self.read_img(paths[2 * view]))
            else:
                image = self.read_img(paths[2 * view])
            image = np.array(image)

            # Cameras
            if sat_name == 'dtu':
                cam, location = self.tr_read_dtu_cam(paths[2 * view + 1], self.interval_scale)
            elif sat_name == 'BlendedMVS':
                cam, location = self.tr_read_blendedmvs_cam(paths[2 * view + 1], self.interval_scale)
            elif sat_name == 'meitan_oblique':
                cam, location = self.tr_read_obliquewhu_cam(paths[2 * view + 1], self.interval_scale)
            else:
                cam, location = self.tr_read_whu_cam(paths[2 * view + 1], self.interval_scale)
                location.append(str(self.args.resize_scale))

            if view == 0:
                # determine a proper scale to resize input
                # scaled_image, scaled_cam, depth_image = scale_input(image, cam, depth_image=depth_image,
                #                                                      scale=self.args.resize_scale)
                # # crop to fit network
                # croped_image, croped_cam, depth_image = crop_input(scaled_image, scaled_cam, depth_image=depth_image,
                #                                                     max_h=self.args.max_h, max_w=self.args.max_w,
                #                                                     resize_scale=self.args.resize_scale)
                outimage = image
                outcam = cam
                outlocation = location
                depth_min = outcam[1][3][0]
                depth_interval = outcam[1][3][1]
                new_ndepths = outcam[1][3][2]
                depth_max = outcam[1][3][3]
            # else:
            #     # determine a proper scale to resize input
            #     scaled_image, scaled_cam = scale_input(image, cam, scale=self.args.resize_scale)
            #     # crop to fit network
            #     croped_image, croped_cam = crop_input(scaled_image, scaled_cam, max_h=self.args.max_h, max_w=self.args.max_w, resize_scale=self.args.resize_scale)


            # multiply intrinsics and extrinsics to get projection matrix
            extrinsics = cam[0, :, :]
            intrinsics = cam[1, 0:3, 0:3]
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])

            proj_matrices.append(proj_mat)
            centered_images.append(self.center_image(image, mode=self.normalize))

        centered_images = np.stack(centered_images).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        # Depth
        depth_values = np.array([depth_min, depth_max, depth_interval], dtype=np.float32)
        # depth_values = np.arange(np.float(depth_min), np.float(depth_max), np.float(depth_interval), dtype=np.float32)
        # depth_values = np.arange(np.float(depth_min), np.float(depth_interval * (new_ndepths - 0.5) + depth_min), np.float(depth_interval), dtype=np.float32)

        mask = np.float32((depth_image >= depth_min) * 1.0) * np.float32((depth_image <= depth_max) * 1.0)

        h, w = depth_image.shape
        depth_ms = {
            "stage1": cv2.resize(depth_image, (w//4, h//4), interpolation=cv2.INTER_NEAREST),
            "stage2": cv2.resize(depth_image, (w//2, h//2), interpolation=cv2.INTER_NEAREST),
            "stage3": depth_image,
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

        if sat_name == 'meitan_oblique':
            name = os.path.splitext(os.path.basename(paths[0]))[0]
            vid = os.path.dirname(paths[0]).split("/")[-1]
        else:
            name = os.path.splitext(os.path.basename(paths[0]))[0]
            vid = os.path.dirname(paths[0]).split("/")[-2]


        return {"imgs": centered_images,
                "proj_matrices": proj_matrices_ms,
                "depth": depth_ms,
                "mask": mask_ms,
                "depth_values": depth_values,
                "depth_interval": depth_interval,
                "outimage": outimage,
                "outcam": outcam,
                "outlocation": outlocation,
                "out_name": name,
                "out_view": vid}


if __name__ == "__main__":
    # some testing code, just IGNORE it
    pass
