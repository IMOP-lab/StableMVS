import os
import subprocess

# 设置环境变量
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# 定义日志目录
# LOG_DIR = "only_traindf"
# if not os.path.exists(LOG_DIR):
#     os.makedirs(LOG_DIR)

# 定义训练路径和模型检查点路径
MVS_TRAINING= "/opt/data/private/WHU-OMVS/train"
# MVS_TRAINING = "/opt/data/private/WHU-OMVS/train"
# LOADCKPT = "/home/zbf/16t/e/邹bf数据/1/model_000014_0.1400.ckpt"
# LOADCKPT = "/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/wo df/model_000006_0.1474.ckpt"

LOADCKPT = '/opt/data/private/2025code/MVS_lowLT/MVS_lowLT/only_traindf/model_000000.ckpt'
LOADCKPT = '/opt/data/private/2025code/MVS_lowLT/1_5.7wt_/wt5.7/model_000016_0.1338.ckpt'

# 执行训练命令
# LR = float(0.00001)
command = [
    "python", "train_whu.py",
    # "--logdir", LOG_DIR,
    "--model", "dino",
    "--batch_size", "1",
    "--trainpath", MVS_TRAINING,
    "--mode", "test",
    "--dataset", "cas_whuomvs",
    # "--resume"
    # "--share_cr"
    # '--lr', "0.001",
    "--loadckpt", LOADCKPT
]

subprocess.run(command)
