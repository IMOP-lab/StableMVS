import os
import subprocess

# 设置环境变量
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'

# 定义日志目录
LOG_DIR = "wt5.7"
#df是直接预测原图，单步，训练时df冻结,其他参数微调，只有sourcs image视角进行df
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# 定义训练路径和模型检查点路径
MVS_TRAINING= "/opt/data/private/WHU-OMVS/train"
# MVS_TRAINING = "/opt/data/private/WHU-OMVS/train"
# LOADCKPT = "/home/zbf/16t/e/邹bf数据/1/model_000014_0.1400.ckpt"
# LOADCKPT = "/home/zbf/Desktop/remote/3d_guaoss/casREDNet_pytorch-master/wo df/model_000006_0.1474.ckpt"

LOADCKPT = '/opt/data/private/2025code/MVS_ZBFXR/dfload3fredpt/1final_load_df1new/model_000031_0.1289.ckpt'
# LOADCKPT = '/opt/data/private/2025code/MVS_lowLT/MVS_lowLT/only_traindf/model_000000.ckpt'

# 执行训练命令
# LR = float(0.00001)
command = [
    "python", "train_whu.py",
    "--logdir", LOG_DIR,
    "--model", "dino",
    "--batch_size", "1",
    "--trainpath", MVS_TRAINING,
    "--mode", "train",
    "--dataset", "cas_whuomvs",
    # "--resume"
    # "--share_cr"
    '--lr', "0.00001",
    "--loadckpt", LOADCKPT
]

subprocess.run(command)
