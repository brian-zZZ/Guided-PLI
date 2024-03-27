# SEED = 42
# 基本参数
n_gpu = 1
gpu_start = 0
gradient_accumulation = 1 # 8 # mark
lr = 5e-4 # 1e-4 # mark
weight_decay = 1e-5  # 1e-4 # mark
decay_interval = 5
lr_decay = 0.995 # 0.995 # 1 # mark
# 训练模型时
do_train = True
do_test = True
do_save_emb = False # True
do_save_pretrained_emb = False # True # False
return_emb = do_save_emb | do_save_pretrained_emb
# 获得表征时 端到端微调
# do_train = False
# do_test = False
# do_save_emb = True
# do_save_pretrained_emb = False
# return_emb = do_save_emb | do_save_pretrained_emb
# 获得表征是 random
# do_train = False
# do_test = False
# do_save_emb = False
# do_save_pretrained_emb = True
# return_emb = do_save_emb | do_save_pretrained_emb

# PLI任务
sampled_frac = 1 # 0.2 # 训练集采样比例, 设置为1以不采样
freeze_seq_encoder = False


# 加载数据配置参数
import yaml
from argparse import ArgumentParser
parser = ArgumentParser(description='Model configuration')
parser.add_argument('--SEED', type=int, default=42)
parser.add_argument('--task', type=str, default='Kinase', choices=['PDBBind', 'Kinase', 'DUDE', 'GPCR'])
parser.add_argument('--random', action='store_true', help='Wether random initialize model weights')
parser.add_argument('--guide', action='store_true', default=False, help='Wether guide the finetuning using transferability metric')
args = parser.parse_args()
args_dict = yaml.load(open("args.yaml", 'r', encoding='utf-8'), Loader=yaml.FullLoader)
for k, v in args_dict.items():
    setattr(args, k, v)

if args.guide:
    return_emb = True

max_seq_len = 511
batch_size = 32 # 2
# gradient_accumulation = 8

task = args.task
task_metrics = {
    'PDBBind': "R", # Personr's ρ
    'Kinase': "AUC",
    'DUDE': "AUC",
}
if task == 'PDBBind':
    epochs = 50
elif task == 'Kinase':
    epochs = 3 # 10
    lr = 1e-3
elif task == 'DUDE':
    # epochs = 10
    epochs = 50
    # lr = 1e-5
    lr = 1e-4

args.max_seq_len = max_seq_len
SEED = args.SEED
random = args.random


# 获取变量
config_variables = dict(globals(), **locals())
config_variables = {k: v for k, v in config_variables.items() if '__' not in k}
config_variables = {k: v for k, v in config_variables.items() if type(v) in [int, float, bool, str, dict, list, tuple]}
print(config_variables)
# 生成保存路径
import os
import time
# 训练模型，即非仅用于保存模型时才创建训练保存路径
current_time = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
if args.guide:
    path_model = 'outputs_guide/%s-%s/' % (task, current_time)
else:
    path_model = 'outputs/%s-%s/' % (task, current_time)
os.makedirs(path_model, exist_ok=True)
# 保存配置参数文件
import json
with open(path_model+"config.json", 'w') as f:
    f.write(json.dumps(config_variables, indent=2))
