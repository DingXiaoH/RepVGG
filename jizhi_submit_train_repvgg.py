import argparse
import datetime
import os
import json

parser = argparse.ArgumentParser('JIZHI submit', add_help=False)
parser.add_argument('arch', default=None, type=str)
parser.add_argument('tag', default=None, type=str)
parser.add_argument('--config', default='/apdcephfs_cq2/share_1290939/xiaohanding/cnt/default_V100x8_elastic_config.json', type=str,
                    help='config file')


args = parser.parse_args()
run_dir = f'{args.arch}_{args.tag}'

cmd = f'python3 -m torch.distributed.launch --nproc_per_node 8 --master_port 12349 main.py ' \
      f'--arch {args.arch} --batch-size 32 --tag {args.tag} --output-dir /apdcephfs_cq2/share_1290939/xiaohanding/swin_exps/{args.arch}_{args.tag} --opts TRAIN.EPOCHS 120 TRAIN.BASE_LR 0.1 TRAIN.WEIGHT_DECAY 4e-5 TRAIN.WARMUP_EPOCHS 5 MODEL.LABEL_SMOOTHING 0.1 AUG.PRESET raug15 DATA.DATASET imagenet'

os.system('cd /apdcephfs_cq2/share_1290939/xiaohanding/RepVGG/')
os.system(f'mkdir runs/{run_dir}')
with open(f'runs/{run_dir}/start.sh', 'w') as f:
    f.write(cmd)
with open(args.config, 'r') as f:
    json_content = json.load(f)
json_content['model_local_file_path'] = f'/apdcephfs_cq2/share_1290939/xiaohanding/RepVGG/runs/{run_dir}'
config_file_path = f'/apdcephfs_cq2/share_1290939/xiaohanding/RepVGG/runs/{run_dir}/config.json'
with open(config_file_path, 'w') as f:
    json.dump(json_content, f)

os.system(f'cp *.py runs/{run_dir}/')
os.system(f'cp -r data runs/{run_dir}/')
os.system(f'cp -r train runs/{run_dir}/')
os.system(f'cd runs/{run_dir}')
os.system(f'jizhi_client start -scfg {config_file_path}')