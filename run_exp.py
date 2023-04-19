import os
from subprocess import Popen
from copy import deepcopy

params_template = {
    'is_train':1,
    'fast_dev_run':False,
    'auto_select_gpus':True,
    'log_every_n_steps':1,
    'backbone':'resnet50',
    'global_pooling':'max',
    'gpus':4,
    'max_epochs':75,
    'batch_size':32,
    'img_size':384,
    'num_classes':4,
    'precision':16,
    'default_root_dir':'/data/home/fangzhengqing/Result/KGDM_FINAL',
    # 'init_ckpt':'version_0/checkpoints/last.ckpt'
}

CUDA_SUFFIX="CUDA_VISIBLE_DEVICES=4,5,6,7"
ALL_METHODS=['basemethod','edl','protopnet','pedl','kgdm']

TEST = True

PLOT_TSNE=False
PLOT_CDF=False

TRAIN_BASE=False
TRAIN_EDL=False
TRAIN_PROTO=False
TRAIN_KGDM=False
TRAIN_PEDL=False

if PLOT_TSNE == True:
    TEST_METHODS=['basemethod','pedl']
    for method in TEST_METHODS:
        params = deepcopy(params_template)
        params.update(prototype_dim=512)
        params.update(prototype_num=10)
        params.update(method=method)
        params.update(batch_size=64)
        params.update(gpus=1)
        cmd = f"{CUDA_SUFFIX} python plot_tsne.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
        print(cmd)
        p = Popen(cmd, shell=True)
        return_code = p.wait()
    exit()



for i in range(0, 5):
    # Basemethod
    if TRAIN_BASE:
        params = deepcopy(params_template)
        params.update(method='basemethod')
        params.update(fold_id=i)
        params.update(strategy="ddp_find_unused_parameters_false")
        params.update(default_root_dir=params['default_root_dir']+f'/{params_template["backbone"]}/{params["method"]}/fold_{i}')
        # params.update(init_ckpt=params['default_root_dir']+f'/{params_template["init_ckpt"]}')
        cmd = f"{CUDA_SUFFIX} python main.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
        print(cmd)
        p = Popen(cmd, shell=True)
        return_code = p.wait()
        # os.system(cmd)
    
    # EDL
    if TRAIN_EDL:
        params = deepcopy(params_template)
        params.update(fold_id=i)
        params.update(method='edl')
        params.update(strategy="ddp_find_unused_parameters_false")
        params.update(default_root_dir=params['default_root_dir']+f'/{params_template["backbone"]}/{params["method"]}/fold_{i}')
        # params.update(init_ckpt=params['default_root_dir']+f'/{params_template["init_ckpt"]}')
        cmd = f"{CUDA_SUFFIX} python main.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
        print(cmd)
        p = Popen(cmd, shell=True)
        return_code = p.wait()
        # os.system(cmd)

    # ProtoPNet
    if TRAIN_PROTO:
        for f_dim in [512]:
            for num_proto in [10]:
                params = deepcopy(params_template)
                params.update(method='protopnet')
                params.update(fold_id=i)
                params.update(prototype_dim=f_dim)
                params.update(prototype_num=num_proto)
                # params.update(init_ckpt=params_template['default_root_dir']+f'/{params["backbone"]}/basemethod/fold_{i}/lightning_logs/version_0/checkpoints/last.ckpt')
                params.update(default_root_dir=params['default_root_dir']+f'/{params["backbone"]}/{params["method"]}/fold_{i}')
                # params.update(init_ckpt=params['default_root_dir']+f'/{params_template["init_ckpt"]}')

                cmd = f"{CUDA_SUFFIX} python main.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
                print(cmd)
                p = Popen(cmd, shell=True)
                return_code = p.wait()
                # os.system(cmd)

    # KGDM
    if TRAIN_KGDM:
        for f_dim in [512]:
            for num_proto in [10]:
                params = deepcopy(params_template)
                params.update(method='kgdm')
                params.update(fold_id=i)
                params.update(prototype_dim=f_dim)
                params.update(prototype_num=num_proto)
                # params.update(init_ckpt=params_template['default_root_dir']+f'/{params["backbone"]}/edl/fold_{i}/lightning_logs/version_0/checkpoints/last.ckpt')
                params.update(default_root_dir=params['default_root_dir']+f'/{params["backbone"]}/{params["method"]}/fold_{i}')
                # params.update(init_ckpt=params['default_root_dir']+f'/{params_template["init_ckpt"]}')

                cmd = f"{CUDA_SUFFIX} python main.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
                print(cmd)
                p = Popen(cmd, shell=True)
                return_code = p.wait()
                # os.system(cmd)
    
    #PEDL
    if TRAIN_PEDL:
        for f_dim in [512]:
            for num_proto in [10]:
                params = deepcopy(params_template)
                params.update(method='pedl')
                params.update(strategy="ddp_find_unused_parameters_false")
                params.update(fold_id=i)
                params.update(prototype_dim=f_dim)
                params.update(prototype_num=num_proto)
                # params.update(init_ckpt=params_template['default_root_dir']+f'/{params["backbone"]}/protopnet/fold_{i}/lightning_logs/version_0/checkpoints/last.ckpt')
                params.update(default_root_dir=params['default_root_dir']+f'/{params["backbone"]}/{params["method"]}/fold_{i}')
                # params.update(init_ckpt=params['default_root_dir']+f'/{params_template["init_ckpt"]}')

                cmd = f"{CUDA_SUFFIX} python main.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
                print(cmd)
                p = Popen(cmd, shell=True)
                return_code = p.wait()
    

if TEST == True:
    TEST_METHODS=['kgdm','pedl','protopnet','edl']
    for method in TEST_METHODS:
        params = deepcopy(params_template)
        params.update(prototype_dim=512)
        params.update(prototype_num=10)
        params.update(method=method)
        params.update(batch_size=64)
        cmd = f"{CUDA_SUFFIX} python report.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
        print(cmd)
        p = Popen(cmd, shell=True)
        return_code = p.wait()
    exit()

if PLOT_CDF == True:
    TEST_METHODS=['edl']
    for method in TEST_METHODS:
        params = deepcopy(params_template)
        params.update(prototype_dim=512)
        params.update(prototype_num=10)
        params.update(method=method)
        params.update(batch_size=64)
        params.update(gpus=1)
        cmd = f"{CUDA_SUFFIX} python plot_ood_cdf.py "+' '.join([f"--{k} {str(v)}" for k,v in params.items()])
        print(cmd)
        p = Popen(cmd, shell=True)
        return_code = p.wait()
    exit()