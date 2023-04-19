import os
import argparse
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from model import Basemodel, Protomodel, KGDMmodel, UDevel, Edlmodel, PEDLmodel
from data import KeraSingle
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import pandas  as pd
import numpy as np
from main import get_args, get_model
import warnings
warnings.filterwarnings("ignore")

from model import KGDMmodel,Basemodel,PEDLmodel,Protomodel,Edlmodel

MODEL_CLASS = {
    'basemethod':Basemodel,
    'edl':Edlmodel,
    'protopnet':Protomodel,
    'kgdm':KGDMmodel,
    'pedl':PEDLmodel
}

def bootstrap_estimation(all_values, r, times=1000):
    all_values = np.nan_to_num(all_values)
    n = all_values.shape[0]
    res = []
    for t in range(times):
        choice = np.random.choice(n,r,replace=True)
        t_values = all_values[choice]
        t_values = np.mean(t_values,axis=0)
        res.append(t_values)
    res = np.stack(res,0)
    return np.mean(res,0), np.percentile(res, 2.5, 0),  np.percentile(res, 97.5, 0)

def main(args):
    # model
   
    all_res = []
    root_dir =  args.default_root_dir
    rerun_data = True
    if rerun_data:
        for fold_id in range(5):
            pl.seed_everything(args.seed)
            args.fold_id=fold_id
            args.default_root_dir = root_dir+f'/{args.backbone}/{args.method}/fold_{fold_id}'
            # load data 
            
            data = KeraSingle(args.batch_size, args.num_workers, args.img_size, fold_id, 5)
            data.setup("fit")
            print(len(data.validset))
            ckpt_path_dir = args.default_root_dir+'/lightning_logs/version_0/checkpoints/'
            ckpt_paths = [os.path.join(ckpt_path_dir,p) for p in os.listdir(ckpt_path_dir) if p.startswith('KGDM')]
            trainer:Trainer = Trainer(devices=1, gpus=1)
            
            for ckpt_path in ckpt_paths:
                # save callback
                # model = get_model(args)
                model = MODEL_CLASS[args.method].load_from_checkpoint(ckpt_path)
                model.method_name = args.method
                model = UDevel(model, args.method, u_thres=np.linspace(0.1,1,10))
                trainer.test(model, datamodule=data)
                res = pd.concat(model.test_res)
                all_res.append(res)
            del data
        all_values = np.stack([res.values for res in all_res],0)
        os.makedirs(f"test_result/{args.backbone}/", exist_ok=True)
        np.save(open(f"test_result/{args.backbone}/{args.method}.npd","wb"), all_values)
    else:
        all_values = np.load(open(f"test_result/{args.backbone}/{args.method}.npd","rb"), allow_pickle=True)
   
    avg, lb, gb = bootstrap_estimation(all_values,5,1000)
    print(avg.shape, lb.shape, gb.shape)
    columns=['precision_in','recall_in', 'auc_cls_0','auc_cls_1','auc_cls_2','auc_cls_3', 'auc_avg', 'CohenKappa','Sensitivity','PPV','F1']
    report_columns=['auc_avg', 'CohenKappa','Sensitivity','PPV','F1']
    resdf = pd.DataFrame(columns=report_columns)
    for i,c in enumerate(columns):
        if not c in report_columns:
            continue
        for j in range(len(avg)):
            if c in ['precision_in', 'recall_in', 'Sensitivity', 'PPV']:
                resdf.loc[j,c]=f"{avg[j,i]*100:.2f}%({lb[j,i]*100:.2f}%,{gb[j,i]*100:.2f}%)"
            else:
                resdf.loc[j,c]=f"{avg[j,i]:.3f}({lb[j,i]:.3f}-{gb[j,i]:.3f})"

    resdf.to_csv(f"test_result/{args.backbone}/{args.method}.csv")
    resdf.to_latex(f"test_result/{args.backbone}/{args.method}.tex")
    
if __name__=='__main__':
    torch.set_printoptions(precision=10, sci_mode=None)
    # load args
    args = get_args()
    pl.seed_everything(args.seed)
    main(args)
    pass

# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method basemethod --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4 --prototype_dim 1024 --prototype_num 10 --init_ckpt /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4/lightning_logs/version_0/checkpoints/last.ckpt
# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method protopnet --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121_protopnet_fold_4 --prototype_dim 1024 --prototype_num 10 --init_ckpt /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4/lightning_logs/version_0/checkpoints/last.ckpt
# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method kgdm --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121_kgdm_fold_4 --prototype_dim 1024 --prototype_num 10 --init_ckpt /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4/lightning_logs/version_0/checkpoints/last.ckpt