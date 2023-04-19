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

def get_args(handle_args=None):
    parser = argparse.ArgumentParser()
    
    # Model setting
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--prototype_dim', type=int, default=128)
    parser.add_argument('--prototype_num', type=int, default=10)
    parser.add_argument('--backbone', type=str, default='resnet50')
    parser.add_argument('--global_pooling', type=str,default='max')
    parser.add_argument('--method', type=str, default='basemethod')

    # Training setting
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--is_train', type=int, default=1)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--fold_id', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=75)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=16)
    parser.add_argument('--save_model_dir', type=str, default='/data/home/fangzhengqing/Result/KeraBase')
    parser.add_argument('--init_ckpt', type=str, default='')

    # Test setting
    parser.add_argument('--u_threshold', type=float, default=0.2)
    parser.add_argument('--eval_ud', type=int, default=0)

    parser.add_argument_group(title="pl.Trainer args")
    parser = Trainer.add_argparse_args(parser)

    if handle_args:
        args = parser.parse_args(handle_args)
    else:
        args = parser.parse_args()
    return args

def get_model(args)->LightningModule:
    fmap_size_dict={
            '256':[8,8],
            '384':[12,12]
        }
    model_config=dict(backbone=args.backbone, pooling=args.global_pooling, num_classes=args.num_classes)
    if args.method == 'basemethod':
        model = Basemodel(**model_config) if args.init_ckpt=='' else Basemodel.load_from_checkpoint(args.init_ckpt, strict=False,**model_config)
    elif args.method == 'edl':
        model = Edlmodel(**model_config) if args.init_ckpt=='' else Edlmodel.load_from_checkpoint(args.init_ckpt, strict=False, **model_config)
    elif args.method == 'protopnet':
        model_config.update(f_dim=args.prototype_dim, fmap_size=fmap_size_dict[str(args.img_size)], num_proto=args.prototype_num)
        model = Protomodel(**model_config) if args.init_ckpt=='' else Protomodel.load_from_checkpoint(args.init_ckpt, strict=False, **model_config)
    elif args.method == 'kgdm':
        model_config.update(f_dim=args.prototype_dim, fmap_size=fmap_size_dict[str(args.img_size)], num_proto=args.prototype_num)
        model = KGDMmodel(**model_config) if args.init_ckpt=='' else KGDMmodel.load_from_checkpoint(args.init_ckpt, strict=False, **model_config)
    elif args.method == 'pedl':
        model_config.update(f_dim=args.prototype_dim, fmap_size=fmap_size_dict[str(args.img_size)], num_proto=args.prototype_num)
        model = PEDLmodel(**model_config) if args.init_ckpt=='' else PEDLmodel.load_from_checkpoint(args.init_ckpt, strict=False, **model_config)
    model.method_name = args.method
    return model


def main(args):
    
    # load data 
    data = KeraSingle(args.batch_size, args.num_workers, args.img_size, args.fold_id, 5)

    # model
    model = get_model(args)
    ckpt_path = args.default_root_dir+'/best_ckpt.ckpt'
    
    if args.is_train==1:
        trainer:Trainer = Trainer.from_argparse_args(args)
        # save callback
        checkpoint_callback = ModelCheckpoint(
            monitor='acc',
            filename='KGDM-'+f'{args.backbone}-f{args.fold_id}'+'-{epoch:02d}-{acc:.6f}',
            save_top_k=6,
            mode='max',
            save_last=True
        )
        
        trainer.callbacks.append(checkpoint_callback)
        trainer.fit(model, datamodule=data)
        model=type(model).load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
        
        trainer.save_checkpoint(ckpt_path)
    else:
        
        # ckpt_path = '/data/home/fangzhengqing/Result/KGDM_ablation/resnet50/basemethod/fold_0/lightning_logs/version_3/checkpoints/KGDM-resnet50-f0-epoch=66-acc=0.712269.ckpt'
        
        data.setup('fit')
        model=type(model).load_from_checkpoint(ckpt_path)

        trainer:Trainer = Trainer(devices=1, gpus=1)
        # model.calibration(data.val_dataloader())
        model = UDevel(model, args.method, u_thres=np.linspace(0.5,1,5))
        
        trainer.test(model, datamodule=data)
        for i,r in enumerate(model.test_res):
            print(r,file=open(f"{args.default_root_dir}/test_res.txt","a"))
            r.to_csv(f"{args.default_root_dir}/test_res_{i}.csv")

if __name__=='__main__':
    torch.set_printoptions(precision=10, sci_mode=None)
    # load args
    args = get_args()
    pl.seed_everything(args.seed)
    main(args)
    pass

# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method basemethod --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4 --prototype_dim 1024 --prototype_num 10 --init_ckpt /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4/lightning_logs/version_0/checkpoints/last.ckpt
# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method protopnet --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121_protopnet_fold_4 --prototype_dim 1024 --prototype_num 10 --init_ckpt /data/home/fangzhengqing/Result/KGDM/densenet121_basemethod_fold_4/lightning_logs/version_0/checkpoints/last.ckpt
# CUDA_VISIBLE_DEVICES=3,4 python main.py --eval_ud 1 --is_train 0 --fast_dev_run False --auto_select_gpus True --log_every_n_steps 1 --backbone densenet121 --method kgdm --gpus 1 --max_epochs 62 --batch_size 32 --fold_id 4 --img_size 384 --num_classes 4 --precision 16 --default_root_dir /data/home/fangzhengqing/Result/KGDM/densenet121/kgdm_fold_4 