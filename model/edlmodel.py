import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from .basemodel import Basemodel
from .kgdmmodel import EDL_Loss


class Edlmodel(Basemodel):
    def __init__(self, backbone: str, num_classes, **kwargs) -> None:
        super().__init__(backbone, num_classes, **kwargs)
        self.kl_coef = 0

    def _loss_metric_minibatch(self, out, minibatch, metrics):
        gt = minibatch['gt'].to(self.device)
        edl_loss = EDL_Loss(self.num_classes)
        kl_coef = min(self.kl_coef/10+0.01,1)
        alpha = out.relu()+1
        err_loss,kl_loss = edl_loss(alpha, gt)
        prob = alpha/alpha.sum(-1, keepdim=True)
        u = self.num_classes/alpha.sum(-1)
        loss = err_loss+kl_coef*kl_loss
        for m in metrics:
            m.update(prob,gt)
        self.log_dict(dict(train_kl_loss=kl_loss, u=u),on_step=False,on_epoch=True)
        return loss

    def training_epoch_end(self, var_list):
        metric_names = ['tacc','tauc','tckappa','tss','tppv', 'tsp']
        scores = [m.compute() for m in self.train_metrics]
        val_res = dict(zip(metric_names, scores))
        if val_res['tacc']>0.95:
            self.kl_coef+=1
        self.log_dict(val_res, on_step=False, on_epoch=True)
        _ = [m.reset() for m in self.train_metrics]


        