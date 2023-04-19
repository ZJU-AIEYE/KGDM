import torch
import torch.nn as nn
import torch.nn.functional as F

from .protomodel import Protomodel
from torch.autograd import Function

def one_hot_embedding(labels, num_classes):
    # Convert to One Hot Encoding
    y = torch.eye(num_classes)
    # y = torch.cat([y,torch.zeros([1,num_classes])])
    return y[labels]

def loglikelihood_loss(y, alpha, device=None):
    # corresponding to equation interpretable form
    y = y.to(device)
    alpha = alpha.to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum(
        (y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return loglikelihood_err , loglikelihood_var

def mae_loss(gts, alpha):
    # corresponding to equation interpretable form

    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = F.nll_loss(torch.log(alpha / S), gts)

    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return loglikelihood_err , loglikelihood_var

def kl_divergence(alpha, num_classes, device=None):
    if not device:
        device = alpha.device
    beta = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    S_alpha = torch.sum(alpha, dim=1, keepdim=True)
    S_beta = torch.sum(beta, dim=1, keepdim=True)
    lnB = torch.lgamma(S_alpha) - \
        torch.sum(torch.lgamma(alpha), dim=1, keepdim=True)
    lnB_uni = torch.sum(torch.lgamma(beta), dim=1,
                        keepdim=True) - torch.lgamma(S_beta)

    dg0 = torch.digamma(S_alpha)
    dg1 = torch.digamma(alpha)

    kl = torch.sum((alpha - beta) * (dg1 - dg0), dim=1,
                   keepdim=True) + lnB + lnB_uni
    return kl

class EDL_Loss():
    def __init__(self, num_classes) -> None:
        self.num_classes = num_classes

    def __call__(self,res_alpha, gts, soft=False, losstype='mse'):
        y = one_hot_embedding(gts, self.num_classes)
        device = res_alpha.device
        y = y.to(device)

        if losstype=='mse':
            loglikelihood_err , loglikelihood_var = loglikelihood_loss(y.float(), res_alpha, device)
        elif losstype=='mae':
            loglikelihood_err , loglikelihood_var = mae_loss(gts, res_alpha)

        kl_alpha = (res_alpha - 1) * (1 - y) + 1
        kl_loss = kl_divergence(kl_alpha, self.num_classes,device)

        return torch.mean(loglikelihood_err+loglikelihood_var), torch.mean(kl_loss)

class KGDMmodel(Protomodel):

    def __init__(self, backbone:str, num_classes, fmap_size=[8,8], f_dim=128, num_proto=10, **kwargs) -> None:
        super().__init__(backbone, num_classes, fmap_size, f_dim, num_proto, **kwargs)
        self.kl_coef = 0
        self.save_hyperparameters()


    def _loss_metric_minibatch(self, out, minibatch, metrics):
        gt = minibatch['gt'].to(self.device)
        edl_loss = EDL_Loss(self.num_classes)

        kl_coef = min(self.kl_coef/10+0.01,1)
        alpha = out['pdl_out'].exp()
        err_loss,kl_loss = edl_loss(alpha, gt)
        prob = alpha/alpha.sum(-1, keepdim=True)
        u = self.num_classes/alpha.sum(-1)

        onehot_gt = torch.eye(self.num_classes,device=gt.device)[gt]
        l_cst = (out['sim']*onehot_gt[:,:,None]).mean()
        # [batchsize, numclass]
        l_sep = -(out['sim']*(1-onehot_gt)[:,:,None]).mean()
        l_ind = self.protolayer.indepent_loss()
        l1_loss = 0.01 * self.protolayer.l1_loss()
        proto_loss = 1*(l_cst+l_sep+l_ind)+l1_loss
        loss = err_loss+kl_coef*kl_loss+1*proto_loss
        for m in metrics:
            m.update(prob,gt)
        if torch.isnan(kl_loss):
            # print(out['cosvalue'], gt)
            loss = 0
        self.log_dict(dict(train_kl_loss=kl_loss, u=u),on_step=False,on_epoch=True)
        return loss

    def training_epoch_end(self, var_list):
        metric_names = ['tacc','tauc','tckappa','tss','tppv', 'tsp']
        scores = [m.compute() for m in self.train_metrics]
        val_res = dict(zip(metric_names, scores))
        if val_res['tacc']>0.95:
            self.kl_coef+=1
        return super().training_epoch_end(var_list)

    # def any_extra_hook(...)
