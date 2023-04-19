from .basemodel import Basemodel
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics import Recall, Specificity, Precision, CohenKappa, Accuracy, AUROC, F1Score, StatScores,Metric
from collections import OrderedDict
from types import MethodType
from copy import deepcopy
from typing import List
import pandas as pd
from .kgdmmodel import one_hot_embedding

from torchmetrics import Metric
from sklearn.metrics import roc_auc_score
class OvO_MAUC(Metric):
    def __init__(self,num_classes, average="macro", ):
        super().__init__()
        self.average = average
        self.num_classes = num_classes
        self.add_state("prob", default=[], dist_reduce_fx="cat", persistent=True)
        self.add_state("gt", default=[], dist_reduce_fx="cat", persistent=True)

    def update(self, probs: torch.Tensor, target: torch.Tensor):
        assert probs.shape[0] == target.shape[0]

        self.prob.append(probs)
        self.gt.append(target)
        

    def compute(self):
        if len(self.gt)==0:
            return torch.tensor(-1)
        prob = torch.cat(self.prob,dim=0)
        gt = torch.cat(self.gt, dim=0)
        
        aucs = roc_auc_score(gt.detach().cpu().numpy(), prob.detach().cpu().numpy(), multi_class="ovo", average=self.average, labels=list(range(self.num_classes)))
        return torch.tensor(aucs)
    
class UDMetrics(Metric):
    def __init__(self, name, num_classes, thres) -> None:
        super().__init__()
        self.name = name
        self.num_classes = num_classes+1
        self.thres = thres

        self.metrics = nn.ModuleDict({
            'Stat':StatScores(task='binary', threshold=thres),
            # 'Acc':Accuracy(num_classes=self.num_classes, task='multiclass', average=None),
            'AUC':AUROC(num_classes = self.num_classes, average=None, task='multiclass'),
            'CohenKappa':CohenKappa(num_classes=self.num_classes, task='multiclass'),
            'Sensitivity':Recall(num_classes=self.num_classes,task='multiclass', average=None),
            'PPV':Precision(num_classes=self.num_classes, task='multiclass', average=None),
            # 'F1':F1Score(num_classes=self.num_classes,task='multiclass', average=None)
        })
    
    def to(self,device):
        for k,v in self.metrics.items():
            v.to(device)

    def update(self, prob, gt, u=None):
        prob, gt = deepcopy(prob), deepcopy(gt)
        pred = prob.argmax(-1) # prediction result
        if u is None:
            u = -(prob*prob.log()).sum(-1)/torch.log(torch.tensor(self.num_classes, device=prob.device))
        pred_is_ood = u>self.thres 
        pred[pred_is_ood]=self.num_classes-1
        
        gt[gt>=self.num_classes]=self.num_classes-1
        prob = torch.cat([prob,torch.zeros_like(u)[:,None]],-1)
        # gt_onehot = one_hot_embedding(gt, self.num_classes)
        for k,v in self.metrics.items():
            if k=='Stat':
                v.update(u,gt==self.num_classes-1)
            elif k=='AUC':
                # prob_iid = prob[~pred_is_ood]
                # gt_iid = gt[~pred_is_ood]
                prob_iid = prob
                gt_iid = gt
                if not len(prob_iid)==0:
                    v.update(prob_iid,gt_iid)
            else:
                # pred_iid = pred[~pred_is_ood]
                # gt_iid = gt[~pred_is_ood]
                pred_iid = pred
                gt_iid = gt
                if not len(pred_iid)==0:
                    v.update(pred_iid, gt_iid)
                

    def compute(self):
        res_dict = OrderedDict()
        for k,v in self.metrics.items():
            if k=='Stat':
                values = v.compute()
                tp, fp, tn, fn, _ =values
                res_dict.update([(f'precision_in',tn/(tn+fn)), (f'recall_in', tn/(tn+fp))])
            elif k=='AUC':
                try:
                    aucs = v.compute()
                except:
                    aucs = torch.zeros([self.num_classes], device=self.device)
                res_dict.update(dict(zip([f'auc_cls_{i}' for i in range(self.num_classes-1)],aucs)))
                res_dict.update( {f'auc_avg': aucs[:self.num_classes-1].mean()})
            elif k=='CohenKappa':
                res_dict.update({k: v.compute()})
            else:
                values = v.compute()
                res_dict.update({k: values[:self.num_classes-1].mean()})
        res_dict.update(F1=(2*res_dict["Sensitivity"]*res_dict["PPV"])/(res_dict["Sensitivity"]+res_dict["PPV"]))
        return res_dict

    def reset(self):
        for k,v in self.metrics.items():
            v.reset()
    
    @property
    def pl_log_dict(self):
        return {k+f'_{self.name}':v for k,v in self.metrics.items()}

def test_step(self:Basemodel, minibatch, batch_idx, test_idx=0):
    if isinstance(minibatch, List):
        gt = minibatch[1].to(self.device)
    else:
        gt = minibatch['gt'].to(self.device)
    out = self._forward_minibatch(minibatch)
    if self.method_name=='basemethod':
        prob = out.softmax(-1) 
    elif self.method_name=='protopnet':
        prob = out['pdl_out'].softmax(-1)
    elif self.method_name=='kgdm':
        alpha = out['pdl_out'].exp()
        prob = alpha/alpha.sum(-1, keepdim=True)
        u = self.num_classes/alpha.sum(-1)
    elif self.method_name=='pedl':
        alpha = out['pdl_out'].exp()
        prob = alpha/alpha.sum(-1, keepdim=True)
        # prob=alpha/alpha.sum(-1, keepdim=True)
        u = self.num_classes/alpha.sum(-1)
    elif self.method_name=='edl':
        logits = out.relu()
        alpha = logits+1
        prob = alpha/alpha.sum(-1, keepdim=True)
        u = self.num_classes/alpha.sum(-1)
    key = f"test_{test_idx}"
    metric = self._generate_metirc() if not key in self.test_metrics.keys() else self.test_metrics[key]
    for m in metric:
        m.to(self.device)
        if self.method_name in ['kgdm','edl','pedl']:
            m.update(prob,gt)
        else:
            m.update(prob,gt)
        # self.log_dict(m.pl_log_dict)

    if not key in self.test_metrics.keys():
        self.test_metrics[key]=metric

    return test_idx

def test_epoch_end(self:Basemodel,res_list):
    n = len(self.test_metrics.keys())
    # print(self.trainer._results.dataloader_idx)
    self.test_res = []
    for test_idx in range(n):
        key = f"test_{test_idx}"
        metric = self.test_metrics[key]
        test_res = [m.compute() for m in metric]
        # merge
        test_res = {k:[r[k].cpu().detach().numpy() for r in test_res] for k in test_res[0].keys()}
        index = [m.thres for m in metric]
        test_res = pd.DataFrame(test_res,index=index)
        self.test_res.append(test_res)
        for m in metric:
            m.reset()


def _generate_metirc(self:Basemodel):
    return nn.ModuleList([UDMetrics(f'u{u}', self.num_classes, u) for u in self.u_thres])

def UDevel(model:Basemodel, method, u_thres=[0.2, 0.4, 0.6, 0.8]):
    model=deepcopy(model)
    model.method_name=method
    model.u_thres = u_thres
    model.test_metrics = nn.ModuleDict()
    model._generate_metirc = MethodType(_generate_metirc, model)
    model.test_step = MethodType(test_step, model)
    model.test_epoch_end = MethodType(test_epoch_end, model)
    return model

# class UDeval_interface(Basemodel):
#     def __init__(self, u_thres=[0.2, 0.4, 0.6, 0.8], **kwargs) -> None:
#         super().__init__(**kwargs)
#         self.save_hyperparameters()
#         self.test_metric = UDMetrics('test', self.num_classes, u_thres)
    
#     def test_step(self, minibatch, batch_idx, test_idx=0):
        
#         out = self._forward_minibatch(minibatch)
#         prob = out.softmax(-1)
#         self.test_metric.to(self.device)
#         self.test_metric.update(prob,minibatch['gt'])

#     def test_epoch_end(self,res_list):
        
#         print(len(res_list))
#         test_res = self.test_metric.compute()
#         self.log_dict(test_res, sync_dist=True, on_epoch=True)
#         self.test_metric.reset()
#         self._current_fx_name



    