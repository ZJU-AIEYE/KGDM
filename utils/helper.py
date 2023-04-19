from typing import OrderedDict
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, balanced_accuracy_score,roc_curve,auc
import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from  scipy.interpolate import interpn


class MeanRecorder(object):
    def __init__(self) -> None:
        super(MeanRecorder).__init__()
        self.data = {}

    def init(self):
        self.data = {}

    def update(self, datas):
        for k,v in datas.items():
            v = v.detach().cpu().numpy() if  type(v) == torch.Tensor else v
            self.data[k] = self.data[k]+[v] if k in self.data.keys() else [v]

    def values(self):
        for k in self.data.keys():
            try:
                yield (k, np.mean(self.data[k]))
            except:
                continue

    def mean_var(self):
        for k in self.data.keys():
            try:
                yield (k,(np.mean(self.data[k],-1), np.std(self.data[k])))
            except:
                continue
    
    def dict(self):
        return OrderedDict(self.values())

    def __repr__(self) -> str:
        return ",".join([ f"{k}={v:.6f}" for k,v in self.values()])

def onehot(y_label):
    # 将整数数组转为一个10位的one hot编码
    num_classes = np.unique(y_label).shape[0]
    return np.eye(num_classes)[y_label]

def plot_roc_curve(fpr, tpr,roc_auc,n_classes):
    # Plot all ROC curves
    lw = 2
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
             label='micro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["micro"]),
             color='deeppink', linestyle=':', linewidth=4)

    plt.plot(fpr["macro"], tpr["macro"],
             label='macro-average ROC curve (area = {0:0.2f})'
                   ''.format(roc_auc["macro"]),
             color='navy', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue','red','blue'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('multi-calss ROC')
    plt.legend(loc="lower right")
    plt.savefig('show.jpg')
    
    

class Confusion_matrix(object):
    def __init__(self, dim):
        super(Confusion_matrix, self).__init__()
        self.cms = dict()
        self.dim = dim
        self.prop = dict()

    def update(self, name:str, pred, target, pred_prop=None):
        assert len(pred) == len(target)
        pred = pred.detach().cpu().numpy() if type(pred) == torch.Tensor else pred
        target = target.detach().cpu().numpy() if type(target) == torch.Tensor else target
        self.cms[name] =self.addnew(self.cms[name], pred,target) if name in self.cms.keys() else (pred,target)

        if not pred_prop is None:
            pred_prop = pred_prop.detach().cpu().numpy() if type(pred_prop) == torch.Tensor else pred_prop
            self.prop[name] = np.concatenate([self.prop[name],pred_prop],axis=0) if name in self.prop.keys() else pred_prop

    def addnew(self,cm,pred,target):
        x = np.concatenate([cm[0],pred],0)
        y = np.concatenate([cm[1],target],0)
        return (x, y)

    def get_cfm(self,name):
        cm = self.cms[name]
        return confusion_matrix(cm[1], cm[0])

    def get_metric(self, name):
        cm = self.cms[name]
        
        cm = self.cms[name]
        result = {
            'f1':f1_score(cm[1],cm[0],average=None),
            'wf1' : f1_score(cm[1],cm[0],average='weighted'),
            'mif1' : f1_score(cm[1],cm[0],average='micro'),
            'maf1' : f1_score(cm[1],cm[0],average='macro'),
            'bacc' : balanced_accuracy_score(cm[1],cm[0]),
            'cfm' : confusion_matrix(cm[1], cm[0])
        }
        result['acc']=self.get_acc(name)
        

        if name in self.prop.keys():
            pred_prop = self.prop[name]
            n_classes = len(np.unique(cm[1]))
            y_onehot = onehot(cm[1])
            # print(pred_prop[0:16])
            # 计算每一类的ROC
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_onehot[:, i], pred_prop[:,i])
                roc_auc[i] = auc(fpr[i], tpr[i])

            # micro roc 曲线绘制
            fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot.ravel(), pred_prop.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

            # macro roc 曲线绘制, 对图上的所有点做线性插值
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            mean_tpr /= n_classes
            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"],tpr["macro"])
            result['fpt']=fpr
            result['tpr']=tpr
            result['roc_auc']=roc_auc
            result['n_classes'] = n_classes
        return result

    def get_overall(self,name):
        cm = self.cms[name]
        result = {
            'wf1' : f1_score(cm[1],cm[0],average='weighted'),
            'mif1' : f1_score(cm[1],cm[0],average='micro'),
            'maf1' : f1_score(cm[1],cm[0],average='macro'),
            'bacc' : balanced_accuracy_score(cm[1],cm[0]),
        }
        result['acc']=self.get_acc(name)
        return result

    def print_res(self,name):
        res = self.get_metric(name)
        
        msg = "\nf1:"
        for f in res['f1']:
            msg+=f" {f:.3f}"
        for k in ['wf1','mif1','maf1','bacc','acc']:
            msg += f",{k}: {res[k]:.3f}"
        msg += f"\n{res['cfm']}"

        if name in self.prop.keys():
            msg += "\nauc:"
            roc_auc = res['roc_auc']
            for i in range(res['n_classes']):
                msg+=f" {roc_auc[i]:.3f}"
            msg += f",micro_auc: {roc_auc['micro']:.3f}, maauc_: {roc_auc['macro']:.3f}\n"
            
        return msg

    def get_acc(self, name):
        cm = self.cms[name]
        cm = confusion_matrix(cm[1],cm[0])
        return np.trace(cm)/np.sum(cm)