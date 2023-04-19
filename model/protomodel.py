import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule, LightningDataModule
from torchmetrics import Recall, Specificity, Precision, CohenKappa, Accuracy, AUROC
from .basemodel import Basemodel
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel
from copy import deepcopy

def weight_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.Linear):
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
        nn.init.kaiming_normal_(m.weight.data)

class CossimMatrix(Function):
    # input: like a @ b
    # a: [b,m,d]
    # b: [b,d,n]
    @staticmethod
    def forward(ctx, a,b, eps=1e-3):
        na = a.norm(dim=2, p=2, keepdim=True)+eps # na: [b,m,1]
        nb = b.norm(dim=1, p=2, keepdim=True)+eps # nb: [b,1,n]
        cossim = (a*na**-1)@(b*nb**-1) # cossim: [b,m,n]
        ctx.save_for_backward(a,b,na,nb,cossim)
        return cossim

    # ga: [b,m,d]
    # gb: [b,d,n]
    # grad_outputs: [b,m,n]
    # b*na**-1*nb**-1 : [b,d,n] @ [b,m,n]
    # cossim*a*na**-2 : [b,m,n] @ [b,m,d]
    # a*na**-1*nb**-1 ：[b,m,d] @ [b,m,n]
    # cossim*b*nb**-2 : [b,m,n] @ [b,d,n]
    @staticmethod
    def backward(ctx, grad_outputs):
        a,b,na,nb,cossim = ctx.saved_tensors
        ga = torch.einsum('bmn,bmnd->bmd',[grad_outputs,(torch.einsum('bdn,bmn->bmnd',[b,na**-1*nb**-1])-torch.einsum('bmn,bmd->bmnd',[cossim,a*na**-2]))])
        gb = torch.einsum('bmdn,bmn->bdn',[(torch.einsum('bmd,bmn->bmdn',[a,na**-1*nb**-1])-torch.einsum('bmn,bdn->bmdn',[cossim,b*nb**-2])),grad_outputs])
        return (ga, gb)



class ProtoLayer(nn.Module):
    def __init__(self, num_classes, num_features, fmap_size=[8,8], f_dim=128, num_proto=10) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.num_proto = num_proto
        self.fmap_size = fmap_size
        self.f_dim = f_dim

        self.flatten = nn.Flatten(start_dim=2)
        # self.fc = nn.Linear(num_features, f_dim)

        self.fc = nn.Identity()
        f_dim = num_features
        self.f_dim = num_features

        

        self.register_parameter('prototype', nn.parameter.Parameter(num_features**-0.5*torch.randn([1,f_dim,num_classes*num_proto])))
        self.register_parameter('weight', nn.parameter.Parameter(torch.ones([num_classes,num_proto])))
        self.register_parameter('weight_e', nn.parameter.Parameter(10*torch.ones([num_classes,num_proto])))
        self.register_buffer('eye',torch.eye(num_proto, requires_grad=False))
        # self.register_buffer('prototype_bk',torch.rand([1,f_dim,num_classes*num_proto]))

        self.apply(weight_init_kaiming)
    
    def cossim_fn(self):
        return CossimMatrix

    def indepent_loss(self):
        prototype =  F.normalize(self.prototype, p=2,dim=-2)
        id_mtx = self.cossim_fn().apply(prototype.permute(0,2,1), prototype)
        device = id_mtx.device
        eye = torch.eye(self.prototype.shape[2], device=device)
        return (id_mtx-eye).exp().mean()

    def l1_loss(self):
        l1_norm = F.normalize(self.weight,p=1,dim=-1)
        return l1_norm.sum()

    def proto_parameters(self):
        params = [self.prototype]
        return params
    
    def fc_parametters(self):
        return list(self.fc.parameters())

    def _update_bk_proto(self, prototype_bk):
        sd = self.state_dict()
        sd.update(prototype=nn.parameter.Parameter(prototype_bk))
        self.load_state_dict(sd)
        # self.prototype=nn.parameter.Parameter(self.prototype_bk.clone())

    def forward(self,x):
        x = F.adaptive_avg_pool2d(x,self.fmap_size)
        f = self.flatten(x)
        b = x.shape[0]
        f = self.fc(f.mT)
        f = F.normalize(f,p=2,dim=-1)
        prototype =  F.normalize(self.prototype, p=2,dim=-2)
        cosvalue = self.cossim_fn().apply(f,prototype.expand(b,-1,-1))
        cosvalue = cosvalue.permute(0,2,1)
        
        if False:
            dist = -cosvalue+1
            dist_bk = dist.clone()
            dist = torch.log((dist+1)/(dist+0.001))-torch.log(torch.tensor(2, device=dist.device))
        else:
            # dist = torch.tan(cosvalue) 
            dist = cosvalue
            dist_bk = -cosvalue+1
        sim,argmaxdist = dist.max(2)
        maxfs = f[torch.arange(0,b,device=sim.device)[:,None],argmaxdist]
        sim = sim.unflatten(1,(self.num_classes, self.num_proto))
        contrib = sim*self.weight*self.weight_e.sigmoid()
        out = contrib.sum(2)

        proto_res = {
            'cosvalue': cosvalue.clone().unflatten(1,(self.num_classes, self.num_proto)), # [b, num_classes, num_proto，h, w]
            'cosdist':dist_bk.unflatten(1,(self.num_classes, self.num_proto)),
            'maxfs':maxfs.clone().unflatten(1,(self.num_classes, self.num_proto)),
            'f':f.clone().detach(),
            'sim':sim,
            'pdl_out': out,
            'contrib': contrib,
            'argmaxdist':argmaxdist
        }
        return proto_res
class Protomodel(Basemodel):
    # The class that standarlize the data input, model output and metric benchmark
    # turtriol: https://zhuanlan.zhihu.com/p/353985363

    def __init__(self, backbone:str, num_classes, fmap_size=[8,8], f_dim=128, num_proto=10, **kwargs) -> None:
        super().__init__(backbone, num_classes, **kwargs)
        self.protolayer = ProtoLayer(num_classes, self.num_filters, fmap_size, f_dim, num_proto)

        self.automatic_optimization=False
        self.save_hyperparameters()

    def forward(self, x):
        x = self.feature(x)
        proto_res = self.protolayer(x)
        return proto_res


    def _loss_metric_minibatch(self, out, minibatch, metrics):
        gt = minibatch['gt'].to(self.device)
        b = minibatch['img'].shape[0]
        ce_loss = F.cross_entropy(out['pdl_out'],gt)

        # find prototype position and target class cosine  distance should be minimize
        onehot_gt = torch.eye(self.num_classes,device=gt.device)[gt]
        l_cst = (out['sim']*onehot_gt[:,:,None]).mean()
        # [batchsize, numclass]
        l_sep = -(out['sim']*(1-onehot_gt)[:,:,None]).mean()
        l_ind = self.protolayer.indepent_loss()
        l1_loss = 0.01 * self.protolayer.l1_loss()
        proto_loss = 1*(l_cst+l_sep+l_ind)+l1_loss

        loss = ce_loss+proto_loss
        # loss = loss+ l1_loss if self.current_epoch > 70 else loss

        prob = out['pdl_out'].softmax(-1)
        for m in metrics:
            m.update(prob,gt)
        loss_dict = dict(loss=loss, l_cst=l_cst, l_sep=l_sep, l_ind=l_ind, l1_loss=l1_loss)
        if self.training:
            self.log_dict(loss_dict, on_epoch=True, on_step=False)
        if  torch.isnan(loss):
            print('nan')
            pass
        return loss

    def configure_optimizers(self):
        opt = torch.optim.AdamW([{'params':self.protolayer.proto_parameters(),'lr':1e-3}], betas=(0.9,0.99), eps=1e-6, weight_decay=0)
        opt2 = torch.optim.AdamW([{'params':self.feature.parameters(),'lr':1e-4}, {'params':self.protolayer.fc_parametters(),'lr':1e-4}], betas=(0.9,0.99), eps=1e-6, weight_decay=1e-2)
        opt3 = torch.optim.AdamW([{'params':self.protolayer.proto_parameters(),'lr':1e-3}, {'params':[self.protolayer.weight],'lr':1e-2}], betas=(0.9,0.99), eps=1e-6, weight_decay=1e-2)
        
        # opt_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt2,5,2,1e-5)
        opt_sch = torch.optim.lr_scheduler.StepLR(opt2,10,0.5)
        return [[opt,opt2,opt3], [opt_sch]]
    
    def training_step(self, minibatch, batch_idx):
        _ = [m.to(self.device) for m in self.train_metrics]
        opt, opt2, opt3 = self.optimizers()
        
        out = self._forward_minibatch(minibatch)
        loss = self._loss_metric_minibatch(out, minibatch, self.train_metrics)
        
        if  torch.isnan(loss):
            pass
        if self.trainer.current_epoch>30:
            opt.zero_grad()
            opt2.zero_grad()
            opt3.zero_grad()
            self.manual_backward(loss)
            if not torch.isnan(loss):
                opt3.step()
        else:
            opt.zero_grad()
            opt2.zero_grad()
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_value_(self.parameters(), 0.5)
            if not torch.isnan(loss):
                opt.step()
                opt2.step()
        # sch = self.lr_schedulers()
        # sch.step(self.current_epoch+batch_idx/len(self.trainer.train_dataloader))
        return dict(train_loss=loss)
    
    def training_epoch_end(self, var_list):
        sch = self.lr_schedulers()
        sch.step()

        # if self.trainer.current_epoch>20 and self.trainer.current_epoch % 10==0:
        if  self.trainer.current_epoch == 70:
            self.replace_nearst_patch()
        return super().training_epoch_end(var_list)

    def replace_nearst_patch(self):
        datamodule:LightningDataModule = self.trainer.datamodule
        from .interp_proto import Explainer
        tmpmodel = Explainer(self)
        tmpmodel.construct_prototypes(datamodule.train_dataloader())
        vcs = [tmpmodel.prototypes[i][j].visual_concept_vec() for i in range(self.num_classes)  for j in range(self.protolayer.num_proto)]
        prototype_bk = torch.stack(vcs,dim=0).permute(1,0)[None,:]
        self.protolayer._update_bk_proto(prototype_bk)
        

    # def any_extra_hook(...)
