from .kgdmmodel import EDL_Loss, KGDMmodel
import torch.nn.functional as F
import torch
from utils.pcgrad.pcgrad import PCGrad
from .protomodel import CossimMatrix
from types import MethodType
class CossimMatrixPartial(CossimMatrix):

    @staticmethod
    def backward(ctx, grad_outputs):
        a,b,na,nb,cossim = ctx.saved_tensors
        # ga = torch.einsum('bmn,bmnd->bmd',[grad_outputs,(torch.einsum('bdn,bmn->bmnd',[b,na**-1*nb**-1])-torch.einsum('bmn,bmd->bmnd',[cossim,a*na**-2]))])
        gb = torch.einsum('bmdn,bmn->bdn',[(torch.einsum('bmd,bmn->bmdn',[a,na**-1*nb**-1])-torch.einsum('bmn,bdn->bmdn',[cossim,b*nb**-2])),grad_outputs])
        return (torch.zeros_like(a), gb)

def cossim_fn(self):
    return CossimMatrixPartial
    
class PEDLmodel(KGDMmodel):
    def __init__(self, backbone: str, num_classes, fmap_size=..., f_dim=128, num_proto=10, **kwargs) -> None:
        super().__init__(backbone, num_classes, fmap_size, f_dim, num_proto, **kwargs)
        # self.protolayer.cossim_fn = CossimMatrixPartial

    def forward(self, x):
        x = self.feature(x)
        proto_res = self.protolayer(x)

        x = F.adaptive_avg_pool2d(x,1)
        x = x.flatten(1)
        out = self.classifier(x)
        proto_res.update(cls_out=out)
        return proto_res

    def _loss_metric_minibatch(self, out, minibatch, metrics):
        gt = minibatch['gt'].to(self.device)
        edl_loss = EDL_Loss(self.num_classes)
        kl_coef = min(self.kl_coef/10+0.01,1)
        alpha = out['cls_out'].relu()+1
        err_loss,kl_loss = edl_loss(alpha, gt)
        u=self.num_classes/alpha.sum(-1)
        
        b = minibatch['img'].shape[0]
        onehot_gt = torch.eye(self.num_classes,device=gt.device)[gt]
        ce_loss = F.cross_entropy(out['pdl_out'],gt)
        err_loss2,kl_loss2 = edl_loss(out['pdl_out'].exp(), gt)
        l_cst = (out['sim']*onehot_gt[:,:,None]).mean()
        # [batchsize, numclass]
        l_sep = -(out['sim']*(1-onehot_gt)[:,:,None]).mean()
        l_ind = self.protolayer.indepent_loss()
        l1_loss = 0.01 * self.protolayer.l1_loss()
        proto_loss = 1*(l_cst+l_sep+l_ind)+l1_loss
        
        loss = err_loss+kl_coef*kl_loss + err_loss2+kl_coef*kl_loss2 + proto_loss
        prob = out['pdl_out'].softmax(-1)
        for m in metrics:
            m.update(prob,gt)
        self.log_dict(dict(train_kl_loss=kl_loss, u=u,l_ind=l_ind,ce_loss=ce_loss,err_loss=err_loss, err_loss2=err_loss2),on_step=False,on_epoch=True)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.SGD([{'params':self.protolayer.proto_parameters(),'lr':1e-2}], weight_decay=0)
        opt2 = torch.optim.AdamW([{'params':self.feature.parameters(),'lr':1e-4}, {'params':self.protolayer.proto_parameters(),'lr':1e-2}, {'params':self.protolayer.fc_parametters(),'lr':1e-4}, {'params':self.classifier.parameters(),'lr':1e-4}], betas=(0.9,0.99), eps=1e-6, weight_decay=1e-2)
        opt3 = torch.optim.SGD([ {'params':self.protolayer.proto_parameters(),'lr':1e-2}, {'params':[self.protolayer.weight],'lr':1e-2,'weight_decay':0}, {'params':self.classifier.parameters(),'lr':1e-4}], weight_decay=0)
        
        # opt_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,5,2,1e-5)
        # opt_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,5,2,1e-5)
        opt_sch = torch.optim.lr_scheduler.StepLR(opt2,10,0.5)
        return [[opt,opt2,opt3], [opt_sch]]