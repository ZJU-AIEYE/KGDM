import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics import Recall, Specificity, Precision, CohenKappa, Accuracy, AUROC
# from torchmetrics.classification import MulticlassAUROC
from torchvision.models import resnet50,resnet101,resnet152,densenet121,densenet201,\
    vgg16_bn,vgg19_bn,inception_v3, vit_b_16, vit_b_32, swin_s, VisionTransformer, SwinTransformer
from typing import List
from types import MethodType
from .temperature_clb import ModelWithTemperature
class Vit_forward(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        for k,v in model.__dict__.items():
            setattr(self,k,v)
        self._process_input = model._process_input

    def forward(self, x: torch.Tensor):
        n, c, h, w = x.shape
        p = self.patch_size
        n_h = h // p
        n_w = w // p
        # Reshape and permute the input tensor
        x = self._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.encoder(x)

        # Classifier "token" as used by standard language architectures
        x = x[:, 1:].mT.unflatten(-1,(n_h,n_w))
        return x

class Swin_feature(nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        for k,v in model.__dict__.items():
            setattr(self,k,v)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)
        return x

def get_model(k):
    model_fn={
        'resnet50':resnet50,
        'resnet101':resnet101,
        'densenet121':densenet121,
        'densenet201':densenet201,
        'vgg16':vgg16_bn,
        'vgg19':vgg19_bn,
        'vit_b_16':vit_b_16,
        'vit_b_32':vit_b_32,
        'swin_s':swin_s,
    }
    pretrain_params={
        "weights":"IMAGENET1K_V1",
    }
    model = model_fn[k](**pretrain_params)
    if k in ['resnet50','resnet101']:
        # pretrain_path = '/home/fangzhengqing/Code/byol_kera/byol_resnet50_e150.pt'
        # sd = torch.load(open(pretrain_path,'rb'), map_location='cpu')
        # model.load_state_dict(sd)
        num_filters = model.fc.in_features
        modules = list(model.children())[:-2]
        model = nn.Sequential(*modules)
    elif k in ['densenet121','densenet201']:
        num_filters = model.classifier.in_features
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    elif k in ['vgg16','vgg19']:
        num_filters = 512
        modules = list(model.children())[:-1]
        model = nn.Sequential(*modules)
    elif k in ['vit_b_16','vit_b_32']:
        model = model_fn[k](weights='IMAGENET1K_SWAG_E2E_V1')
        num_filters = model.heads.head.in_features
        model = Vit_forward(model)
    elif k in ['swin_s']:
        num_filters = model.head.in_features
        model = Swin_feature(model)
    return model, num_filters

class Basemodel(LightningModule):
    # The class that standarlize the data input, model output and metric benchmark
    # turtol: https://zhuanlan.zhihu.com/p/353985363

    def __init__(self, backbone:str, num_classes, pooling='avg') -> None:
        super().__init__()
        self.feature, self.num_filters = get_model(backbone)
        self.classifier = nn.Linear(self.num_filters, num_classes)
        self.num_classes = num_classes
        self._pooling_method = pooling

        self.train_metrics = [m(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device) for m in [Accuracy, AUROC, CohenKappa, Recall, Precision, Specificity]]
        self.val_metrics = [m(num_classes=self.num_classes, average='macro',task='multiclass').to(self.device) for m in [Accuracy, AUROC, CohenKappa, Recall, Precision, Specificity]]
        self.test_metrics = [m(num_classes=self.num_classes, average='macro', task='multiclass').to(self.device) for m in [Accuracy, AUROC, CohenKappa, Recall, Precision, Specificity]]
        self.temperature = None
        self.calibrated=False
        self.save_hyperparameters()
        

    def forward(self, x):
        x = self.feature(x)
        if self._pooling_method=='avg':
            x = F.adaptive_avg_pool2d(x,1)
        elif self._pooling_method=='max':
            x = F.adaptive_max_pool2d(x,1)
        x = x.flatten(1)
        out = self.classifier(x)
        return out

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def _forward_minibatch(self, minibatch):
        if isinstance(minibatch, List):
            img = minibatch[0].to(self.device)
        else:
            img = minibatch['img'].to(self.device)
        out = self(img)
        return out

    def _loss_metric_minibatch(self, out, minibatch, metrics):
        gt = minibatch['gt'].to(self.device)
        loss = F.cross_entropy(out,gt)
        prob = out.softmax(-1)
        for m in metrics:
            m.update(prob,gt)
        return loss

    def training_step(self, minibatch, batch_idx):
        _ = [m.to(self.device) for m in self.train_metrics]
        out = self._forward_minibatch(minibatch)
        loss = self._loss_metric_minibatch(out, minibatch, self.train_metrics)
        self.log('train_loss', loss, on_epoch=True, on_step=True)
        return dict(loss=loss)

    # def training_step_end(...)

    def training_epoch_end(self, var_list):
        metric_names = ['tacc','tauc','tckappa','tss','tppv', 'tsp']
        scores = [m.compute() for m in self.train_metrics]
        val_res = dict(zip(metric_names, scores))
        self.log_dict(val_res, on_step=False, on_epoch=True)
        _ = [m.reset() for m in self.train_metrics]

    def validation_step(self, minibatch, batch_idx):
        _ = [m.to(self.device) for m in self.val_metrics]
        out = self._forward_minibatch(minibatch)
        loss = self._loss_metric_minibatch(out, minibatch, self.val_metrics)
        self.log('val_loss', loss, on_epoch=True, on_step=True)
        return dict(val_loss=loss)

    # def validation_step_end(...)

    def validation_epoch_end(self, res_list):
        metric_names = ['acc','auc','ckappa','ss','ppv', 'sp']
        scores = [m.compute() for m in self.val_metrics]
        val_res = dict(zip(metric_names, scores))
        self.log_dict(val_res, sync_dist=True)
        _ = [m.reset() for m in self.val_metrics]

    def test_step(self, minibatch, batch_idx, test_idx):
        _ = [m.to(self.device) for m in self.test_metrics]
        out = self._forward_minibatch(minibatch)
        loss = self._loss_metric_minibatch(out, minibatch, self.test_metrics)

    # def test_step_end(...)

    def test_epoch_end(self,res_list):
        metric_names = ['acc','auc','ckappa','ss','ppv', 'sp']
        scores = [m.compute() for m in self.test_metrics]
        test_res = dict(zip(metric_names, scores))
        self.log_dict(test_res, sync_dist=True)
        _ = [m.reset() for m in self.test_metrics]
        return test_res

    def configure_optimizers(self):
        # opt = torch.optim.AdamW([{'params':self.feature.parameters(), 'lr':5e-5}, {'params':self.classifier.parameters(), 'lr':1e-4}], betas=(0.9,0.99), eps=1e-8, weight_decay=1e-4)
        opt = torch.optim.SGD([{'params':self.feature.parameters(), 'lr':1e-3}, {'params':self.classifier.parameters(), 'lr':1e-2}], weight_decay=1e-2)
        opt_sch = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt,5,2,1e-5)
        return [[opt], [opt_sch]]


    def calibration(self, valoader=None):
        valoader = valoader if valoader else self.trainer.datamodule.val_dataloader()
        model = ModelWithTemperature(self)
        model.set_temperature(valoader)
        self.temperature = model.temperature
        self.calibrated=True
        

    # def any_extra_hook(...)
