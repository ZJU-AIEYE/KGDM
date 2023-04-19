from .protomodel import Protomodel
import numpy as np
import cv2
import heapq
import torch
from typing import List
from types import MethodType
import torch.nn.functional as F

tensor2cpu = lambda x: x.detach().cpu()
tensor2py = lambda x: x.detach().cpu().numpy()

class VisualConcept():
    def __init__(self, img:torch.Tensor, img_gt:torch.Tensor, concept_vector:torch.Tensor, mask:torch.Tensor) -> None:
        super().__init__()
        self.img = tensor2cpu(img)
        self.img_gt = img_gt
        self.cv = concept_vector
        self.mask = tensor2cpu(mask)
        self.proto = None

    def set_prototype(self, p):
        self.proto = p

    # def get_masked_img(self):
    #     return self.img * self.mask

    def dist(self, other):
        cv = other.cv if isinstance(other, VisualConcept) else other
        if isinstance(other, torch.Tensor):
            sim = F.cosine_similarity(self.cv[None,:], cv[None,:])[0]
            return tensor2py(sim)
            dist = -sim+1
            return torch.log((dist+1)/(dist+1e-3))-torch.log(torch.tensor(2))
        else:
            sim = np.dot(self.cv, cv)/(np.linalg.norm(self.cv)*np.linalg.norm(cv))
            return sim
            # dist = -sim+1
            # return np.log((dist+1)/(dist+0.001))-np.log(2)

    def self_dist(self):
        if self.proto is None:
            return -1
        return self.dist(self.proto)

    def _visualize_(self, heat=False, vertical=True, hmfirst=True, cont=True):
        # print(type(originimage),type(heatmap))
        originimage = np.uint8(255*tensor2cpu(self.img).permute(1,2,0).flip(2))
        heatmap = np.uint8(255*tensor2py(self.mask))
        heatmap = cv2.resize(heatmap, [self.img.shape[1],self.img.shape[2]], interpolation=cv2.INTER_CUBIC)
        # contour
        thresh = np.ones_like(heatmap)
        ret = np.percentile(heatmap.flatten(),80)
        # ret=192
        thresh[heatmap < ret] = 0
        thresh[heatmap > ret] = 255
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        #colormap
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        if heat:
            newimage = cv2.addWeighted(heatmap, 0.3, originimage, 0.7, 0)
        else:
            newimage = cv2.addWeighted(heatmap, 0, originimage, 1, 0)

        if cont:
            newimage = cv2.drawContours(newimage, contours, -1, (0, 255, 255), 2)

        # cv2.putText(newimage, f"l{self.img_gt}_d{self.self_dist():.3f}", (0, 25), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        cropimage = originimage*(thresh==255)[:,:,None]
        cropimage[cropimage==0]=128
        
        # out rectangle
        coor = []
        for cidx,cnt in enumerate(contours):
            x0, y0, w, h = cv2.boundingRect(cnt)
            crop_size = max(w, h)
            coor.append(list((x0,y0, crop_size)))
        coors = np.array(coor)
        try:
            results =  list(coors[np.argmax(coors[:,2], axis=0), :])  # 返回面积最大的那一组坐标
            x0=results[0]
            y0 = results[1]
            size =  results[2] 
            cropimage = cropimage[y0:y0+size,x0:x0+size]
            cropimage = cv2.resize(cropimage, [self.img.shape[1],self.img.shape[2]], interpolation=cv2.INTER_CUBIC)
        except:
            pass

        imgs_show = [newimage,cropimage] if hmfirst else [cropimage,newimage]
        return np.concatenate(imgs_show,0) if vertical else  np.concatenate(imgs_show,1)

    def __gt__(self, other):
        return self.self_dist() > other.self_dist()

    def __lt__(self, other):
        return self.self_dist() < other.self_dist()

    def __eq__(self, other):
        return self.self_dist() == other.self_dist()


class Prototype():
    def __init__(self, name, gt, fdim, vec, visual_concepts: List[VisualConcept] = [], num_example=10) -> None:
        self.name = name
        self.fdim = fdim
        self.vec = vec
        self.example_visual_concepts = []
        self.num_example = num_example
        self.gt = gt
        for vc in visual_concepts:
            self.add_visual_concept(vc)

    def add_visual_concept(self, vc: VisualConcept):
        if vc.img_gt != self.gt:
            return
        vc.set_prototype(self.vec)
        # self.example_visual_concepts.append(vc)
        if vc.self_dist()>0 or len(self.example_visual_concepts)==0:
            self.example_visual_concepts.append(vc)
        # keep nearst
        if len(self.example_visual_concepts)==1 and vc.self_dist() > self.example_visual_concepts[0].self_dist():
            self.example_visual_concepts[0]=vc
        # heapq.heappush(self.example_visual_concepts, vc)
        # print(vc.self_dist())
    
    def reserve_n(self):
        if len(self.example_visual_concepts) > self.num_example:
            self.example_visual_concepts = heapq.nlargest(self.num_example, self.example_visual_concepts)
            heapq.heapify(self.example_visual_concepts)

    def _visualize_(self, num = None):
        # fig = plt.figure()
        num  = self.num_example if num is None else num
        vclist = heapq.nlargest(num, self.example_visual_concepts)
        imgs = [v._visualize_() for v in vclist]
        return np.concatenate(imgs, 1)

    def _nearst_(self, vc, num):
        dists = [evc.dist(vc) for evc in self.example_visual_concepts]
        index = np.argsort(dists)[::-1]
        if num==1:
            return self.example_visual_concepts[index[0]]
        # else:
        #     for i in index[:num]:
        #         yield (self.example_visual_concepts[i],dists[i])
    
    def _visualize_nearst_(self, vc, num=5):
        vclist:List[VisualConcept] = self._nearst_(vc,num)
        imgs = [v._visualize_() for v in vclist]
        return np.concatenate(imgs, 1)

    def visual_concepts(self):
        return heapq.nlargest(self.num_example, self.example_visual_concepts)

    def visual_concept(self):
        return heapq.nlargest(1, self.example_visual_concepts)[0]

    def visual_concept_vec(self):
        return heapq.nlargest(1, self.example_visual_concepts)[0].cv if len(self.example_visual_concepts)>0 else self.vec

    def __purity__(self):
        vcs = self.visual_concepts()
        gts = [vc.img_gt for vc in vcs]
        counts = np.bincount(gts)
        if len(counts) == 0:
            return -1
        return np.max(counts) / len(gts)

    def __metainfo__(self):
        res = {}
        for k,v in self.metainfo.items():
            if isinstance(v,np.ndarray):
                v = v.tolist()
            res[k]=v
        return res

    def __repr__(self) -> str:
        return f"Prototype: {self.name}, {len(self.example_visual_concepts)} VCs"

def addproto(self:Protomodel, resdict,img,gt):
    fmap_size = self.protolayer.fmap_size
    num_class = self.num_classes
    num_proto = self.protolayer.num_proto
    fs = resdict['maxfs']
    view = (resdict['cosvalue']).unflatten(-1,(fmap_size[0],fmap_size[1]))
    view = view.clip_(min=0)
    for i in range(img.shape[0]):
        # maxproto = resdict['cosvalue'][i].max(-1)[0].argmax()
        # nclass = maxproto//num_proto
        # nproto = maxproto%num_proto
        for nclass in range(num_class):
            for nproto in range(num_proto):
                vec = fs[i,nclass,nproto]
                vc = VisualConcept(img[i],gt[i],vec,view[i,nclass,nproto])
                # print(sim[nclass*num_proto+nproto])
                self.prototypes[nclass][nproto].add_visual_concept(vc)

def _empty_proto(self:Protomodel):
    prototypes=[]
    num_proto = self.protolayer.num_proto
    protos = self.protolayer.prototype.mT[0]
    protos = F.normalize(protos,2,-1)
    for nclass in range(self.num_classes):
        ip = []
        for nproto in range(num_proto):
            p = Prototype(f"c{nclass}_p{nproto}", nclass, self.protolayer.f_dim, protos[nclass*num_proto+nproto],num_example=20)
            ip.append(p)
        prototypes.append(ip)
    return prototypes

def construct_prototypes(self:Protomodel, traindataloader):
    self.prototypes=self._empty_proto()
    self.eval()
    with torch.no_grad():
        for i,batch in enumerate(traindataloader):
            # print(idx,end='\r')
            resdict = self._forward_minibatch(batch)
            img,gt = batch['img'], batch['gt']
            self.addproto(resdict, img,gt)
            print(f'{i}/{len(traindataloader)}',end='\r')

    for nclass in range(self.num_classes):
        for nproto in range(self.protolayer.num_proto):
            self.prototypes[nclass][nproto].reserve_n()
            

def interpret(self:Protomodel, x, visual_num=3):
    self.eval()
    fmap_size = self.protolayer.fmap_size
    num_proto = self.protolayer.num_proto
    res=[]
    with torch.no_grad():
        out = self.forward(x)
        
        weight = self.protolayer.weight*self.protolayer.weight_e.sigmoid()
        weight = weight.flatten()
        
        for i in range(x.shape[0]):
            cosvalue = out['cosvalue'][i].flatten(0,1)
            contrib = out['contrib'][i].flatten(0,1)
            fs = out['maxfs'][i].flatten(0,1)
            score, pids = contrib.sort(descending=True)
            pids=pids[:visual_num]
            score=score[:visual_num]
            mask = cosvalue[pids].unflatten(-1,(fmap_size[0], fmap_size[1])).clip_(min=0)

            cos_value = F.adaptive_max_pool2d(mask,1).flatten()
            w = weight[pids]
            vcs=[VisualConcept(x[i],-1,fs[p],mask[idx]) for idx,p in enumerate(pids)]
            protos = [self.prototypes[p//num_proto][p%num_proto] for idx,p in enumerate(pids)]
            tvcs = []
            for vc, p in zip(vcs,protos):
                mask = F.cosine_similarity(out['f'][i], p.vec[None,:]) 
                vc.mask = mask.unflatten(0, fmap_size).clip_(min=0)
                # vc.mask=None
                tvcs.append(vc)
            resi = dict(pids=pids, cosvalue=cos_value,score=score,weight=w,vcs=tvcs, protos = protos)
            res.append(resi)
    return res


def Explainer(model:Protomodel):
    model.construct_prototypes = MethodType(construct_prototypes, model)
    model.interpret = MethodType(interpret, model)
    model._empty_proto = MethodType(_empty_proto, model)
    model.addproto = MethodType(addproto, model)
    return model
