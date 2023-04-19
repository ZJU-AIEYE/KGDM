from report import MODEL_CLASS,get_args,KeraSingle,Trainer
import torch
import pytorch_lightning as pl
from main import get_args
import os
import torch.nn.functional as F
# from tsnecuda import TSNE
from typing import List
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np


METHOD={'basemethod':'ResNet50','protopnet':'ProtoPNet', 'pedl':'KGDM', 'edl':'KGDM'}
DATASET=['SRRT','SRRT','SRRPV','PUB','XSH']
def main(args):
    # model
    root_dir =  args.default_root_dir
    rerun_data = True
    

    if rerun_data:
        fold_id = 0
        plt.figure(figsize=(8,4))

        for method in ['basemethod','edl']:
            pl.seed_everything(args.seed)
            args.method = method
            args.fold_id=fold_id
            args.default_root_dir = root_dir+f'/{args.backbone}/{args.method}/fold_{fold_id}'
            # load data 
        
            data = KeraSingle(args.batch_size, args.num_workers, args.img_size, fold_id, 5)
            data.setup("fit")
            data.setup("test")
            ckpt_path = args.default_root_dir+'/lightning_logs/version_0/checkpoints/last.ckpt'
            print(ckpt_path)
            model = MODEL_CLASS[args.method].load_from_checkpoint(ckpt_path)
            model.cuda()
            model.eval()
            for i,dataloader in enumerate([data.train_dataloader()]+data.test_dataloader()):
                if i!=2:
                    continue
                features = []
                gts = []
                us = []
                for batch in dataloader:
                    if isinstance(batch, List):
                        img = batch[0].to(model.device)
                        gt = batch[1]
                    else:
                        img = batch['img'].to(model.device)
                        gt = batch['gt']
                    
                    with torch.no_grad():
                        out = model(img)
                        if args.method in ['pedl']:
                            alpha = out['pdl_out'].exp()
                        elif args.method=='edl':
                            alpha = out.relu()+1
                        elif args.method=='basemethod':
                            alpha = out.exp()
                        # fmap=model.feature(img)
                        # selected = gt>=model.num_classes
                        
                        # f = F.adaptive_max_pool2d(fmap,1).flatten(1)[selected]
                        # features.append(f)
                        # gts.append(gt[selected])
                        # logit = model.classifier(f).relu()+1
                        logit = alpha
                        prob = logit/logit.sum(-1,keepdim=True)
                        u = -(prob*prob.log()).sum(-1)/torch.tensor(model.num_classes).log()
                        
                        us.append(u)
                us = torch.cat(us,0)
                print(us.mean())
                print(us.max())

                ts = torch.linspace(0,1.05,21)
                cds = []
                for t in ts:
                    cds.append(len(us[us<t])/len(us))
                # cds = np.concatenate(cds,0)

                # ax = plt.subplot(1,1,1)
                plt.title(f"CDF curve of not-IK disease in {DATASET[i]}")
                # ax.axis('off')
                # sns.lineplot(ts,cds,hue=gts,ax=ax,palette='bright')
                plt.plot(ts,cds,label=f'{METHOD[args.method]}')
            # handles, labels  =  ax.get_legend_handles_labels()
            # ax.legend(handles, ['AK', 'BK', 'FK', 'HSK'])   
            plt.legend()
            plt.ylabel('probability')
            plt.xlabel('% of max entropy')
            plt.xticks([0,0.2,0.4,0.6,0.8,1.0],['0%','20%','40%','60%','80%','100%'])
            plt.yticks([0,0.2,0.4,0.6,0.8,1.0],['0%','20%','40%','60%','80%','100%'])
            os.makedirs(f"test_result/{args.backbone}/", exist_ok=True)
        plt.savefig(f"test_result/{args.backbone}/cdf.pdf")
       

if __name__=='__main__':
    torch.set_printoptions(precision=10, sci_mode=None)
    # load args
    args = get_args()
    pl.seed_everything(args.seed)
    main(args)
    pass