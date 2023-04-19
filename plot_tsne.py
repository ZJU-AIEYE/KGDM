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


METHOD={'basemethod':'ResNet50','protopnet':'ProtoPNet', 'pedl':'KGDM', 'edl':'EDL'}
DATASET=['SRRT','SRRT','SRRPV','PUB','XSH']
def main(args):
    # model
    root_dir =  args.default_root_dir
    rerun_data = True
    if rerun_data:
        fold_id = 0
        pl.seed_everything(args.seed)
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
        plt.figure(figsize=(24,4))
        for i,dataloader in enumerate([data.train_dataloader()]+data.test_dataloader()):
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
                    fmap=model.feature(img)
                    selected = gt<model.num_classes
                    f = F.adaptive_max_pool2d(fmap,1).flatten(1)[selected]
                    features.append(f)
                    gts.append(gt[selected])
                    logit = model.classifier(f).relu()+1
                    u = model.num_classes/logit.sum(-1)
                    us.append(u)
            us = torch.cat(us,0)
            print(us.mean())
            features=torch.cat(features,0).cpu().numpy()
            gts=torch.cat(gts,0).cpu().numpy()
            early_exaggeration = 6.0
            lr=max(gts.shape[0]/early_exaggeration, 50)
            pca = PCA(128)
            tsne = TSNE(2, early_exaggeration=24)
            # tsne = TSNE(n_iter=5000,  n_iter_without_progress=250,early_exaggeration=early_exaggeration,verbose=1, perplexity=30, num_neighbors=64, learning_rate=lr)
            # tsne_results = pca.fit_transform(features)
            tsne_results = tsne.fit_transform(features)
            print(tsne_results.shape)
            print(gts.shape)
            ax = plt.subplot(1,5,i+1)
            ax.set_title(f"t-SNE visulization of {METHOD[args.method]} on {DATASET[i]}")
            ax.axis('off')
            sns.scatterplot(tsne_results[:,0],tsne_results[:,1],hue=gts,ax=ax,palette='bright')
            handles, labels  =  ax.get_legend_handles_labels()
            ax.legend(handles, ['AK', 'BK', 'FK', 'HSK'])   
            os.makedirs(f"test_result/{args.backbone}/", exist_ok=True)
        plt.savefig(f"test_result/{args.backbone}/{args.method}.pdf")
       

if __name__=='__main__':
    torch.set_printoptions(precision=10, sci_mode=None)
    # load args
    args = get_args()
    pl.seed_everything(args.seed)
    main(args)
    pass