from .data_interface import DInterface
from utils.data.kera_seq_dataset import Keratitis_sequence,single_transform,test_transform,collate_fn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

class KeraSingle(DInterface):
    def __init__(self,  batch_size, num_workers, img_size, fold_id, cross_validation_fold=5):
        
        super().__init__(batch_size, num_workers)
        self.trainset_url = '/data/home/fangzhengqing/Data/Kera_ICML/train'
        self.testset_url = '/data/home/fangzhengqing/Data/Kera_ICML/test'
        self.srrpv_url = '/data/home/fangzhengqing/Data/Kera_ICML/valid'
        self.xs_url = '/data/home/fangzhengqing/Data/SRREX'
        self.pub_url = '/data/home/fangzhengqing/Data/Internet_kera'
        self.img_size = img_size
        self.cv_fold = cross_validation_fold
        self.fold_id = fold_id

    
    def setup(self, stage) -> None:
       # Assign train/val datasets for use in dataloaders
        fold_id = self.fold_id
        if stage == "fit":
            datadset = Keratitis_sequence(self.trainset_url, single_transform(self.img_size),cross_validation_fold= self.cv_fold)
            self.trainset, self.validset = datadset.cross_validation_split(fold_id)

        # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.testset = Keratitis_sequence(self.testset_url, test_transform(self.img_size))
            self.srrpv = Keratitis_sequence(self.srrpv_url, test_transform(self.img_size))
            self.pub = ImageFolder(self.pub_url,test_transform(self.img_size))
            self.xs = ImageFolder(self.xs_url,test_transform(self.img_size))
    
    def test_dataloader(self):
        pst_worker=False
        return [
            DataLoader(self.validset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=pst_worker),
            DataLoader(self.srrpv, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=pst_worker),
            DataLoader(self.pub, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=pst_worker),
            DataLoader(self.xs, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=pst_worker)
        ]

        # if stage == "predict":
        #     self.mnist_predict = MNIST(self.data_dir, train=False, transform=self.transform)
    