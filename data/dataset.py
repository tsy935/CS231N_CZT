import torch.utils.data as data
import torchvision as vision
import pandas as pd
from PIL import Image
from pathlib import Path
from constants.constants import NUM_CLASSES, MEAN, STD, CULTURE_LABELS, NUM_CROPS
import torch

class IMetDataset(data.Dataset):
    """
        Dataset defined for IMet data
        If train, resize all, crop only images with culture labels (5 crops), 
        plus some random horizontal flip;
        If evaluate, resize all, crop all (5 crops)
    """
    def __init__(self, 
                 root_dir: Path, 
                 csv_file=None,
                 mode='train'):
        self._root = root_dir      
        self.df = pd.read_csv(csv_file)
        self._img_id = (self.df['id'] + '.png').values
        self.mode = mode
        
        # if labels available
        if 'attribute_ids' in self.df:
            self.labels = self.df.attribute_ids.map(lambda x: x.split()).values
        else:
            self.labels = None
        
        self.transform_crp = None
        self.transform_rs = None
        if mode == 'train':
            self.preproc = 'resize'
            self.transform_rs = self.compose_transforms('resize')
            self.transform_normalize = self.compose_transforms('train_norm')
            # if there is a culture label
            if any(lab in self.labels for lab in CULTURE_LABELS):
                self.preproc = 'resize_crop'
                self.transform_crp = self.compose_transforms('crop')
        else:
            # if evaluate, resize and crop all images
            self.preproc = 'resize_crop'
            self.transform_crp = self.compose_transforms('crop')
            self.transform_rs = self.compose_transforms('resize')
            self.transform_normalize = self.compose_transforms('evaluate_norm')
            
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        
        # data augmentation
        if self.preproc == 'resize_crop': # resize and crop    
            img_crp = []
            for i_crp in range(NUM_CROPS):
                img_crp.append(self.transform_normalize(self.transform_crp(img)))
            img_rs = self.transform_normalize(self.transform_rs(img))
            img_tensor = torch.stack([img_crp, img_rs], dim = 0) # shape (6, C, H, W)
            print('Shape of img_tensor:{}'.format(img_tensor.size()))
        else: # resize only
            img_rs = self.transform_normalize(self.transform_rs(img))
            C, H, W = img_rs.size()
            img_tensor = img_rs.view(1, C, H, W) # shape (1, C, H, W)
            print('Shape of img_tensor:{}'.format(img_tensor.size()))

        
        if self.labels is not None: # if label is available
            label = self.labels[idx]
            if self.mode == 'train' and self.preproc == 'resize_crop':
                label_rs = torch.zeros((1, NUM_CLASSES))
                label_crp = torch.zeros((NUM_CROPS, NUM_CLASSES))
                for i in label:
                    label_rs[0, int(i)] = 1
                    # if a culture attribute label
                    if i in CULTURE_LABELS:
                        label_crp[:, int(i)] = 1
                
                label_tensor = torch.cat((label_rs, label_crp), dim=0) # shape (6, NUM_CLASSES)
            else: # resize only or evaluate mode
                label_tensor = torch.zeros((1, NUM_CLASSES)) # shape (1, NUM_CLASSES)
                for i in label:
                    label_tensor[0, int(i)] = 1                                      
        else: # if label not available
            label_tensor = None
            
        example = (img_tensor,
                   label_tensor,
                   img_id,
                   self.preproc)
        
        return example
        
    def compose_transforms(self, transform_method):
        """
            For training data, resize to 224x224, random horizontal flip, normalize;
            For dev/test data, resize to 224x224, normalize.
        """
        transforms = {'resize': vision.transforms.Compose([
                                vision.transforms.Resie(224),
                                vision.transforms.ToTensor()]),
                      'crop': vision.transforms.Compose([
                              vision.transforms.RandomCrop(224),
                              vision.transforms.ToTensor()]),
                      'train_norm': vision.transforms.Compose([
                                  vision.transforms.RandomHorizontalFlip(),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ]),
                      'evaluate_norm': vision.transforms.Compose([
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ])}
        
        return transforms[transform_method]
            
        
        