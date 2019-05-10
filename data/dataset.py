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
                    
        # define transformations
        self.transform_train_rs = self.compose_transforms('train_resize')
        self.transform_train_crp = self.compose_transforms('train_crop')
        self.transform_eval_rs = self.compose_transforms('evaluate_resize')
        self.transform_eval_crp = self.compose_transforms('evaluate_crop')       
            
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        
        # define preprocessing/augmentation method
        if self.mode == 'train' and label is not None:
            preproc = 'resize'
            # if there is a culture label                
            for lab in label:
                if int(lab) in CULTURE_LABELS:
                    preproc = 'resize_crop'
                    break
                             
        else:
            # if evaluate, resize and crop all images
            preproc = 'resize_crop'
        
        #print(preproc)
        
        # data augmentation
        if preproc == 'resize_crop': # resize and crop
            if self.mode == 'train': # train, augmentation includes random horizontal flip
                imgs = []
                imgs.append(self.transform_train_rs(img))
                for i_crp in range(NUM_CROPS):
                    imgs.append(self.transform_train_crp(img))
                img_tensor = torch.stack(imgs, dim = 0) # shape (6, C, H, W)
            else: # evaluate, augmentation does not include random transforms
                imgs = []
                imgs.append(self.transform_eval_rs(img))
                for i_crp in range(NUM_CROPS):
                    imgs.append(self.transform_eval_crp(img))
                img_tensor = torch.stack(imgs, dim = 0) # shape (6, C, H, W)
        else: # resize only
            if self.mode == 'train':
                img_rs = []
                for i_rs in range(NUM_CROPS+1):
                    img_rs.append(self.transform_train_rs(img))
                img_tensor = torch.stack(img_rs, dim=0) # shape (6, C, H, W)
            else:
                img_rs = self.transform_eval_rs(img)
                C, H, W = img_rs.size()
                img_tensor = img_rs.view(1, C, H, W) # shape (1, C, H, W)
                
        # get label
        if label is not None: # if label is available
            if self.mode == 'train' and preproc == 'resize_crop':
                label_rs = torch.zeros((1, NUM_CLASSES))
                label_crp = torch.zeros((NUM_CROPS, NUM_CLASSES))
                for i in label:
                    label_rs[0, int(i)] = 1
                    # if a culture attribute label
                    if i in CULTURE_LABELS:
                        label_crp[:, int(i)] = 1
                
                label_tensor = torch.cat((label_rs, label_crp), dim=0) # shape (6, NUM_CLASSES)
            else: # resize only or evaluate mode
                label_tensor = torch.zeros((NUM_CROPS+1, NUM_CLASSES)) # shape (6, NUM_CLASSES)
                for i in label:
                    label_tensor[:, int(i)] = 1
            
            #pos_weights = self.compute_pos_weights(label_tensor)
        else: # if label not available
            label_tensor = {}
            #pos_weights = None
        
        
        #print('Shape of label_tensor:{}'.format(label_tensor.size()))
        hogs_feature = {}
        example = (img_tensor,
                   label_tensor,
                   img_id,
                   preproc,
                   hogs_feature)
        
        return example
        
    def compose_transforms(self, transform_method):
        """
            For training data, resize/crop to 224x224, random horizontal flip, random color jitter, normalize;
            For dev/test data, resize/crop to 224x224, normalize.
        """
        transforms = {'train_resize': vision.transforms.Compose([
                                  vision.transforms.Resize((224,224)),
                                  vision.transforms.RandomHorizontalFlip(),
                                  vision.transforms.ColorJitter(hue=.05, saturation=.05),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ]),
                      'train_crop': vision.transforms.Compose([
                                  vision.transforms.RandomCrop(224),
                                  vision.transforms.RandomHorizontalFlip(),
                                  vision.transforms.ColorJitter(hue=.05, saturation=.05),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ]),
                      'evaluate_resize': vision.transforms.Compose([
                                  vision.transforms.Resize((224,224)),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ]),
                      'evaluate_crop': vision.transforms.Compose([
                                  vision.transforms.RandomCrop(224),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ])}
        
        return transforms[transform_method]
    
    def compute_pos_weights(self, label_tensor):
        """
        Compute pos_weight for each class in the batch, pos_weight = (# negative samples) / (# positive samples)
        Args:
            label_tensor: Tensor of one-hot encoded labels, shape (batch_size, num_classes)
        """
        label = label_tensor.numpy()
        batch_size = label.shape[0]
        frequencies = np.sum(label, axis=0)
        
        pos_weights = np.ones((1, NUM_CLASSES))
        indices = frequencies != 0.
        pos_weights[indices] = np.divide(batch_size - frequencies[indices], frequencies[indices])
        print(pos_weights)
        return pos_weights
                   

class IMetDatasetBase(data.Dataset):
    """
        Dataset defined for IMet data, all images are resized, no cropping
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
                    
        # define transformations
        self.transform_train_rs = self.compose_transforms('train_resize')
        self.transform_eval_rs = self.compose_transforms('evaluate_resize')
            
        
    def __len__(self):
        return len(self._img_id)
    
    def __getitem__(self, idx):
        img_id = self._img_id[idx]
        file_name = self._root / img_id
        img = Image.open(file_name)
        if self.labels is not None:
            label = self.labels[idx]
        else:
            label = None
        
        # data augmentation
        if self.mode == 'train':
            img_tensor = self.transform_train_rs(img) # shape (C, H, W)
        else:
            img_tensor = self.transform_eval_rs(img) # shape (C, H, W)
           
        # get label
        if label is not None: # if label is available
            label_tensor = torch.zeros((NUM_CLASSES))
            for i in label:
                label_tensor[int(i)] = 1                                                    
        else: # if label not available
            label_tensor = None
        
        #print('Shape of img_tensor:{}'.format(img_tensor.size()))
        #print('Shape of label_tensor:{}'.format(label_tensor.size()))
        example = (img_tensor,
                   label_tensor,
                   img_id,
                   'resize')
        
        return example
        
    def compose_transforms(self, transform_method):
        """
            For training data, resize/crop to 224x224, random horizontal flip, random color jitter, normalize;
            For dev/test data, resize/crop to 224x224, normalize.
        """
        transforms = {'train_resize': vision.transforms.Compose([
                                  vision.transforms.RandomResizedCrop(224),
                                  vision.transforms.RandomHorizontalFlip(),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ]),
                      'evaluate_resize': vision.transforms.Compose([
                                  vision.transforms.Resize(256),
                                  vision.transforms.CenterCrop(224),
                                  vision.transforms.ToTensor(),
                                  vision.transforms.Normalize(mean=MEAN,std=STD)
                                  ])}
        
        return transforms[transform_method]


        
        