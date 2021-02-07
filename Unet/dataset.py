import os
import numpy as np 

from PIL import Image

from torchvision.datasets.folder import *
from torch.utils.data import DataLoader

class CasiaWebFace(ImageFolder):
    def __init__(self, mask_root, label_root, transform=None, 
                 loader=default_loader, is_valid_file=None):

        super(CasiaWebFace, self).__init__(mask_root, loader=loader, 
                                       transform=transform,
                                       is_valid_file=is_valid_file)
        self.imgs = self.samples
        self.label_root = label_root

    def __getitem__(self, index):
            """
            Args:
                index (int): Index

            Returns:
                tuple: (sample, target) where target is class_index of the target class.
            """
            # Load mask image 
            path, target = self.samples[index]
            sample = self.loader(path)#.convert('L')
            #print(np.array(sample))
            #sample.show()

            # Load label image 
            strings = path.split(os.sep)
            image_name = strings[-1].split(".")[0]
            class_name = strings[-2]
           
            label_path = os.path.join(self.label_root, class_name, image_name + ".jpg")
            sample_label = self.loader(label_path)
            #print(np.array(sample_label))
            #sample_label.show()
            
            if self.transform is not None:
                sample = self.transform(sample)
                sample_label = self.transform(sample_label)
 
            return sample, sample_label