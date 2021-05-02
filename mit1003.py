import os
import glob
import torch

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset
from utils import normalize_tensor

class MIT1003Data(Dataset):
    def __init__(self, root_path, image_path, fixation_path):
        super(MIT1003Data, self).__init__()
        self.image_path = root_path + image_path
        self.fixation_path = root_path + fixation_path
        self.images = glob.glob(os.path.join(self.image_path,'*.jpeg'))
        self.fiks = glob.glob(os.path.join(self.fixation_path,'*fixPts.jpg'))
        self.maps = glob.glob(os.path.join(self.fixation_path,'*fixMap.jpg'))

    def __getitem__(self, index):
        i = Image.open(self.images[index])
        f = Image.open(self.fiks[index]).convert('L')
        m = Image.open(self.maps[index]).convert('L')
        itransform = transforms.Compose([
                                transforms.Resize((256, 256), interpolation=Image.LANCZOS),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
        ftransform = transforms.Compose([
                                transforms.Resize((64, 64), interpolation=Image.NEAREST),
                                transforms.ToTensor(),
                                transforms.Lambda(lambda fix: torch.gt(fix, 0.5))
                            ])
        mtransform = transforms.Compose([
                                transforms.Resize((64, 64), interpolation=Image.LANCZOS),
                                transforms.ToTensor(),
                                transforms.Lambda(normalize_tensor)
                            ])
        i = itransform(i)
        f = ftransform(f)
        m = mtransform(m)
        return {'img': i, 'fix': f, 'map': m, 'name': self.images[index],
                'fname': self.fiks[index], 'mname': self.maps[index]}}

    def __len__(self):
        return len(self.images)