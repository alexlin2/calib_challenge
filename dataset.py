from numpy import nan
from numpy.core.numeric import NaN
from torch.utils import data
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import glob
import torch
from PIL import Image
from decimal import Decimal

img_dir = "/home/alexlin/calib_challenge/images_labeled"
labels_dir = "/home/alexlin/calib_challenge/labeled"

class CalibrationDataset(Dataset):
    
    def __init__(self, img_dir, labels_dir, long_size=512):
        self.img_dir = img_dir
        self.labels_dir = labels_dir
        self.image_path, self.pitch_yaw_labels = self.get_data()
        self.long_size = long_size

    def __len__(self):
        assert len(self.image_path) == len(self.pitch_yaw_labels)
        return len(self.image_path)

    def __getitem__(self, index):
        img = Image.open(self.image_path[index]).convert('RGB')
        w, h = img.size
        re_w = self.long_size
        re_h = int(h*re_w/w)
        re_size = (re_w, re_h)
        img = img.resize(re_size)
        x = F.to_tensor(img)
        y = torch.tensor(self.pitch_yaw_labels[index], dtype = torch.double)
        return x, y

    def get_data(self):
        labels = []
        img_path = []
 
        for i, name in enumerate(sorted(glob.glob(self.labels_dir + "/*.txt"))):
            with open(name) as f:
                lines = f.readlines()
                imgs = sorted(glob.glob(self.img_dir + "/frame%d*.png" % i))
                for index, line in enumerate(lines):
                    if line.split()[0] == "nan":
                        continue
                    pitch_yaw = [float(t) / 0.05 for t in line.split()]
                    labels.append(pitch_yaw)
                    img_path.append(imgs[index])
        return img_path, labels
        


if __name__ == "__main__":
    dataset = CalibrationDataset(img_dir=img_dir, labels_dir=labels_dir)
    torch.set_printoptions(precision=10)
    print(dataset[0][1])
