from torch.utils.data import Dataset
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
class MyDataset(Dataset):

    def __init__(self,image,list_dict,imgH=32,imgW=100):
        self.image = image
        self.list_dict= list_dict
        self.imgH = imgH
        self.imgW = imgW

    def __len__(self):
        return len(self.list_dict.keys())

    def __getitem__(self, idx):
        key = str(idx)
        line_bboxes = self.list_dict[key]
        images = []
        labels = []
        for box in line_bboxes:
            x1,y1,x2,y2 = box
            copy = self.image.copy()
            roi = copy[y1:y2,x1:x2]
            roi = Image.fromarray(roi)
            roi = roi.convert('L')
            images.append(roi)
            labels.append(idx)

        return images,labels

