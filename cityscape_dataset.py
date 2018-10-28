import numpy as np
import matplotlib.pyplot as plt
import random
import json
import os
import torch.nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import common
import bbox_helper

if common.USE_CLOUD:
    img_root='../../datasets/full_dataset/train_extra/'
else:
    img_root='./cityscapes_samples/'

def readJson(paths):
    return_list = []
    # vehicle=['car','truck','bus','on rails','motorcycle','bicycle','cargroup','truckgroup','busgroup','bicyclegroup']
    vehicle=['car','truck','bus','cargroup','truckgroup','busgroup']
    human=['person','rider','persongroup','ridergroup']
    for path in paths:
        # Get img path
        file_name=os.path.basename(path)
        tokens = file_name.split('_')
        foldername=tokens[0]
        image_name=tokens[0]+'_'+tokens[1]+'_'+tokens[2]+'_leftImg8bit.png'
        img_path=os.path.join(img_root,foldername,image_name)
        # Obtain ground_truth boundary box and label(class 1:vehicle, class 2:human, class 3:object)
        with open(path,'r') as f:
            frame_info=json.load(f)
            objects=frame_info['objects']
            label=[]
            bbox=[]
            for obj in objects:
                left_top,right_bottom=polygonToBox(obj['polygon'])
                if np.any(right_bottom-left_top>1024.0):
                    continue
                if obj['label'] in vehicle:
                    label.append(1.0)
                    bbox.append((left_top,right_bottom))
                elif obj['label'] in human:
                    label.append(2.0)
                    bbox.append((left_top,right_bottom))
            if len(bbox)!=0:
                return_list.append({
                    'img_path': img_path,
                    'label': label,
                    'bbox':bbox
                })
    return return_list

def polygonToBox(polygon):
    polygons=np.asarray(polygon, dtype=np.float32)
    left_top=np.min(polygons, axis=0)
    right_bottom=np.max(polygons,axis=0)
    return left_top, right_bottom

class CityScapeDataset(Dataset):

    def __init__(self, dataset_list):
        self.dataset_list = dataset_list

        # TODO: implement prior bounding box
        self.prior_bboxes = bbox_helper.generate_prior_bboxes(prior_layer_cfg=bbox_helper.prior_layer_cfg)

        # Pre-process parameters:
        self.mean = np.asarray((127, 127, 127))
        self.std = 128.0

    def get_prior_bbox(self):
        return self.prior_bboxes

    def __len__(self):
        return len(self.dataset_list)

    def __getitem__(self, idx):
        """
        Load the data from list, and match the ground-truth bounding boxes with prior bounding boxes.
        :return bbox_tensor: matched bounding box, dim: (num_priors, 4)
        :return bbox_label: matched classification label, dim: (num_priors)
        """

        # TODO: implement data loading
        # 1. Load image as well as the bounding box with its label
        item = self.dataset_list[idx]
        img = Image.open(item['img_path'])
        label=item['label']
        bbox=item['bbox']
        bbox_arr=np.array(bbox).reshape(-1,4)   # tuple to array

        # 2. Random crop to 1024*1024
        bbox_croped=[]
        label_croped=[]
        num_box_arr=len(bbox_arr)
        flag=False
        count=0
        while flag is False:
            count+=1
            crop_startX=random.uniform(0, 1)*1024
            crop_size=1024
            # if after 200 random, still not find a good crop position, then let crop pos = bbox pos
            if count == 200:
                crop_startX=bbox_arr[0][0]
                crop_size=bbox_arr[0][2]-bbox_arr[0][0]
                # print('bbox_arr 200',bbox_arr)
                # print('img_path',item['img_path'])
                # print('crop_startX',crop_startX)
                # print('crop_size',crop_size)
            for i in range(num_box_arr):
                if bbox_arr[i][2]>2048:  # bamberg_000000_000441_gtCoarse_polygons.json strange data
                    bbox_arr[i][2]=2048
                if bbox_arr[i][0]>=crop_startX and bbox_arr[i][2]<=crop_startX+crop_size:
                    flag=True
                    box=[bbox_arr[i][0]-crop_startX,bbox_arr[i][1],bbox_arr[i][2]-crop_startX,bbox_arr[i][3]]
                    bbox_croped.append(box)
                    label_croped.append(label[i])

        crop_pos= (crop_startX,0,crop_startX+crop_size,crop_size)
        img_croped = img.crop(crop_pos)
        resized_size= 300
        img_resized = img_croped.resize((resized_size, resized_size))
        # img_resized.save("img300.jpg", "JPEG")
        bbox_resized=np.divide(bbox_croped,crop_size/resized_size)

        # 3. Convert the bounding box from corner form (left-top, right-bottom): [(x,y), (x+w, y+h)] to
        #    center form: [(center_x, center_y, w, h)]
        bbox_center_form=bbox_helper.corner2center(torch.tensor(bbox_resized))

        # 4. Do the augmentation if needed. e.g. random clip the bounding box or flip the bounding box
        # Random flip
        will_flip = random.uniform(0, 1)
        if will_flip > 0.5:
            bbox_center_form[:,0] = resized_size - bbox_center_form[:,0]  # x coordinates after flip
            img_resized=img_resized.transpose(Image.FLIP_LEFT_RIGHT)
        # common.drawRectsPLT(img_resized,bbox_helper.center2corner(bbox_center_form),[int(i) for i in label_croped])

        # Normalize image
        img_normalized = (img_resized-self.mean)/self.std

        # 5. Normalize the bounding box position value from 0 to 1,
        sample_labels = torch.tensor(label_croped, dtype=torch.float32)
        sample_bboxes =torch.tensor(bbox_center_form/resized_size, dtype=torch.float32)

        sample_img =np.asarray(img_normalized, dtype=np.float32)
        img_tensor=torch.from_numpy(sample_img)

        # 6. Do the matching prior and generate ground-truth labels as well as the boxes
        bbox_tensor, bbox_label_tensor = bbox_helper.match_priors(self.prior_bboxes, sample_bboxes, sample_labels, iou_threshold=0.5)

        # [DEBUG] check the output.
        assert isinstance(bbox_label_tensor, torch.Tensor)
        assert isinstance(bbox_tensor, torch.Tensor)
        assert bbox_tensor.dim() == 2
        assert bbox_tensor.shape[1] == 4
        assert bbox_label_tensor.dim() == 1
        assert bbox_label_tensor.shape[0] == bbox_tensor.shape[0]

        return bbox_tensor, bbox_label_tensor,img_tensor

# test readJson
# print(len(readJson(common.getJsonList()[0:1])[0]['label']))
# print(len(readJson(common.getJsonList()[0:1])[0]['bbox']))
# print(common.getJsonList())
# readJson((common.getJsonList()))

# test plot img with bbox
# mean = np.asarray((127, 127, 127))
# std = 128.0
# item=readJson(common.getJsonList())
# img =np.asarray(Image.open(item[0]['img_path']),dtype=np.float32)/255.0
# bbox=item[0]['bbox']
# print('bbox',bbox)
# rects=np.array(bbox).reshape(-1,4)
# common.drawRectsPLT(img,rects,[2])

# test how many labels in the small dataset
# label=[]
# list=readJson(common.getJsonList())
# for i in range(len(list)):
#     label.extend(list[i]['label'])
# print(len(set(label)))   # 27 different labels in the small dataset

# test dataloader
# dataset_list=readJson(common.getJsonList())
# train_dataset = CityScapeDataset(dataset_list)
# train_data_loader = torch.utils.data.DataLoader(train_dataset,
#                                                 batch_size=1,
#                                                 shuffle=True,
#                                                 num_workers=0)
# idx, (offset_boxes, bbox_label,image) = next(enumerate(train_data_loader))
# rects=(bbox_helper.center2corner(train_dataset.get_prior_bbox().detach())).squeeze()*300
# texts=bbox_label.squeeze()
# image=(image.squeeze()*128.0).add(127.0).numpy()
# img = Image.fromarray(image.astype(np.uint8), 'RGB')
# for i in range(texts.shape[0]):
#     if texts[i] in [1,2]:
#         # print('rects[i,:]',rects[i,:])
#         # print('texts[i]',texts[i])
#         common.drawRectsPLT(img,rects[i,:].unsqueeze(0),texts[i].unsqueeze(0).int())
