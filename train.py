import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable

import common
from util import module_util
from bbox_helper import loc2bbox, nms_bbox, center2corner
from bbox_loss import MultiboxLoss
from ssd_net import SSD
from cityscape_dataset import readJson, CityScapeDataset



WILL_TRAIN = False
WILL_TEST = True
batch_size = 12
num_workers = 0



if common.USE_CLOUD:
    batch_size=32
    num_workers=0


# Read all json file into data_list and randomly shuffle data
data_list = readJson(common.getJsonList()[0:10])
total_items = len(data_list)

# Divide data into train, validate and test lists
n_train_sets = 0.6 * total_items
train_set_list = data_list[: int(n_train_sets)]
n_valid_sets = 0.3 * total_items
valid_set_list = data_list[int(n_train_sets): int(n_train_sets + n_valid_sets)]
test_set_list = data_list[int(n_train_sets + n_valid_sets):]

# Load train and validate data into dataloader
train_dataset = CityScapeDataset(train_set_list)
train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
print('Total training items', len(train_dataset),', Total training mini-batches in one epoch:', len(train_data_loader))
valid_set = CityScapeDataset(valid_set_list)
valid_data_loader = DataLoader(valid_set, batch_size=int(batch_size/2), shuffle=True, num_workers=num_workers)
print('Total validation set:', len(valid_set))
test_set = CityScapeDataset(test_set_list)
test_data_loader = DataLoader(test_set, batch_size=1, shuffle=True, num_workers=num_workers)
print('Total test set:', len(test_set))



if WILL_TRAIN:
    # Print key info of tensor inputs
    idx, (loc_targets, conf_targets, image) = next(enumerate(train_data_loader))
    print('loc_targets tensor shape:', loc_targets.shape)
    print('conf_targets tensor shape:', conf_targets.shape)
    print('image tensor shape:', image.shape)
    # Create the instance of our defined network
    net = SSD(num_classes=3)
    net.cuda()
    criterion = MultiboxLoss([0.1, 0.1, 0.2, 0.2],iou_threshold=0.5, neg_pos_ratio=3.0)
    # optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr = 0.001)
    train_losses = []
    valid_losses = []
    max_epochs = 50
    itr = 0

    # Train process
    for epoch_idx in range(0, max_epochs):
        for train_batch_idx, (train_loc_targets, train_conf_targets, train_image) in enumerate(train_data_loader):
            itr += 1
            net.train()
            train_loc_targets = Variable(train_loc_targets.cuda().float())
            train_conf_targets = Variable(train_conf_targets.cuda().long())
            train_image = train_image.permute(0, 3, 1, 2)
            train_image = Variable(train_image.cuda().float())
            optimizer.zero_grad()
            train_conf_preds, train_loc_preds = net.forward(train_image)
            train_conf_loss, train_loc_loss = criterion.forward(train_conf_preds, train_loc_preds, train_conf_targets, train_loc_targets)
            train_total_loss = train_conf_loss + train_loc_loss
            train_total_loss.backward()
            optimizer.step()
            train_losses.append((itr, train_total_loss))
            print('Train Epoch: %d Itr: %d Conf Loss: %f Loc Loss: %f Total Loss: %f' %
                  (epoch_idx, itr, train_conf_loss.item(), train_loc_loss.item(), train_total_loss.item()))

            # Validation process:
            if train_batch_idx % 5 == 0:
                net.eval()
                valid_conf_loss_set = []
                valid_loc_loss_set = []
                valid_itr = 0
                for valid_batch_idx, (valid_loc_targets, valid_conf_targets, valid_image) in enumerate(valid_data_loader):
                    valid_loc_targets = Variable(valid_loc_targets.cuda().float())
                    valid_conf_targets = Variable(valid_conf_targets.cuda().long())
                    valid_image = valid_image.permute(0, 3, 1, 2)
                    valid_image = Variable(valid_image.cuda().float())
                    valid_conf_preds, valid_loc_preds = net.forward(valid_image)
                    valid_conf_loss, valid_loc_loss = criterion.forward(valid_conf_preds, valid_loc_preds, valid_conf_targets, valid_loc_targets)
                    valid_conf_loss_set.append(valid_conf_loss.item())
                    valid_loc_loss_set.append(valid_loc_loss.item())
                    valid_itr += 1
                    if valid_itr > 3:
                        break
                avg_valid_conf_loss = np.mean(np.asarray(valid_conf_loss_set))
                avg_valid_loc_loss = np.mean(np.asarray(valid_loc_loss_set))
                avg_valid_loss = avg_valid_conf_loss + avg_valid_loc_loss
                print('Valid Epoch: %d Itr: %d Conf Loss: %f Loc Loss: %f Total Loss: %f' % (
                    epoch_idx, itr, avg_valid_conf_loss.item(), avg_valid_loc_loss.item(), avg_valid_loss.item()))
                valid_losses.append((itr, avg_valid_loss))
        # Save model to disk for every epoch
        net_state = net.state_dict()
        torch.save(net_state, 'ssd_net_'+str(epoch_idx)+'.pth')
    train_losses = np.asarray(train_losses)
    valid_losses = np.asarray(valid_losses)


    # Plot loss
    plt.plot(train_losses[:, 0],      # iteration
             train_losses[:, 1])      # loss value
    plt.plot(valid_losses[:, 0],      # iteration
             valid_losses[:, 1])      # loss value
    plt.xlabel('Iteration')
    plt.ylabel('Losses')
    plt.legend(['Train Losses', 'Valid Losses'])
    plt.show()



# Test process
if WILL_TEST:
    test_net = SSD(num_classes=3)
    test_net.cuda()
    test_net_state = torch.load('ssd_net.pth')
    test_net.load_state_dict(test_net_state)
    test_net.eval()
    for test_batch_idx, (test_loc_targets, test_conf_targets, test_image) in enumerate(test_data_loader):
    # test_batch_idx, (test_loc_targets, test_conf_targets, test_image) = next(enumerate(test_data_loader))
        test_image_permuted = test_image.permute(0, 3, 1, 2)
        test_image_permuted = Variable(test_image_permuted.cuda().float())
        test_conf_preds, test_loc_preds = test_net.forward(test_image_permuted)
        # uset CityScapeDataset function get_prior_bbox
        test_bbox_priors=test_set.get_prior_bbox().unsqueeze(0)
        test_bbox_preds=loc2bbox(test_loc_preds.cpu(), test_bbox_priors.cpu(), center_var=0.1, size_var=0.2)
        sel_bbox_preds=nms_bbox(test_bbox_preds.squeeze().detach(), test_conf_preds.squeeze().detach().cpu(), overlap_threshold=0.5, prob_threshold=0.5)
        rects=[]
        texts=[]
        for key in sel_bbox_preds.keys():
            for value in sel_bbox_preds[key]:
                texts.append(key)
                rects.append(value)
        rects=center2corner(torch.tensor(rects))*300
        if len(rects)!=0:
            if rects.dim()==1:
                rects=rects.unsqueeze(0)
            texts=torch.tensor(texts)
            test_image=(test_image.squeeze()*128.0).add(127.0).numpy()
            img = Image.fromarray(test_image.astype(np.uint8), 'RGB')
            common.drawRectsPLT(img,rects,texts)