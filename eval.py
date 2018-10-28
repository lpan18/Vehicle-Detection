import numpy as np
import random
import torch
import sys
import os

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import common
from util import module_util
from bbox_helper import loc2bbox, nms_bbox, center2corner, generate_prior_bboxes, prior_layer_cfg
from bbox_loss import MultiboxLoss
from ssd_net import SSD


WILL_TEST = True
USE_GPU = False
IMG_MEAN = np.asarray((127, 127, 127))
IMG_STD = 128.0


def drawRectsWithImgPLT(img, rects, classes):
    '''
    function to draw image with the detected box and its label
    :param img: in PLT image format, 300 * 300
    :param rects: array represents the detected boxes
    :param classes: array of detected class indexes
    '''
    plt.imshow(img)
    colors = ['green', 'blue', 'red']  # negative, vehicle, human
    texts = ['background', 'car', 'person']

    for i in range(len(rects)):
        left = rects[i][0]
        top = rects[i][1]
        width = rects[i][2] - rects[i][0]
        height = rects[i][3] - rects[i][1]
        plt.gca().add_patch(patches.Rectangle((left, top), (width),(height),
                              fill=False, edgecolor=colors[classes[i]], linewidth=2))
        plt.text(left, top + height, texts[classes[i]],
                        horizontalalignment='left',
                        fontsize=20, color=colors[classes[i]],
                        verticalalignment='top')
    plt.show()



path_to_trained_model = 'ssd_net.pth'

img_file_path = sys.argv[1] # the index should be 1, 0 is the 'eval.py'
img = Image.open(img_file_path)
img_norm = (img - IMG_MEAN) / IMG_STD
img_np = np.asarray([img_norm], dtype="float32")
img_tensor = torch.from_numpy(img_np)
prior_bboxes = generate_prior_bboxes(prior_layer_cfg = prior_layer_cfg)

if WILL_TEST:

    if USE_GPU:
        test_net_state = torch.load(path_to_trained_model)
    else:
        test_net_state = torch.load(path_to_trained_model, map_location='cpu')
    test_net = SSD(num_classes=3)
    test_net.load_state_dict(test_net_state)
    test_net.eval()

    test_image_permuted = img_tensor.permute(0, 3, 1, 2)
    test_image_permuted = Variable(test_image_permuted.float())

    test_conf_preds, test_loc_preds = test_net.forward(test_image_permuted)
    test_bbox_priors = prior_bboxes.unsqueeze(0)
    test_bbox_preds = loc2bbox(test_loc_preds.cpu(), test_bbox_priors.cpu(), center_var=0.1, size_var=0.2)
    sel_bbox_preds = nms_bbox(test_bbox_preds.squeeze().detach(), test_conf_preds.squeeze().detach().cpu(), overlap_threshold=0.5, prob_threshold=0.5)

    rects = []
    classes = []

    for key in sel_bbox_preds.keys():
        for value in sel_bbox_preds[key]:
            classes.append(key)
            rects.append(value)
    rects = center2corner(torch.tensor(rects))*300

    if len(rects) != 0:
        if rects.dim() == 1:
            rects = rects.unsqueeze(0)
        classes = torch.tensor(classes)
        drawRectsWithImgPLT(img, rects, classes)
    else:
    	print("no object detected in this img")
