import os
import glob
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

USE_CLOUD = False
if USE_CLOUD:
    label_root='../../datasets/full_dataset_labels/train_extra/'
else:
    label_root='./cityscapes_samples_labels/'


def getJsonList():
    json_list=glob.glob(os.path.join(label_root, 'dortmund', '*.json'))
    return json_list

# draw rectangle plt, rects is corner form
def drawRectsPLT(img, rects, texts):
    plt.imshow(img)
    colors = ['green', 'blue', 'red']  # background, vehicle, human
    for i in range(len(rects)):
        plt.gca().add_patch(patches.Rectangle((rects[i][0], rects[i][1]), (rects[i][2]-rects[i][0]),(rects[i][3]-rects[i][1]),
                              fill=False, edgecolor=colors[texts[i]], linewidth=2))
    plt.show()


# unused functions
# def getCurrentDirectory():
#     return os.getcwd()

# draw rectangle cv2, rects is corner form
# def drawRectsCv2(img, rects, texts):
#     for i in range(len(rects)):
#         cv2.rectangle(img, (rects[i][0], rects[i][1]), (rects[i][2], rects[i][3]), (0, 0, 255), 2)
#         cv2.putText(img, texts[i], (rects[i][0], rects[i][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),1)
#     cv2.imshow('image', img)
#     # cv2.imwrite('bad-honnef_000000_000000_leftImg8bit_show.jpg',img)
#     cv2.waitKey(0)

# def getImageList(str):
#     img_list = glob.glob(os.path.join('./', '*_'+str, '*', '*.png'))
#     return img_list
