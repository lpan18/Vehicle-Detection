import torch
import math
import numpy as np

''' Prior Bounding Box  ------------------------------------------------------------------------------------------------
'''
# TODO: compute S_{k} (reference: SSD Paper equation 4.)
m = 6 - 1  # first-layer is set seperately with smin/2=0.1
smin = 0.2
smax = 0.9
input_size = 300
sk = []
step = math.floor(100 * (smax - smin) / (m - 1))
sk.append(smin / 2)
for k in range(0, m):
    sk.append(round(smin + k * step / 100, 2))
# print('sk',sk) # sk [0.1, 0.2, 0.37, 0.54, 0.71, 0.88]

# compute bbox_size
bbox_size = [math.floor(i * input_size) for i in sk]  # [30, 60, 111, 162, 213, 264]
# TODO: define your feature map settings

prior_layer_cfg = [
    {'layer_name': 'Layer_5', 'feature_dim_hw': (38, 38), 'bbox_size': (
        bbox_size[0], bbox_size[0]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
    {'layer_name': 'Layer_11', 'feature_dim_hw': (19, 19), 'bbox_size': (
        bbox_size[1], bbox_size[1]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
    {'layer_name': 'Conv8_2', 'feature_dim_hw': (10, 10), 'bbox_size': (
        bbox_size[2], bbox_size[2]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
    {'layer_name': 'Conv9_2', 'feature_dim_hw': (5, 5), 'bbox_size': (
        bbox_size[3], bbox_size[3]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
    {'layer_name': 'Conv10_2', 'feature_dim_hw': (3, 3), 'bbox_size': (
        bbox_size[4], bbox_size[4]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
    {'layer_name': 'Conv11_2', 'feature_dim_hw': (1, 1), 'bbox_size': (
        bbox_size[5], bbox_size[5]), 'aspect_ratio': (1.0, 1 / 2, 1 / 3, 2.0, 3.0, 1.0)},
]

def generate_prior_bboxes(prior_layer_cfg):
    """
    Generate prior bounding boxes on different feature map level. This function used in 'cityscape_dataset.py'

    Use VGG_SSD 300x300 as example:
    Feature map dimension for each output layers:
       Layer    | Map Dim (h, w) | Single bbox size that covers in the original image
    1. Conv4    | (38x38)        | (30x30) (unit. pixels)
    2. Conv7    | (19x19)        | (60x60)
    3. Conv8_2  | (10x10)        | (111x111)
    4. Conv9_2  | (5x5)          | (162x162)
    5. Conv10_2 | (3x3)          | (213x213)
    6. Conv11_2 | (1x1)          | (264x264)
    NOTE: The setting may be different using MobileNet v3, you have to set your own implementation.
    Tip: see the reference: 'Choosing scales and aspect ratios for default boxes' in original paper page 5.
    :param prior_layer_cfg: configuration for each feature layer, see the 'example_prior_layer_cfg' in the following.
    :return prior bounding boxes with form of (cx, cy, w, h), where the value range are from 0 to 1, dim (1, num_priors, 4)
    """

    priors_bboxes = []
    # iterate each layers
    for feat_level_idx in range(0, len(prior_layer_cfg)):
        layer_cfg = prior_layer_cfg[feat_level_idx]
        layer_feature_dim = layer_cfg['feature_dim_hw']
        layer_aspect_ratio = layer_cfg['aspect_ratio']

        # feature map size
        fk = layer_feature_dim[0]
        for y in range(0, fk):
            for x in range(0, fk):
                # Todo: compute bounding box center
                cx = (x + 0.5) / fk
                cy = (y + 0.5) / fk
                # Todo: generate prior bounding box with respect to the aspect ratio
                count = 0
                for aspect_ratio in layer_aspect_ratio:
                    count += 1
                    if(count != 6):
                        h = sk[feat_level_idx] / math.sqrt(aspect_ratio)
                        w = sk[feat_level_idx] * math.sqrt(aspect_ratio)
                    else:
                        if feat_level_idx==len(prior_layer_cfg)-1:
                            sk_ba = math.sqrt(sk[feat_level_idx] * 1.05)
                        else:
                            sk_ba = math.sqrt(sk[feat_level_idx] * sk[feat_level_idx+1])
                        h = sk_ba / math.sqrt(aspect_ratio)
                        w = sk_ba * math.sqrt(aspect_ratio)
                    priors_bboxes.append([cx, cy, w, h])

    # Convert to Tensor
    priors_bboxes = torch.tensor(priors_bboxes)
    priors_bboxes = torch.clamp(priors_bboxes, 0.0, 1.0)
    # num_priors = priors_bboxes.shape[0]

    # [DEBUG] check the output shape
    assert priors_bboxes.dim() == 2
    assert priors_bboxes.shape[1] == 4
    return priors_bboxes

def iou(a: torch.Tensor, b: torch.Tensor):
    """
    # Compute the Intersection over Union
    Note: function iou(a, b) used in match_priors
    :param a: bounding boxes, dim: (n_items, 4) center form
    :param b: bounding boxes, dim: (n_items, 4) or (1, 4) if b is a reference
    :return: iou value: dim: (n_item)
    """
    # [DEBUG] Check if input is the desire shape
    assert a.dim() == 2
    assert a.shape[1] == 4
    assert b.dim() == 2
    assert b.shape[1] == 4

    # TODO: implement IoU of two bounding box
    intersection=torch.zeros([a.shape[0]])
    union=torch.zeros([a.shape[0]])

    # if b(1, 4) then repeat b to the same dimension to a
    if(b.shape[0]==1):
        b=b.repeat(a.shape[0],1)

    for i in range(a.shape[0]):
        if(abs(a[i, 0] - b[i, 0]) < (a[i, 2] + b[i, 2])/ 2.0 and abs(a[i, 1] - b[i, 1]) < (a[i, 3] + b[i, 3]) / 2.0):
            lt_x_inter = torch.max((a[i, 0] - (a[i, 2] / 2.0)),
                             (b[i, 0] - (b[i, 2] / 2.0)))
            lt_y_inter = torch.min((a[i, 1] + (a[i, 3] / 2.0)),
                             (b[i, 1] + (b[i, 3] / 2.0)))
            rb_x_inter = torch.min((a[i, 0] + (a[i, 2] / 2.0)),
                             (b[i, 0] + (b[i, 2] / 2.0)))
            rb_y_inter = torch.max((a[i, 1] - (a[i, 3] / 2.0)),
                             (b[i, 1] - (b[i, 3] / 2.0)))
            intersection[i]=abs((rb_x_inter - lt_x_inter)* (rb_y_inter - lt_y_inter))
        union[i] = (a[i, 2] * a[i, 3]) + (b[i, 2] * b[i, 3]) - intersection[i]
    iou=torch.div(intersection,union)

    # [DEBUG] Check if output is the desire shape
    assert iou.dim() == 1
    assert iou.shape[0] == a.shape[0]
    return iou


def bbox2loc(bbox, priors, center_var=0.1, size_var=0.2):
    """
    Compute boxes (cx, cy, h, w) to SSD locations form. - encode
    :param bbox: bounding box (cx, cy, h, w) , dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: loc: (cx, cy, h, w)
    """
    # assert priors.shape[0] == 1
    # assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    b_center = bbox[..., :2]
    b_size = bbox[..., 2:]

    return torch.cat([
        1 / center_var * ((b_center - p_center) / p_size),
        torch.log(b_size / p_size) / size_var
    ], dim=-1)

def match_priors(prior_bboxes: torch.Tensor, gt_bboxes: torch.Tensor, gt_labels: torch.Tensor, iou_threshold: float):
    """
    Match the ground-truth boxes with the priors.
    Note: Use this function in your 'cityscape_dataset.py', see the SSD paper page 5 for reference. (note that default box = prior boxes)

    :param gt_bboxes: ground-truth bounding boxes, dim:(n_samples, 4)
    :param gt_labels: ground-truth classification labels, negative (background) = 0, dim: (n_samples)
    :param prior_bboxes: prior bounding boxes on different levels, dim:(num_priors, 4)
    :param iou_threshold: matching criterion
    :return matched_boxes: real matched bounding box, dim: (num_priors, 4)
    :return matched_labels: real matched classification label, dim: (num_priors)
    """
    # [DEBUG] Check if input is the desire shape
    assert gt_bboxes.dim() == 2
    assert gt_bboxes.shape[1] == 4
    assert gt_labels.dim() == 1
    assert gt_labels.shape[0] == gt_bboxes.shape[0]
    assert prior_bboxes.dim() == 2
    assert prior_bboxes.shape[1] == 4

    matched_labels = np.zeros(prior_bboxes.shape[0])
    matched_boxes = prior_bboxes.numpy().copy()

    # TODO: implement prior matching
    iou_calculated=[]
    iou_max=[]
    for i in range(gt_bboxes.shape[0]):
        iou_calculated=iou(prior_bboxes,gt_bboxes[i].unsqueeze(0))
        # match maximum iou
        iou_max_index= torch.argmax(iou_calculated, dim=0)
        matched_labels[iou_max_index]=gt_labels[i]
        matched_boxes[iou_max_index]=gt_bboxes[i]
        # match threshold>0.5
        iou_threshold_index=[i for i in range(iou_calculated.shape[0]) if iou_calculated[i]>iou_threshold]
        if len(iou_threshold_index) !=0:
            matched_labels[iou_threshold_index]=gt_labels[i]
            matched_boxes[iou_threshold_index]=gt_bboxes[i]
    matched_labels=torch.tensor(matched_labels,dtype=torch.float32)
    matched_boxes=torch.tensor(matched_boxes,dtype=torch.float32)
    offset_boxes=bbox2loc(matched_boxes,prior_bboxes)

    # [DEBUG] Check if output is the desire shape
    assert matched_boxes.dim() == 2
    assert matched_boxes.shape[1] == 4
    assert matched_labels.dim() == 1
    assert matched_labels.shape[0] == matched_boxes.shape[0]

    return offset_boxes, matched_labels


''' NMS ----------------------------------------------------------------------------------------------------------------
'''

def nms_bbox(bbox_loc, bbox_confid_scores, overlap_threshold=0.5, prob_threshold=0.6):
    """
    Non-maximum suppression for computing best overlapping bounding box for a object
    Use this function when testing the samples.

    :param bbox_loc: bounding box loc and size, dim: (num_priors, 4)
    :param bbox_confid_scores: bounding box confidence probabilities, dim: (num_priors, num_classes)
    :param overlap_threshold: the overlap threshold for filtering out outliers
    :return: selected bounding box with classes
    """

    # [DEBUG] Check if input is the desire shape
    assert bbox_loc.dim() == 2
    assert bbox_loc.shape[1] == 4
    assert bbox_confid_scores.dim() == 2
    assert bbox_confid_scores.shape[0] == bbox_loc.shape[0]

    sel_bbox = {}

    # Todo: implement nms for filtering out the unnecessary bounding boxes
    num_classes = bbox_confid_scores.shape[1]
    # remove bounding box with iou>overlap_threshold
    for class_idx in range(1, num_classes):    # in range (1,) no need to compute for class 0 as it's the background
        # Tip: use prob_threshold to set the prior that has higher scores and filter out the low score items for fast computation
        scores = bbox_confid_scores[:, class_idx]
        # keep bounding box with confidence probabilities > prob_threshold
        index_prob_keeped=np.argwhere(scores>prob_threshold)
        bbox_loc_keeped=bbox_loc[index_prob_keeped].numpy().squeeze()
        # scores=scores[index_prob_keeped].numpy().squeeze()
        # order = scores.argsort()[::-1]  # return order index from high to low
        # no need to convert to numpy (as above) before sort, use torch directly
        scores=scores[index_prob_keeped].squeeze()
        _, order = scores.sort(dim=0, descending=True)
        keep = []
        # iterate from high score to low score and remove bbox with IoU>threshold
        # while order.size > 1:  # if order convert to numpy
        while len(order) > 1:
            i = order[0]
            keep.append(i)
            iou_calculated = iou(torch.tensor([bbox_loc_keeped[i] for i in order[1:]]), torch.tensor(bbox_loc_keeped[i]).unsqueeze(0))
            inds = np.where(iou_calculated <= overlap_threshold)[0]
            order = order[inds + 1]  # +1 to give the orignal indices, as when compute iou, we use order[1:] from the second item
        bbox = bbox_loc_keeped[keep].squeeze()
        sel_bbox[class_idx] = bbox.tolist()
        # pass
    return sel_bbox


''' Bounding Box Conversion --------------------------------------------------------------------------------------------
'''

def loc2bbox(loc, priors, center_var=0.1, size_var=0.2):
    """
    Compute SSD predicted locations to boxes(cx, cy, h, w). - decode
    :param loc: predicted location, dim: (N, num_priors, 4)
    :param priors: default prior boxes, dim: (1, num_prior, 4)
    :param center_var: scale variance of the bounding box center point
    :param size_var: scale variance of the bounding box size
    :return: boxes: (cx, cy, h, w)
    """
    assert priors.shape[0] == 1
    assert priors.dim() == 3

    # prior bounding boxes
    p_center = priors[..., :2]
    p_size = priors[..., 2:]

    # locations
    l_center = loc[..., :2]
    l_size = loc[..., 2:]

    # real bounding box
    return torch.cat([
        center_var * l_center * p_size + p_center,      # b_{center}
        p_size * torch.exp(size_var * l_size)           # b_{size}
    ], dim=-1)

def center2corner(center):
    """
    Convert bounding box in center form (cx, cy, w, h) to corner form (x,y) (x+w, y+h)
    :param center: bounding box in center form (cx, cy, w, h)
    :return: bounding box in corner form (x,y) (x+w, y+h)
    """
    return torch.cat([center[..., :2] - center[..., 2:] / 2,
                      center[..., :2] + center[..., 2:] / 2], dim=-1)

def corner2center(corner):
    """
    Convert bounding box in corner form (x,y) (x+w, y+h) to center form (cx, cy, w, h)
    :param center: bounding box in corner form (x,y) (x+w, y+h)
    :return: bounding box in center form (cx, cy, w, h)
    """
    return torch.cat([(corner[..., :2] + corner[..., 2:]) / 2,
                      corner[..., 2:] - corner[..., :2]], dim=-1)

# test iou
# generate_prior_bboxes(prior_layer_cfg)
# a=torch.tensor([[1.0, 2.0, 2.0, 2.0],[2, 1, 2, 2],[1, 100, 1, 1]])
# b=torch.tensor([[2.0, 1.0, 2.0, 2.0],[1, 2, 2, 2],[2, 1, 1, 1]])
# print(iou(a,b))

# test match_priors
# prior_bboxes=torch.tensor([[2.0, 1, 1, 1.5],[2, 1, 1, 0.8],[5, 5, 1, 0.7]])
# gt_bboxes=torch.tensor([[2.0, 1.0, 1, 1],[2.0, 1.0, 1, 2]])
# gt_labels=torch.tensor([10,20])
# match_priors(prior_bboxes,gt_bboxes,gt_labels,0.5)

# test center2corner corner2center
# center = torch.tensor([[1, 2, 2, 2], [2, 1, 2, 2]])
# corner = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]])
# print(center2corner(center))
# print(corner2center(corner))
