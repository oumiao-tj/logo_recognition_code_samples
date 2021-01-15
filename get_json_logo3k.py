import os
import argparse

parser = argparse.ArgumentParser(description='main')
parser.add_argument("--detect_thres", type=float, default = 0.1)
parser.add_argument("--gpu_id", type=str, default = "0")
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import numpy as np
import json
from PIL import Image
import imageio
import tqdm

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import tensorflow.python.keras.backend as K
from core.yolo_detection import YOLO

import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from torchvision import transforms
from config import PROPOSAL_NUM, INPUT_SIZE
from core import nts

device = torch.device('cuda')

def _build_session(graph):
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
    sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
    tf.Session(config=sess_config, graph=graph)
    return

_build_session(tf.Graph())

def load_yolo(yolo_weights_path, yolo_anchors_path, yolo_classes_path, score = 0.1, gpu_num = 1):
    yolo = YOLO(**{"model_path": yolo_weights_path,
                   "anchors_path": yolo_anchors_path,
                   "classes_path": yolo_classes_path,
                   "score" : score,
                   "gpu_num" : gpu_num,
                   "model_image_size" : (416, 416),
                   })
    return yolo

def load_nts(nts_weights_path, device):
    net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=3000)
    ckpt = torch.load(nts_weights_path, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    net = net.to(device)
    net.eval()
    return net

def get_statistics(outputs, targets, iou_threshold):
    """ Compute true positives, predicted scores and predicted labels per sample """
    metrics = []
    for sample_i in range(len(outputs)):

        if outputs[sample_i] is None:
            continue

        output = outputs[sample_i]
        pred_boxes = output[:, :4]
        pred_scores = output[:, 5]
        pred_labels = output[:, -1]

        true_positives = np.zeros(pred_boxes.shape[0])

        annotations = targets[sample_i]
        target_labels = annotations[:, 0] if len(annotations) else []
        if len(annotations):
            detected_boxes = []
            target_boxes = annotations[:, 1:]

            for pred_i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):

                # If targets are found break
                if len(detected_boxes) == len(annotations):
                    break

                # Ignore if label is not one of the target labels
                if pred_label not in target_labels:
                    continue

                iou, box_index = bbox_iou(pred_box.unsqueeze(0), target_boxes).max(0)

                if iou >= iou_threshold and box_index not in detected_boxes:
                    true_positives[pred_i] = 1
                    detected_boxes += [box_index]
        metrics.append([true_positives, pred_scores, pred_labels])
    return metrics

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def compute_ap(recall, precision):
    """ Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.

    # Arguments
        recall:    The recall curve (list).
        precision: The precision curve (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def ap_per_class(tp, conf, pred_cls, target_cls):
    """ Compute the average precision, given the recall and precision curves.
    Source: https://github.com/rafaelpadilla/Object-Detection-Metrics.
    # Arguments
        tp:    True positives (list).
        conf:  Objectness value from 0-1 (list).
        pred_cls: Predicted object classes (list).
        target_cls: True object classes (list).
    # Returns
        The average precision as computed in py-faster-rcnn.
    """

    # Sort by objectness
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)

    # Create Precision-Recall curve and compute AP for each class
    ap, p, r = [], [], []
    for c in unique_classes:
        i = pred_cls == c
        n_gt = (target_cls == c).sum()  # Number of ground truth objects
        n_p = i.sum()  # Number of predicted objects

        if n_p == 0 and n_gt == 0:
            continue
        elif n_p == 0 or n_gt == 0:
            ap.append(0)
            r.append(0)
            p.append(0)
        else:
            # Accumulate FPs and TPs
            fpc = (1 - tp[i]).cumsum()
            tpc = (tp[i]).cumsum()

            # Recall
            recall_curve = tpc / (n_gt + 1e-16)
            r.append(recall_curve[-1])

            # Precision
            precision_curve = tpc / (tpc + fpc)
            p.append(precision_curve[-1])

            # AP from recall-precision curve
            ap.append(compute_ap(recall_curve, precision_curve))

    # Compute F1 score (harmonic mean of precision and recall)
    p, r, ap = np.array(p), np.array(r), np.array(ap)
    f1 = 2 * p * r / (p + r + 1e-16)

    return p, r, ap, f1, unique_classes.astype("int32")

yolo_weights_path = "./model_data/logo3k_yolo_weights.h5"
yolo_anchors_path = "./model_data/yolo/yolo_anchors.txt"
yolo_classes_path = "./model_data/yolo/data_classes.txt"
yolo = load_yolo(yolo_weights_path, yolo_anchors_path, yolo_classes_path, score = args.detect_thres, gpu_num = 1)
print("YOLO weights loaded!")

nts_weights_path = "./model_data/logo3k_nts.ckpt"
nts = load_nts(nts_weights_path, device)
print("NTS-Net weights loaded!")

with open("logo_label_map.json", "r") as f:
    logo_label_map = json.load(f)
with open("train_test_split.json", "r") as f:
    train_test_split = json.load(f)

targets = [[], [], []]
outputs = [[], [], []]
true_labels = [[], [], []]
img_id = -1
with open("data_test.txt", "r") as f:
    line = f.readline()
    while line:
        img_id += 1
        if img_id % 1000 == 0:
            print("{} images processed!".format(img_id))

        tmp = line.split('.jpg')
        logo_id = logo_label_map[tmp[0].split('/')[-2]]
        path = tmp[0] + '.jpg'
        true_boxes = tmp[1].strip().split()
        target = []
        for box in true_boxes:
            xmin, ymin, xmax, ymax, _ = box.split(',')
            target.append(list(map(int, [logo_id, xmin, ymin, xmax, ymax])))
        target = torch.tensor(target, dtype=torch.float)
        true_labels[2].extend(target[:, 0].tolist())
        targets[2].append(target)
        true_labels[train_test_split[img_id]].extend(target[:, 0].tolist())
        targets[train_test_split[img_id]].append(target)

        image = Image.open(path)
        frame = imageio.imread(path)
        cropped_frames = []
        pred_boxes = yolo.detect_image(image)
        if not pred_boxes:
            outputs[2].append(None)
            outputs[train_test_split[img_id]].append(None)
            line = f.readline()
            continue
        output = []
        for i, box in enumerate(pred_boxes):
            xmin, ymin, xmax, ymax, _, yolo_score = box
            output.append([xmin, ymin, xmax, ymax, yolo_score])
            cropped_frames.append(frame[ymin:ymax, xmin:xmax])
        output = torch.tensor(output)
        imgs = torch.zeros(len(pred_boxes), 3, INPUT_SIZE[0], INPUT_SIZE[1])
        for i, cropped_frame in enumerate(cropped_frames):
            img = Image.fromarray(cropped_frame, mode="RGB")
            img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
            img = transforms.ToTensor()(img)
            img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
            imgs[i] = img[...]
        with torch.no_grad():
            imgs = imgs.to(device)
            _, logits, _, _, _, _ = nts(imgs)
        _, predicts = torch.max(logits, 1)
        prob = F.softmax(logits, dim = 1)
        indices = list(range(len(pred_boxes)))
        scores = prob[indices, predicts[indices]].cpu()
        output = torch.cat([output, scores.unsqueeze(1), predicts.cpu().unsqueeze(1)], dim=1)
        output = output[(-scores).argsort()]
        outputs[2].append(output)
        outputs[train_test_split[img_id]].append(output)

        line =f.readline()

print("All images processed!")

print("Using detect_thres: {}".format(args.detect_thres))

for i in range(3):
    metrics = get_statistics(outputs[i], targets[i], iou_threshold=0.5)
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*metrics))]

    precision, recall, AP, f1, ap_class = ap_per_class(true_positives,\
        pred_scores, pred_labels, true_labels[i])

    print("The mAP on {}% validation set ({} images) is {}.\n\n".format\
        ((i+1)*10, len(targets[i]), AP.mean()))
