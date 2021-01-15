#from google_images_download import google_images_download
import piexif
import os
import numpy as np
import json
import argparse
from PIL import Image
import imageio
import torch.utils.data
from torchvision import transforms
import torch
import torch.nn.functional as F
from torch.nn import DataParallel
from config import INPUT_SIZE, PROPOSAL_NUM, NUM_CLASSES
from core import nts
from collections import defaultdict
from core.utils import progress_bar
from itertools import product
from shutil import copy

def extract_features(img_dir, logo_label_lst, nts_weight_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=NUM_CLASSES)
    ckpt = torch.load(nts_weight_path, map_location=device)
    net.load_state_dict(ckpt['net_state_dict'])
    net = net.to(device)
    net = DataParallel(net)
    net.eval()

    batchsize = 20
    eps = 1e-8
    i = -1
    false_predict, true_predict, similarity = {}, {}, {}
    logo_label_dict = {x[0]:int(x[1]) for x in logo_label_lst}
    label_logo_dict = {int(x[1]):x[0] for x in logo_label_lst}
    for root, dir, files in os.walk(img_dir):
        if i == -1:
            i += 1
            continue
        i += 1
        curr_logo = root.split('/')[-1][:-5]
        ignore = False
        if curr_logo not in logo_label_dict or curr_logo in {"MGM Resorts International", "Hyundai", "sony"}:
            ignore = True
        if ignore:
            curr_label = 0
        else:
            curr_label = logo_label_dict[curr_logo]
        true_features = []
        false_predict[curr_logo] = []
        true_predict[curr_logo] = []
        for j in range(0, len(files), batchsize):
            imgs = []
            for k in range(j, min(j+batchsize, len(files))):
                file = files[k]
                img = imageio.imread(os.path.join(root, file))
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')
                img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                imgs.append(img)
            curr_batchsize = len(imgs)
            imgs = torch.stack(imgs, dim=0)
            with torch.no_grad():
                imgs = imgs.to(device)
                _, concat_logits, _, _, _, concat_out = net(imgs)
            _, concat_predict = torch.max(concat_logits, 1)
            probs = F.softmax(concat_logits, dim=1)
            true_idx = (concat_predict.data == curr_label).nonzero().flatten()
            for k in range(curr_batchsize):
                if k in true_idx or ignore:
                    true_predict[curr_logo].append((files[j+k], float(probs[k,curr_label]), \
                        label_logo_dict[curr_label]))
                else:
                    false_predict[curr_logo].append((files[j+k], float(probs[k,curr_label]), \
                        label_logo_dict[int(concat_predict[k])]))
            if ignore:
                true_features.append(concat_out)
            else:
                true_features.append(concat_out[true_idx])

        if len(true_predict[curr_logo]) + len(false_predict[curr_logo]) <= 10:
            true_predict[curr_logo].extend(false_predict[curr_logo])
            false_predict[curr_logo] = []

        true_features = torch.cat(true_features, dim=0)
        top = torch.mm(true_features, torch.transpose(true_features, 0, 1))
        true_features_norm = torch.norm(true_features, dim=1)
        bot = torch.clamp(torch.mm(true_features_norm.unsqueeze(1), \
            true_features_norm.unsqueeze(0)), min=eps)
        cos_sim = top / bot
        similarity[curr_logo] = cos_sim.cpu().tolist()
        progress_bar(i, NUM_CLASSES, 'extracting features')
    with open("./data/true_predict_{}.json".format(NUM_CLASSES), "w") as f:
        json.dump(true_predict, f)
    with open("./data/false_predict_{}.json".format(NUM_CLASSES), "w") as f:
        json.dump(false_predict, f)
    with open("./data/similarity_{}.json".format(NUM_CLASSES), "w") as f:
        json.dump(similarity, f)

def keep_centers(cos_sim_matrix, thres):
    n = len(cos_sim_matrix)
    nbhd = defaultdict(set)
    for i in range(n-1):
        for j in range(i+1, n):
            if cos_sim_matrix[i][j] >= thres:
                nbhd[i].add(j)
                nbhd[j].add(i)
    order = sorted(list(range(n)), key = lambda i: len(nbhd[i]), reverse=True)
    keep = []
    deleted = set()
    for node in order:
        if node not in deleted:
            keep.append(node)
            deleted = deleted.union(nbhd[node])
    return keep

def optimize_logo(cos_sim_matrix):
    n = len(cos_sim_matrix)
    if n <= 30:
        return list(range(n))
    curr_keep = list(range(n))
    thres = 1
    while len(curr_keep) > 30 and thres >= 0.95:
        thres -= 0.01
        prev_keep, curr_keep = curr_keep, keep_centers(cos_sim_matrix, thres)
    return prev_keep, thres + 0.01

def optimize_pool(similarity_dict, true_predict_dict):
    keep_files = {}
    i = 0
    for logo in similarity_dict:
        if len(true_predict_dict[logo]) <= 30:
            keep_files[logo] = {"files": [x[0] for x in true_predict_dict[logo]],\
                "max_cos_sim": None, "min_cos_sim": None, "clean_thres": None}
            i += 1
            continue
        if len(similarity_dict[logo]) != len(true_predict_dict[logo]):
            raise ValueError("Data mismatch!")
        keep_files[logo] = {"files": [], "max_cos_sim": 0, "min_cos_sim": 1,\
            "clean_thres": None}
        keep_idx, keep_files[logo]["clean_thres"] = optimize_logo(similarity_dict[logo])
        for idx in keep_idx:
            keep_files[logo]["files"].append(true_predict_dict[logo][idx][0])
        keep_idx_len = len(keep_idx)
        for a in range(keep_idx_len - 1):
            for b in range(a + 1, keep_idx_len):
                keep_files[logo]["max_cos_sim"] = max(keep_files[logo]["max_cos_sim"],\
                    similarity_dict[logo][a][b])
                keep_files[logo]["min_cos_sim"] = min(keep_files[logo]["min_cos_sim"],\
                    similarity_dict[logo][a][b])
        i += 1
        progress_bar(i, NUM_CLASSES, 'Optimizing pool')
    with open("./data/keep_files_{}.json".format(NUM_CLASSES), "w") as f:
        json.dump(keep_files, f)

def build_pool(keep_files_dict, img_dir):
    pool_dir = img_dir + "_pool"
    if not os.path.exists(pool_dir):
        os.mkdir(pool_dir)
    for logo in keep_files_dict:
        if not os.path.exists(os.path.join(pool_dir, logo)):
            os.mkdir(os.path.join(pool_dir, logo))
        for file in keep_files_dict[logo]["files"]:
            src = os.path.join(img_dir, logo+" logo", file)
            dest = os.path.join(pool_dir, logo, file)
            copy(src, dest)

def extract_pool_features(pool_dir, nts_weight_path, save_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=1)
    ckpt = torch.load(nts_weight_path, map_location=device)
    pretrained_dict = ckpt['net_state_dict']
    net_dict = net.state_dict()
    pretrained_dict = {key: val for key, val in pretrained_dict.items()\
        if val.shape == net_dict[key].shape}
    net_dict.update(pretrained_dict)
    net.load_state_dict(net_dict)
    net = net.to(device)
    net = DataParallel(net)
    net.eval()

    batchsize = 20
    label = -1
    cropped_pool_classes = []
    cropped_pool_images = []
    cropped_pool_labels = []
    features = []
    for root, dir, files in os.walk(pool_dir):
        if label == -1:
            label += 1
            continue
        curr_logo = root.split('/')[-1]
        cropped_pool_classes.append(curr_logo)
        for j in range(0, len(files), batchsize):
            imgs = []
            for k in range(j, min(j+batchsize, len(files))):
                file = files[k]
                cropped_pool_images.append(os.path.join(root, file))
                cropped_pool_labels.append(label)
                img = imageio.imread(os.path.join(root, file))
                if len(img.shape) == 2:
                    img = np.stack([img] * 3, 2)
                img = Image.fromarray(img, mode='RGB')
                img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                imgs.append(img)
            imgs = torch.stack(imgs, dim=0)
            with torch.no_grad():
                imgs = imgs.to(device)
                _, _, _, _, _, concat_out = net(imgs)
            features.append(concat_out.cpu())
        label += 1
        progress_bar(label, NUM_CLASSES, 'extracting pool features')

    nts_feature_pool = torch.cat(features, dim=0)
    nts_feature_pool = nts_feature_pool / nts_feature_pool.norm(dim=1, keepdim=True)
    nts_feature_pool = np.array(nts_feature_pool)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    json.dump(cropped_pool_classes, open(save_dir+"/cropped_pool_classes.json", "w"))
    json.dump(cropped_pool_images, open(save_dir+"/cropped_pool_images.json", "w"))
    json.dump(cropped_pool_labels, open(save_dir+"/cropped_pool_labels.json", "w"))
    np.save(save_dir+"/nts_feature_pool.npy", nts_feature_pool)

if __name__ == "__main__":
    img_dir = "./data/LogoEluvio_cleaned_2962classes_crop_merge"
    nts_weight_path = "./model_data/feature_extractor_{}.ckpt".format(NUM_CLASSES)
#    logo_label_lst = json.load(open("./data/logo_label_map_{}.json"\
#        .format(NUM_CLASSES), "r"))
#    extract_features(img_dir, logo_label_lst, nts_weight_path)

#    similarity_dict = json.load(open("./data/similarity_{}.json"\
#        .format(NUM_CLASSES), "r"))
#    true_predict_dict = json.load(open("./data/true_predict_{}.json"\
#       .format(NUM_CLASSES), "r"))
#    optimize_pool(similarity_dict, true_predict_dict)

#    keep_files_dict = json.load(open("./data/keep_files_{}.json"\
#        .format(NUM_CLASSES), "r"))
#    build_pool(keep_files_dict, img_dir)

    pool_dir = "./data/LogoEluvio_cleaned_2960classes_crop_merge_pool"
    save_dir = "./data/retrieval"
    extract_pool_features(pool_dir, nts_weight_path, save_dir)
