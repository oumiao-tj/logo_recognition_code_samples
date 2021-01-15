import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.python.util.deprecation as deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False

import argparse
import cv2
import numpy as np
import json
import colorsys
from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import imageio

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


class LogoRecognition:
    def __init__(self, parser=None):
        # restrict gpu memory for tensorflow
        self._build_session(tf.Graph())
        # specify device for torch
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = self.__add_args(parser)

        if self.args.inference_type not in {'c', 'r'}:
            raise ValueError("Wrong inference_type!")
        if self.args.input_type not in {'image', 'video'}:
            raise ValueError("Wrong input_type!")
        with open(self.args.r_labels, 'r') as infile:
            self.r_labels = json.load(infile)
        with open(self.args.r_classes, 'r') as infile:
            self.r_classes = json.load(infile)
        with open(self.args.r_imagenames, 'r') as infile:
            self.r_imagenames = json.load(infile)
        self.num_classes = len(self.r_classes)
        self.r_feature_pool = np.load(self.args.r_feature_pool)
        self.r_feature_pool = torch.from_numpy(self.r_feature_pool)
        print("---feature pool loaded!---")
        self.yolo = self.load_yolo(self.args.yolo_weights, self.args.yolo_anchors, self.args.yolo_classes, score = 0.1, gpu_num = 1)
        print("---YOLO model loaded!---")
        self.nts = self.load_nts(self.args.nts_weights)
        print("---NTS model loaded!---")

        self.tags = []

    def __add_args(self, parser=None):
        parser = argparse.ArgumentParser(description='Logo') if not parser else parser
        io_path = 'model_data'

        parser.add_argument("--inference_type", type=str, help="'c' for classification inference or 'r' for retrieval inference", default = 'r')
        parser.add_argument("--input_type", type=str, help="'image' or 'video'", default = 'image')
        parser.add_argument("--input_path", type=str, help="image/video input path", default = './test_image.jpg')
        parser.add_argument("--save_dir", type=str, help="directory to save output images/videos", default = './')

        parser.add_argument("--yolo_weights", help='yolo weights', default = os.path.join(io_path,'yolo/yolo_weights_logos.h5'))
        parser.add_argument("--nts_weights", help='nts weights', default = os.path.join(io_path,'nts/nts50_4_2.ckpt'))
        parser.add_argument("--yolo_anchors", help='yolo anchors', default = os.path.join(io_path,'yolo/yolo_anchors.txt'))
        parser.add_argument("--yolo_classes", help='yolo classes', default = os.path.join(io_path,'yolo/data_classes.txt'))
        parser.add_argument("--r_feature_pool", help='feature pool for retrieval inference', default = os.path.join(io_path,'retrieval/nts_feature_pool.npy'))
        parser.add_argument("--r_labels", help='label file for retrieval inference', default = os.path.join(io_path,'retrieval/cropped_pool_labels.json'))
        parser.add_argument("--r_classes", help='class file for retrieval inference', default = os.path.join(io_path,'retrieval/cropped_pool_classes.json'))
        parser.add_argument("--r_imagenames", help='imagename file for retrieval inference', default = os.path.join(io_path,'retrieval/cropped_pool_images.json'))

        parser.add_argument("--detect_thres", type=float, help='threshold for logo detection', default = 0.1)
        parser.add_argument("--classify_thres", type=float, help='threshold for logo classification', default = 0.8)
        parser.add_argument("--retrieval_thres", type=float, help='threshold for logo retrieval', default = 0.7)
        parser.add_argument("--retrieval_interval", type=float, help='interval for detecting logo retrieval multi-classes', default = -1)
        parser.add_argument("--enlarge_bbox_ratio", type=float, help='ratio for enlarging bboxes generated by YOLO', default = 1.2)

        args = parser.parse_args()

        return args

    def _main(self, input_path=None, save_dir=None):
        if not input_path:
            input_path = self.args.input_path
        if not save_dir:
            save_dir = self.args.save_dir
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        tmp = input_path.split('/')[-1].split('.')
        output_file = '.'.join(tmp[:-1]) + '_' + self.args.inference_type + '_tagged.' + tmp[-1]
        output_json = '.'.join(tmp[:-1]) + '_' + self.args.inference_type + '_tagged.json'
        if self.args.input_type == 'image':
            frame = imageio.imread(input_path)
            image, res = self.tag_frame(frame)
            image.save(os.path.join(save_dir, output_file))
            with open(os.path.join(save_dir, output_json), 'w') as f:
                json.dump(res, f)
        else:
            tags = self.tag_video(input_path, os.path.join(save_dir, output_file))
            with open(os.path.join(save_dir, output_json), 'w') as f:
                json.dump(tags, f)


    def _build_session(self, graph):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3, allow_growth=True)
        sess_config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
        tf.Session(config=sess_config, graph=graph)
        return

    def load_nts(self, nts_weights_path):
        net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=self.num_classes)
        ckpt = torch.load(nts_weights_path, map_location=self.device)
        net.load_state_dict(ckpt['net_state_dict'])
        net = net.to(self.device)
        net = DataParallel(net)
        net.eval()
        return net

    def load_yolo(self, yolo_weights_path, yolo_anchors_path, yolo_classes_path, score = 0.1, gpu_num = 1):
        yolo = YOLO(**{"model_path": yolo_weights_path,
                       "anchors_path": yolo_anchors_path,
                       "classes_path": yolo_classes_path,
                       "score" : score,
                       "gpu_num" : gpu_num,
                       "model_image_size" : (416, 416),
                      }
                   )
        return yolo

    def yolo_detection(self, frames):
        cropped_list, index_list, bbox_list = [], [], []
        for i, frame in enumerate(frames):
            h = frame.shape[0]
            w = frame.shape[1]
            image = Image.fromarray(frame, mode='RGB')
            curr_bbox_list = self.yolo.detect_image(image)
            candidates, new_bbox_list = self.yolo_filter(frame, curr_bbox_list, h, w)
            cropped_list.extend(candidates)
            bbox_list.extend(new_bbox_list)
            index_list.extend([i] * len(candidates))

        return cropped_list, index_list, bbox_list

    def yolo_filter(self, frame, bbox_list, h, w, min_logo_size = (10,10)):
        candidates =[]
        new_bbox_list = []
        for xmin, ymin, xmax, ymax, num, score in bbox_list:
            # do not consider tiny logos
            if xmax-xmin > min_logo_size[1] and ymax-ymin > min_logo_size[0] and score >= self.args.detect_thres:
                # enlarge each bbox predicted by YOLO
                enlarge_w = (xmax - xmin) * (self.args.enlarge_bbox_ratio - 1) / 2
                enlarge_h = (ymax - ymin) * (self.args.enlarge_bbox_ratio - 1) / 2
                xmin = max(0, int(xmin - enlarge_w))
                ymin = max(0, int(ymin - enlarge_h))
                xmax = min(w, int(xmax + enlarge_w))
                ymax = min(h, int(ymax + enlarge_h))
                candidates.append(frame[ymin:ymax, xmin:xmax])
                new_bbox_list.append((xmin, ymin, xmax, ymax, num, score))
        return candidates, new_bbox_list

    def run(self, frames):
        n_frames = frames.shape[0]
        res = {}
        for i in range(n_frames):
            res[i] = {'logo_detection':{}}

        cropped_list, index_list, bbox_list = self.yolo_detection(frames)
        if cropped_list:
            if self.args.inference_type == 'r':
                features_cand = self.nts_feature_extractor(cropped_list)
                rows, scores = self.cos_similarity(features_cand)
                for row, score, bbox, index in zip(rows, scores, bbox_list, index_list):
                    if score >= self.args.retrieval_thres:
                        h = frames[index].shape[0]
                        w = frames[index].shape[1]
                        if 'tags' not in res[index]['logo_detection']:
                            res[index]['logo_detection']['tags'] = []
                        curr_tag = {'text': self.r_classes[self.r_labels[row]],
                                    'confidence': float(score),
                                    'box': {'x1': float(round(bbox[0]/w, 4)),
                                            'y1': float(round(bbox[1]/h, 4)),
                                            'x2': float(round(bbox[2]/w, 4)),
                                            'y2': float(round(bbox[3]/h, 4))
                                           },
                                    'true_box': {'x1': int(max(bbox[0], 0)),
                                                 'y1': int(max(bbox[1], 0)),
                                                 'x2': int(min(bbox[2], w)),
                                                 'y2': int(min(bbox[3], h))
                                                }
                                   }
                        res[index]['logo_detection']['tags'].append(curr_tag)
            else:
                labels, scores = self.nts_classifier(cropped_list)
                for label, score, bbox, index in zip(labels, scores, bbox_list, index_list):
                    if score >= self.args.classify_thres:
                        h = frames[index].shape[0]
                        w = frames[index].shape[1]
                        if 'tags' not in res[index]['logo_detection']:
                            res[index]['logo_detection']['tags'] = []
                        curr_tag = {'text': self.r_classes[label],
                                    'confidence': float(score),
                                    'box': {'x1': float(round(bbox[0]/w, 4)),
                                            'y1': float(round(bbox[1]/h, 4)),
                                            'x2': float(round(bbox[2]/w, 4)),
                                            'y2': float(round(bbox[3]/h, 4))
                                           },
                                    'true_box': {'x1': int(max(bbox[0], 0)),
                                                 'y1': int(max(bbox[1], 0)),
                                                 'x2': int(min(bbox[2], w)),
                                                 'y2': int(min(bbox[3], h))
                                                }
                                   }
                        res[index]['logo_detection']['tags'].append(curr_tag)

        return res

    def nts_classifier(self, cropped_list):
        n_list = len(cropped_list)
        logits = []
        batch_size = 20
        for i in range(0, n_list, batch_size):
            imgs = torch.zeros(batch_size, 3, INPUT_SIZE[0], INPUT_SIZE[1])
            for j in range(i, min(n_list, i+batch_size)):
                img = Image.fromarray(cropped_list[j], mode="RGB")
                img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                imgs[j-i] = img[...]
            with torch.no_grad():
                imgs = imgs.to(self.device)
                _, concat_logits, _, _, _, _ = self.nts(imgs)
            logits.append(concat_logits)
        logits = torch.cat(logits, dim=0)[:n_list,:]
        _, predicts = torch.max(logits, 1)
        labels = predicts.tolist()
        prob = F.softmax(logits, dim = 1)
        scores = [prob[i, labels[i]].item() for i in range(n_list)]

        return labels, scores

    def nts_feature_extractor(self, cropped_list):
        n_list = len(cropped_list)
        features = []
        batch_size = 20
        for i in range(0, n_list, batch_size):
            imgs = torch.zeros(batch_size, 3, INPUT_SIZE[0], INPUT_SIZE[1])
            for j in range(i, min(n_list, i+batch_size)):
                img = Image.fromarray(cropped_list[j], mode="RGB")
                img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
                img = transforms.ToTensor()(img)
                img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                imgs[j-i] = img[...]
            with torch.no_grad():
                imgs = imgs.to(self.device)
                _, _, _, _, _, curr_feature = self.nts(imgs)
            features.append(curr_feature / torch.norm(curr_feature, dim=1, keepdim=True))
        features = torch.cat(features, dim=0)

        return features[:n_list,:].cpu()

    def cos_similarity(self, features_cand, eps=1e-8):
        n_bbox = features_cand.size(0)
        cos_sim = torch.mm(self.r_feature_pool, torch.transpose(features_cand, 0, 1))
        scores, rows = torch.max(cos_sim, 0)
        #thres = cos_sim - (scores - self.args.retrieval_interval).unsqueeze(0)
        #mask = (thres > 0)
        #indices = torch.nonzero(mask)
        #logos = [set() for _ in range(n_bbox)]
        #for index in indices:
        #    logos[index[1]].add(index[0])
        #for i in range(n_bbox):
        #    if len(logos[i]) > 1:
        #        scores[i] = 0

        return rows.cpu().numpy(), np.round(scores.cpu().numpy(), 3)

    def tag_frame(self, frame):
        image = Image.fromarray(frame)
        image = image.convert('RGB')
        frame = np.asarray(image, dtype="uint8")
        frames = np.array([frame])
        res = self.run(frames)
        tags = res[0]['logo_detection']
        if not tags:
            return image, []
        boxes = tags['tags']

        font_path = os.path.join(os.path.dirname(__file__),'core/font/FiraMono-Medium.otf')
        font = ImageFont.truetype(font=font_path,
                    size=np.floor(3e-2 * frame.shape[0] + 0.5).astype('int32'))
        thickness = (frame.shape[0] + frame.shape[1]) // 300

        for box in boxes:
            text, confidence, bbox = box['text'], box['confidence'], box['true_box']
            left, top, right, bottom = bbox['x1'], bbox['y1'], bbox['x2'], bbox['y2']
            label = '{} {:.2f}'.format(text, confidence)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(frame.shape[0], np.floor(bottom + 0.5).astype('int32'))
            right = min(frame.shape[1], np.floor(right + 0.5).astype('int32'))

            if bottom > frame.shape[0] or right > frame.shape[1]:
                continue

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, bottom])

            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline='Red')
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill='Red')

            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        return image, res

    def tag_video(self, video_path, output_path):
        vid = cv2.VideoCapture(video_path)
        if not vid.isOpened():
            raise IOError("Couldn't open webcam or video")
        video_FourCC = cv2.VideoWriter_fourcc(*'mp4v')
        video_fps = vid.get(cv2.CAP_PROP_FPS)
        video_size = (int(vid.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        isOutput = True if output_path else False
        if isOutput:
            print('-'*50)
            print(f"Tagging video {video_path}")
            out = cv2.VideoWriter(output_path, video_FourCC, video_fps, video_size)
        accum_time = 0
        curr_fps = 0
        fps = "FPS: ??"
        prev_time = timer()
        n_frame = 0
        tags = []
        while vid.isOpened():
            return_value, frame = vid.read()
            if not return_value:
                break
            frame = frame[:,:,::-1] #opencv images are BGR, translate to RGB
            image, res = self.tag_frame(frame)
            tags.extend(res)
            result = np.asarray(image)
            curr_time = timer()
            exec_time = curr_time - prev_time
            prev_time = curr_time
            accum_time = accum_time + exec_time
            curr_fps = curr_fps + 1
            if accum_time > 1:
                accum_time = accum_time - 1
                fps = "FPS: " + str(curr_fps)
                curr_fps = 0
            cv2.putText(result, text=fps, org=(3, 15), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.50, color=(255, 0, 0), thickness=2)
            if isOutput:
                out.write(result[:,:,::-1])
            n_frame += 1
            if n_frame % 100 ==0:
                print('{} frames tagged!'.format(n_frame))
        vid.release()
        out.release()
        print(f"Tagged video saved as {output_path}!")
        return tags



if __name__ == "__main__":
    LR = LogoRecognition()
    LR._main()
