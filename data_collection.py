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
from torch.nn import DataParallel
from config import INPUT_SIZE, PROPOSAL_NUM
from core import nts
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler


class DataCollection:
    def __init__(self, parser=None):
        self.download_dir = parser.parse_args().download_dir
        self.limit = parser.parse_args().limit
        self.print_paths = parser.parse_args().print_paths
        brand_name_list_path = parser.parse_args().brand_name_list_path
        self.nts_weight_path = parser.parse_args().nts_weight_path
        self.batchsize = parser.parse_args().batchsize
        if brand_name_list_path:
            with open(brand_name_list_path, 'r') as infile:
                self.brand_name_list = json.load(infile)

    def _main(self):
        #print("-"*50 + "\nSTART DOWNLOADING!\n" + "-"*50)
        #self.downloading(self.brand_name_list, self.limit, self.download_dir)
        #self.downloading(self.brand_name_list[:10], self.limit, self.download_dir)
        print("\n" + "-"*50 + "\nSTART PRECLEANING!\n" + "-"*50)
        self.preclean(self.download_dir)
        print("\n" + "-"*50 + "\nSTART ENCODING!\n" + "-"*50)
        self.encoding(self.download_dir)

    # require to install Chrome browser and chromedriver if setting "limit" > 100
    def downloading(self, brand_name_list, limit, download_dir):
        response = google_images_download.googleimagesdownload()
        for name in brand_name_list:
            keywords = name + ' logo'
            img_dir = '_'.join([x for x in keywords.split() if x])
            arguments = {"keywords":keywords,"limit":limit,"print_urls":False,'format':'jpg',\
                         "output_directory":download_dir,"image_directory":img_dir,\
                         "chromedriver":'./chromedriver'}
            paths = response.download(arguments)
            if self.print_paths:
                print(paths)

    # clean damaged/wrong format/too large images
    # remove EXIF data for .jpg and .jpeg images
    def preclean(self, download_dir):
        i = -1
        for root, dir, files in os.walk(download_dir):
            if i == -1:
                i += 1
                continue
            i += 1
            j = 0
            for file in files:
                if file[-4:].lower() not in {'.jpg', '.png', 'jpeg'}:
                    print("Wrong format!", os.path.join(root, file))
                    os.remove(os.path.join(root, file))
                else:
                    try:
                        img = Image.open(os.path.join(root, file))
                        if img.size[0] * img.size[1] > 1e7:
                            print('Too large!', os.path.join(root, file))
                            os.remove(os.path.join(root, file))
                        else:
                            if file[-4:].lower() in {'.jpg', 'jpeg'}:
                                piexif.remove(os.path.join(root, file))
                            tmp = file.split('.')
                            os.rename(os.path.join(root, file), root + '/' + str(j) + '.' + tmp[-1])
                            j += 1
                    except:
                        print('Cannot open!', os.path.join(root, file))
                        os.remove(os.path.join(root, file))
            print(i, root, j)

    def encoding(self, download_dir):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # test_model = '../YOLO_NTS_inference/core/model_data/1145class.ckpt'

        net = nts.AttentionNet(topN=PROPOSAL_NUM, num_classes=1)
        ckpt = torch.load(self.nts_weight_path, map_location=device)
        pretrained_dict = ckpt['net_state_dict']
        net_dict = net.state_dict()
        pretrained_dict = {key: val for key, val in pretrained_dict.items() if val.shape == net_dict[key].shape}
        net_dict.update(pretrained_dict)
        net.load_state_dict(net_dict)
        net = net.to(device)
        net = DataParallel(net)
        net.eval()

        features = {}
        filenames = {}
        i = -1
        for root, dir, files in os.walk(download_dir):
            if i == -1:
                i += 1
                continue
            i += 1
            curr_brand = root.split('/')[-1]
            filenames[curr_brand] = []
            curr_features = []
            for j in range(0, len(files), self.batchsize):
                imgs = []
                for k in range(j, min(j+self.batchsize, len(files))):
                    file = files[k]
                    try:
                        img = imageio.imread(os.path.join(root, file))
                        if len(img.shape) == 2:
                            img = np.stack([img] * 3, 2)
                        img = Image.fromarray(img, mode='RGB')
                        img = transforms.Resize(INPUT_SIZE, Image.BILINEAR)(img)
                        img = transforms.ToTensor()(img)
                        img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(img)
                        img = torch.unsqueeze(img, 0)
                        imgs.append(img)

                        filenames[curr_brand].append(file)
                    except:
                        print(os.path.join(root, file))
                        os.remove(os.path.join(root, file))
                imgs = torch.cat(imgs, dim=0)
                with torch.no_grad():
                    imgs = imgs.to(device)
                    _, _, _, _, _, concat_out = net(imgs)
                curr_feature = concat_out.cpu().tolist()
                curr_features.extend(curr_feature)

            x = np.array(curr_features)
            print(x.shape)
            #scaler = StandardScaler()
            scaler = MinMaxScaler()
            x = scaler.fit_transform(x)
            pca = PCA(n_components=0.7, svd_solver='full')
            x = pca.fit_transform(x)
            print(i, curr_brand, x.shape)

            features[curr_brand] = x.tolist()

        # save encoding features
        with open(download_dir + '_minmax_features.json', 'w') as outfile:
            json.dump(features, outfile)
        print('\nfeatures saved to ' +download_dir + '_minmax_features.json')

        # save filenames corresponding to encoding features
        with open(download_dir + '_minmax_filenames.json', 'w') as outfile:
            json.dump(filenames, outfile)
        print('\nfilenames saved to ' + download_dir + '_minmax_features.json')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Main')
    parser.add_argument("--download_dir", type=str, help='directory for downloading images', default = 'data/downloads')
    parser.add_argument("--limit", type=int, help='number of images to download for each keyword', default = 10)
    parser.add_argument("--print_paths", type=bool, help='print downloading paths or not', default = False)
    parser.add_argument("--brand_name_list_path", type=str, help='path to brand_name_list.json', default = '')
    parser.add_argument("--batchsize", type=int, help='batchsize for encoding', default = 100)
    parser.add_argument("--nts_weight_path", type=str, help='path to nts_weight', default = 'model_data/nts/nts50_4_2_1660_cropping.ckpt')

    data = DataCollection(parser)
    data._main()
