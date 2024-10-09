import dsmil as mil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models
import torchvision.transforms.functional as VF
from torchvision import transforms

import sys, argparse, os, glob
import pandas as pd
import numpy as np
from PIL import Image,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES=True

from collections import OrderedDict
from skimage import exposure, io, img_as_ubyte, transform
import warnings
import copy
from sklearn.utils import shuffle
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_fscore_support

class BagDataset():
    def __init__(self, csv_file, transform=None):
        self.files_list = csv_file
        self.transform = transform

    def __len__(self):
        return len(self.files_list)

    def __getitem__(self, idx):
        path = self.files_list[idx]
        img = Image.open(path)
        img_name = path.split(os.sep)[-1]
        x = img_name.split('.')[0].split('-')[1]
        y = img_name.split('.')[0].split('-')[2]
        img_pos = np.asarray([int(x[1:]), int(y[1:])]) # row, col

        im = np.array(img)
        imagelenth=512
        if (np.size(im, 0) != imagelenth) | (np.size(im, 1) != imagelenth):
            z = np.zeros((imagelenth, imagelenth, 3))
            z[0:np.size(im, 0), 0:np.size(im, 1), :] = im[:, :, :]
            img = z
        sample = {'input': img, 'position': img_pos}

        if self.transform:
            sample = self.transform(sample)
        return sample


class ToTensor(object):
    def __call__(self, sample):
        img = sample['input']
        img = VF.to_tensor(img)
        sample['input'] = img
        return sample


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img


def bag_dataset(args, csv_file_path):
    transformed_dataset = BagDataset(csv_file=csv_file_path,
                                     transform=Compose([
                                         ToTensor()
                                     ]))
    dataloader = DataLoader(transformed_dataset, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, drop_last=False)
    return dataloader, len(transformed_dataset)


def test(args, df, milnet):
    saveproball=pd.DataFrame()
    milnet.eval()
    bags_list=df.Internal_ID.unique()
    bags_list=bags_list.tolist()
    num_bags = len(bags_list)
    Tensor = torch.FloatTensor

    test_labels=[]
    test_labels_int=[]
    test_predictions = []
    for i in range(0, num_bags):
        feats_list = []
        pos_list = []
        classes_list = []
        csv_file_id=df[df.Internal_ID==bags_list[i]]
        # true_label=csv_file_id['GT_tiles_level'].unique()
        true_label=[0,1,2,3,4]
        test_labels_int.extend(true_label)# true class
        csv_file_path=csv_file_id['tileAddress'].reset_index(drop=True)
        dataloader, bag_size = bag_dataset(args, csv_file_path)
        with torch.no_grad():
            for iteration, batch in enumerate(dataloader):
                patches = batch['input'].float().cuda()
                patch_pos = batch['position']
                feats, classes = milnet.i_classifier(patches)
                feats = feats.cpu().numpy()
                classes = classes.cpu().numpy()
                feats_list.extend(feats)
                pos_list.extend(patch_pos)
                classes_list.extend(classes)
            pos_arr = np.vstack(pos_list)
            feats_arr = np.vstack(feats_list)
            classes_arr = np.vstack(classes_list)
            bag_feats = torch.from_numpy(feats_arr).cuda()
            ins_classes = torch.from_numpy(classes_arr).cuda()
            bag_prediction, A, _ = milnet.b_classifier(bag_feats, ins_classes)
            #
            dfsave = pd.DataFrame()
            idcol = pd.Series(bags_list[i], index=range(len(classes_list)))
            dfsave.insert(0, 'Internal_ID', idcol)
            dfsave.insert(1, 'tileAddress', csv_file_path)

            insnp = ins_classes.squeeze().cpu().numpy()
            ins1 = pd.DataFrame(insnp)
            ins1.rename(columns={0: 'insProb0', 1: 'insProb1', 2: 'insProb2', 3: 'insProb3', 4: 'insProb4'},inplace=True)


            bagnp = bag_prediction.squeeze().cpu().numpy()
            bagover=np.array([bagnp for i in range(len(classes_list))])
            bag = pd.DataFrame(bagover)
            bag.rename(columns={0: 'bagProb0', 1: 'bagProb1', 2: 'bagProb2', 3: 'bagProb3', 4: 'bagProb4'},inplace=True)

            bag_prediction_softmax = torch.softmax(bag_prediction, 1)

            bagnp1 = bag_prediction_softmax.squeeze().cpu().numpy()
            bagover1 = np.array([bagnp1 for i in range(len(classes_list))])
            bag1 = pd.DataFrame(bagover1)
            bag1.rename(columns={0: 'bagProbsoftmax0', 1: 'bagProbsoftmax1', 2: 'bagProbsoftmax2', 3: 'bagProbsoftmax3', 4: 'bagProbsoftmax4'},inplace=True)

            saveprob=pd.concat([dfsave,ins1,bag,bag1],axis=1)
            saveproball=pd.concat([saveproball,saveprob],axis=0)
            test_predictions.extend([bag_prediction_softmax.squeeze().cpu().numpy()])
            print('bag_prediction',bag_prediction_softmax)

    test_labels_int = np.array(test_labels_int)
    test_predictions = np.array(test_predictions)
    test_predictions_maxs = np.argmax(test_predictions, axis=1)
    test_predictions_labels=[]
    for i in range(len(test_predictions)):
        predictLabel1 = np.zeros(args.num_classes)
        trueLabel1 = np.zeros(args.num_classes)
        predictLabel1[test_predictions_maxs[i]] = 1
        trueLabel1[int(test_labels_int[i])]=1
        test_predictions_labels.extend([predictLabel1])
        test_labels.extend([trueLabel1])
    test_labels = np.array(test_labels)

    auc_value, _, thresholds_optimal = multi_label_roc(test_labels, test_predictions, args.num_classes,pos_label=1)

    bag_score=0
    dpre = pd.DataFrame(columns=['Internal_ID','prediction probability','true label', 'predict label'])
    for i in range(len(bags_list)):
        dpre = dpre.append({'Internal_ID':bags_list[i],'prediction probability':test_predictions[i],'true label':test_labels[i], 'predict label':test_predictions_labels[i]},ignore_index=True)
        bag_score = np.array_equal(test_labels[i], test_predictions_labels[i]) + bag_score
    avg_score = bag_score / len(bags_list)

    print('right number of validation set:',avg_score)
    print('aus',auc_value)

    dpre.to_csv('wsi_44fujian_5class_0713_test0_model1_predict_result.csv') # 输出全片级别的预测结果

    saveproball.to_csv('wsi_44fujian_5class_0713_test0_model1_insbag_predict_result.csv')  #输出小图块的预测结果

def multi_label_f1(labels, predictions, num_classes, f1_score_average,pos_label=1):
    f1s=[]
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(num_classes):#change
        label = labels[:, c]
        prediction = predictions[:, c]
        f1_score = f1_score(label, prediction, average=f1_score_average)
        f1s.append(f1_score)
    return f1s

def multi_label_roc(labels, predictions, num_classes, pos_label=1):
    fprs = []
    tprs = []
    thresholds = []
    thresholds_optimal = []
    aucs = []
    if len(predictions.shape)==1:
        predictions = predictions[:, None]
    for c in range(num_classes):#change
        label = labels[:, c]
        prediction = predictions[:, c]
        fpr, tpr, threshold = roc_curve(label, prediction, pos_label=1)
        fpr_optimal, tpr_optimal, threshold_optimal = optimal_thresh(fpr, tpr, threshold)
        if label.sum()!=0:
            c_auc = roc_auc_score(label, prediction)
        else:
            c_auc=0
        aucs.append(c_auc)
        thresholds.append(threshold)
        thresholds_optimal.append(threshold_optimal)
    return aucs, thresholds, thresholds_optimal

def optimal_thresh(fpr, tpr, thresholds, p=0):
    loss = (fpr - tpr) - p * tpr / (fpr + tpr + 1)
    idx = np.argmin(loss, axis=0)
    return fpr[idx], tpr[idx], thresholds[idx]


if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    parser = argparse.ArgumentParser(
        description='Testing workflow includes attention computing and color map production')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of output classes')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size of feeding patches')
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--feats_size', type=int, default=512)
    parser.add_argument('--gpu_index', type=int, nargs='+', default=(0, 1), help='GPU ID(s) [0]')

    parser.add_argument('--thres', nargs='+', type=float, default=[0.884022,0.1,0.2,0.43,0.4])
    parser.add_argument('--class_name', nargs='+', type=str, default=None)


    parser.add_argument('--aggregator_weights', type=str,
                        default='1.pth')

    parser.add_argument('--embedder_weights', type=str,default = 'model.pth')

    parser.add_argument('--bag_path', type=str,default='/media/amax/f30805f9-1e6b-43a3-91da-6fffa2c90b34/Used_WSI_Tiles_100_512_CN')

    parser.add_argument('--patch_ext', type=str, default='png')
    parser.add_argument('--map_path', type=str, default='test/output')
    parser.add_argument('--export_scores', type=int, default=0)
    parser.add_argument('--score_path', type=str, default='test/score')
    args = parser.parse_args()

    gpu_ids = tuple(args.gpu_index)

    os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    if args.embedder_weights == 'ImageNet':
        print('Use ImageNet features')
        resnet = models.resnet18(pretrained=True, norm_layer=nn.BatchNorm2d)
    else:
        resnet = models.resnet18(pretrained=False, norm_layer=nn.InstanceNorm2d)
    for param in resnet.parameters():
        param.requires_grad = False
    resnet.fc = nn.Identity()
    i_classifier = mil.IClassifier(resnet, args.feats_size, output_class=args.num_classes).cuda()
    b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes).cuda()
    milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    if args.embedder_weights != 'ImageNet':
        state_dict_weights = torch.load(args.embedder_weights)
        new_state_dict = OrderedDict()
        for i in range(4):
            state_dict_weights.popitem()
        state_dict_init = i_classifier.state_dict()
        for (k, v), (k_0, v_0) in zip(state_dict_weights.items(), state_dict_init.items()):
            name = k_0
            new_state_dict[name] = v
        i_classifier.load_state_dict(new_state_dict, strict=False)

    state_dict_weights = torch.load(args.aggregator_weights)
    state_dict_weights["i_classifier.fc.weight"] = state_dict_weights["i_classifier.fc.0.weight"]
    state_dict_weights["i_classifier.fc.bias"] = state_dict_weights["i_classifier.fc.0.bias"]
    milnet.load_state_dict(state_dict_weights, strict=False)

    # df = pd.read_csv('fujian_44WSI_1024and512_CN_0401.csv') ##输入文件

    df = pd.read_csv('inputfile_id_tileaddress.csv')##输入文件
    # df=df[df['tile_prediction']>0.5779]


    os.makedirs(args.map_path, exist_ok=True)
    if args.export_scores:
        os.makedirs(args.score_path, exist_ok=True)
    if args.class_name == None:
        args.class_name = ['class {}'.format(c) for c in range(args.num_classes)]
    if len(args.thres) != args.num_classes:
        raise ValueError('Number of thresholds does not match classes.')
    test(args, df, milnet)