import torchvision.transforms as transforms
# from models import resnet
# import crl_utils

# import utils
import shutil
from PIL import Image
from torch.utils.data import Dataset
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR
import os.path as osp
import numpy as np
import time
import torch.nn.functional as F
import skimage.io
import ext_transforms as et
import segmentation_models_pytorch as smp
import utils
from metric.iou import IoU

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', type=str, required=True )
parser.add_argument('--gpus', type=str, required=True )
# parser.add_argument('--num_epoch', type = int, required=True)
parser.add_argument('--data_folder', type=str, required=True)
parser.add_argument('--file_name', type=str, required=True)

args = parser.parse_args()

class ISIC2017_test(Dataset):
    def __init__(self, imgdir, annodir, test_transforms = None):  
        super(ISIC2017_test, self).__init__()  
        
        self.imgdir = imgdir  
        self.annodir = annodir
        img_list = os.listdir(imgdir)
        self.labeled_img_list = []
        for name in img_list:
            if name.split('_')[-1].split('.')[0] != 'superpixels' and name.split('.')[-1] != 'csv':
                self.labeled_img_list.append(name)

        self.test_transforms = test_transforms
        
            
    def __len__(self):
        
        return len(self.labeled_img_list)
        
    def __getitem__(self, idx):
        
            
        label = Image.open( osp.join(  self.annodir , 'ISIC_' + self.labeled_img_list[idx].split('_')[1].split('.')[0]  + '_segmentation.png' ) )
        # label = torch.tensor(label)
        image = Image.open(osp.join(self.imgdir , self.labeled_img_list[idx] ))
        # image1, label1 = self.train_transforms(image, label) 
        image2, label2 = self.test_transforms( image, label )
        label2[label2 == 255] = 1
            
        return image2, label2, self.labeled_img_list[idx].split('_')[1].split('.')[0]

transform_test = et.ExtCompose([
    et.ExtResize( (192, 256) ),
    # et.ExtCenterCrop(513),
    et.ExtToTensor(),
    # transforms.Normalize((0.595, 0.297, 0.272), (0.247, 0.312, 0.299)),
])

test_trainset = ISIC2017_test(imgdir= args.data_folder + '/ISIC-2017_Test_v2_Data', annodir = args.data_folder +  '/ISIC-2017_Test_v2_Part1_GroundTruth', test_transforms = transform_test)
test_loader = torch.utils.data.DataLoader(
    test_trainset, batch_size=50, shuffle=False, num_workers=10)

from sklearn import metrics
 
def calc_aurc_eaurc(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    sort_values = sorted(zip(softmax_max[:], correctness[:]), key=lambda x:x[0], reverse=True)
    sort_softmax_max, sort_correctness = zip(*sort_values)
    risk_li, coverage_li = coverage_risk(sort_softmax_max, sort_correctness)
    aurc, eaurc = aurc_eaurc(risk_li)

    return aurc, eaurc

def coverage_risk(confidence, correctness):
    risk_list = []
    coverage_list = []
    risk = 0
    for i in range(len(confidence)):
        coverage = (i + 1) / len(confidence)
        coverage_list.append(coverage)

        if correctness[i] == 0:
            risk += 1

        risk_list.append(risk / (i + 1))

    return risk_list, coverage_list

def aurc_eaurc(risk_list):
    r = risk_list[-1]
    risk_coverage_curve_area = 0
    optimal_risk_area = r + (1 - r) * np.log(1 - r)
    for risk_value in risk_list:
        risk_coverage_curve_area += risk_value * (1 / len(risk_list))

    aurc = risk_coverage_curve_area
    eaurc = risk_coverage_curve_area - optimal_risk_area

    print("AURC {0:.2f}".format(aurc*1000))
    print("EAURC {0:.2f}".format(eaurc*1000))

    return aurc, eaurc

def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = metrics.average_precision_score(-1 * correctness + 1, 1 -1 * softmax_max)

    print("AUPR {0:.2f}".format(aupr_err*100))
    print('FPR {0:.2f}'.format(fpr_in_tpr_95*100))

    return aupr_err, fpr_in_tpr_95

def calc_ece(softmax, label, bins=15):
    bin_boundaries = torch.linspace(0, 1, bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    softmax = torch.tensor(softmax)
    labels = torch.tensor(label)

    softmax_max, predictions = torch.max(softmax, 1)
    correctness = predictions.eq(labels)

    ece = torch.zeros(1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = softmax_max.gt(bin_lower.item()) * softmax_max.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()

        if prop_in_bin.item() > 0.0:
            accuracy_in_bin = correctness[in_bin].float().mean()
            avg_confidence_in_bin = softmax_max[in_bin].mean()

            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

    print("ECE {0:.2f} ".format(ece.item()*100))

    return ece.item()

def calc_nll_brier(softmax, logit, label, label_onehot):
    brier_score = np.mean(np.sum((softmax - label_onehot) ** 2, axis=1))

    logit = torch.tensor(logit, dtype=torch.float)
    label = torch.tensor(label, dtype=torch.int)
    logsoftmax = torch.nn.LogSoftmax(dim=1)

    log_softmax = logsoftmax(logit)
    nll = calc_nll(log_softmax, label)

    print("NLL {0:.2f} ".format(nll.item()*10))
    print('Brier {0:.2f}'.format(brier_score*100))

    return nll.item(), brier_score

def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

def evaluate_new(model, data_loader):
    test_iou = IoU(2)
    model.eval()
    softm = nn.Softmax(dim = 1)
    softmax_list = []
    correct_list = []
    logit_list = []
    label_list = []

    for img, tar, iname in data_loader:

        img, tar = img.cuda(), tar.long()

        output = model(img)
        soft_out = softm(output)
        out_res = output.cpu().max(1)[1]
        test_iou.add(out_res, tar)
        
        
        softmax_list.extend(soft_out.permute([0,2,3,1]).reshape([img.size()[0]*192*256,2]).detach().cpu().numpy() )
        logit_list.extend( output.permute([0,2,3,1]).reshape([img.size()[0]*192*256,2]).detach().cpu().numpy() )
        correct_list.append( (out_res == tar).int().flatten() )
        label_list.append( tar.int().flatten() )
        
        
    softmax_list = np.array(softmax_list)
    logit_list = np.array(logit_list)
    correct_list = torch.stack(correct_list).flatten().numpy()
    label_list = torch.stack(label_list).flatten().long()

    onehot_list = torch.nn.functional.one_hot(label_list, num_classes=2 ).numpy()
    label_list = label_list.numpy()
    test_iou_value = test_iou.value()
    
    aurc, eaurc = calc_aurc_eaurc(softmax_list, correct_list)
    
    aupr_err, fpr_in_tpr_95 = calc_fpr_aupr(softmax_list, correct_list)
    
    ece = calc_ece(softmax_list, label_list, bins=15)
    
    nll, brier = calc_nll_brier(softmax_list, logit_list, label_list, onehot_list)
    
    return test_iou_value, aurc, eaurc, aupr_err, fpr_in_tpr_95, ece,nll, brier ,np.mean(correct_list)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)

    )
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load( osp.join( args.save_folder , args.file_name) ))
    model.to(device)

    result_logger = utils.Logger(os.path.join(args.save_folder , 'result.log'))

    see_res = evaluate_new(model,test_loader)

    result_logger.write([see_res[0][1], see_res[1]*1000, see_res[2]*1000, see_res[3]*100, see_res[4]*100, see_res[5]*100, see_res[6]*10, see_res[7]*100, see_res[8]])

if __name__ == "__main__":
    main()