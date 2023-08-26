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
parser.add_argument('--label_size', type = int, required=True)
parser.add_argument('--con_weight', type = float, required=True)
parser.add_argument('--data_folder', type = str, required=True)
# parser.add_argument('--num_epoch', type = int, required=True)

args = parser.parse_args()

class ISIC2017_labeled(Dataset):
    def __init__(self, imglist, imgdir, annodir, train_transforms=None, test_transforms = None):  
        super(ISIC2017_labeled, self).__init__()  
        
        self.imgdir = imgdir  
        self.annodir = annodir
        # img_list = os.listdir(imgdir)
        self.labeled_img_list = np.load(imglist)
        # for name in img_list:
        #     if name.split('_')[-1].split('.')[0] != 'superpixels' and name.split('.')[-1] != 'csv':
        #         self.labeled_img_list.append(name)
        

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        
            
    def __len__(self):
        
        return len(self.labeled_img_list)
        
    def __getitem__(self, idx):
        
            
        label = Image.open( osp.join(  self.annodir , 'ISIC_' + self.labeled_img_list[idx].split('_')[1].split('.')[0]  + '_segmentation.png' ) )
        # label = torch.tensor(label)
        image = Image.open(osp.join(self.imgdir , self.labeled_img_list[idx] ))
        image1, label1 = self.train_transforms(image, label) 
        image2, label2 = self.test_transforms( image, label )
        label1[label1 == 255] = 1
        label2[label2 == 255] = 1
            
        return image1, label1, image2, label2, self.labeled_img_list[idx].split('_')[1].split('.')[0]

class ISIC2017_unlabeled(Dataset):
    def __init__(self, imglist,imgdir, annodir, test_transforms = None):  
        super(ISIC2017_unlabeled, self).__init__()  
        
        self.imgdir = imgdir  
        self.annodir = annodir
        # img_list = os.listdir(imgdir)
        self.labeled_img_list = np.load(imglist)
        # for name in img_list:
        #     if name.split('_')[-1].split('.')[0] != 'superpixels' and name.split('.')[-1] != 'csv':
        #         self.labeled_img_list.append(name)

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

train_transform = et.ExtCompose([
    et.ExtResize(size=(192,256)),
    et.ExtRandomScale((0.7, 1.5)),
    et.ExtRandomCrop(size=(192,256), pad_if_needed=True),
    et.ExtRandomHorizontalFlip(),
    et.ExtRandomVerticalFlip(),
    et.ExtToTensor(),
    # et.ExtNormalize(mean=[0.595, 0.297, 0.272],
    #                 std=[0.247, 0.312, 0.299]),
])

transform_test = et.ExtCompose([
    et.ExtResize( (192, 256) ),
    # et.ExtCenterCrop(513),
    et.ExtToTensor(),
    # transforms.Normalize((0.595, 0.297, 0.272), (0.247, 0.312, 0.299)),
])

trainset = ISIC2017_labeled( imglist= 'ISIC2017_datalist/labeled_' + str(args.label_size) + '_list.npy',
    imgdir= args.data_folder + '/ISIC-2017_Training_Data', annodir =  args.data_folder + '/ISIC-2017_Training_Part1_GroundTruth', train_transforms=train_transform, test_transforms = transform_test )
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=32, shuffle=True, num_workers=10)

unlabeled_trainset = ISIC2017_unlabeled(imglist= 'ISIC2017_datalist/unlabeled_' + str(args.label_size) + '_list.npy', imgdir= args.data_folder + '/ISIC-2017_Training_Data', annodir = args.data_folder + '/ISIC-2017_Training_Part1_GroundTruth', test_transforms = transform_test)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_trainset, batch_size=64, shuffle=True, num_workers=10)

val_trainset = ISIC2017_test(imgdir= args.data_folder + '/ISIC-2017_Validation_Data', annodir = args.data_folder +  '/ISIC-2017_Validation_Part1_GroundTruth', test_transforms = transform_test)
val_loader = torch.utils.data.DataLoader(
    val_trainset, batch_size=50, shuffle=False, num_workers=10)    

def val_fun(loader, model):
    test_iou = IoU(2)
    model.eval()
    for img, tar, iname in loader:
        img, tar = img.cuda(), tar.long()
        output = model(img)
        out_res = output.cpu().max(1)[1]

        test_iou.add(out_res, tar)
    test_iou_value = test_iou.value()
    return test_iou_value


def CRL_loss_base(con, score, iter_i):
    # print(len(con))
    permi= torch.randperm(len(con))
    all_CRL_loss = 0

    con = con[ permi ]
    score = score[ permi ]    
    for i in range(iter_i):
        
        
        
        con1 = torch.roll(con, i + 1)
        score1 = torch.roll(score, i + 1)
        # print(score1 == score)
        for_see = torch.nn.functional.relu(-torch.sign(con1 - con) * 
                                                    (score1 - score) + torch.abs( con1 - con ))
        
        all_CRL_loss = all_CRL_loss + torch.sum(for_see) / len(con)

    return all_CRL_loss/iter_i

def rank_loss_function_correct(model, image, target, iname, device):

    model.eval()

    image, target = image.to(device), target.int()
        
    output = model(image)

    softm = nn.Softmax(dim = 1)
    soft_out = softm(output)

    correct_batch = []

    for k in range(len(iname)):

        correct_batch.append( correct_change[iname[k]] )

    correct_batch = torch.stack(correct_batch)
    correct_batch = correct_batch.flatten()

    correct_batch = (correct_batch - torch.min(correct_batch) ) / (torch.max(correct_batch) - torch.min(correct_batch)+ 1e-7)

    correct_batch = correct_batch.to(device)
    score_batch = soft_out.max(1)[0]
    score_batch = score_batch.flatten()

    return CRL_loss_base(correct_batch, score_batch, 1)

def rank_loss_function_constant(model, image, target, iname, device):
    model.eval()

    image, target = image.to(device), target.int()
        
    output = model(image)

    softm = nn.Softmax(dim = 1)
    soft_out = softm(output)
    constant_batch = []
    for k in range(len(iname)):

        constant_batch.append( constant_change[iname[k]] )

    constant_batch = torch.stack(constant_batch)
    constant_batch = constant_batch.flatten()

    constant_batch = (constant_batch - torch.min(constant_batch) ) / (torch.max(constant_batch) - torch.min(constant_batch) + 1e-7)

    constant_batch = constant_batch.to(device)

    score_batch = soft_out.max(1)[0]

    score_batch = score_batch.flatten()
    return CRL_loss_base(constant_batch, score_batch, 1)


def constant_for_dataset(model, data_loader_label, data_loader_unlabel, epoch, device):
    
    model.eval()
    for _,_, img, tar, iname in data_loader_label:
        img, tar = img.to(device), tar.int()
        
        output = model(img)
        
        out_res = output.cpu().max(1)[1]
        correct_or = (out_res == tar).int()

        if epoch == 0:
            for k in range(len(iname)):
                correct_change[iname[k]] = correct_or[k]
                constant_change[iname[k]] = torch.zeros(tar[k].size()).int()
                out_temp[iname[k]] = out_res[k]
        else:
            for k in range(len(iname)):
                constant_or = ( out_temp[iname[k]] == out_res[k] ).int()
                constant_change[iname[k]] = constant_change[iname[k]] + constant_or
                correct_change[iname[k]] = correct_change[iname[k]] + correct_or[k]
                out_temp[iname[k]] = out_res[k]    

    for img, tar, iname in data_loader_unlabel:
        img, tar = img.to(device), tar.int()
        
        output = model(img)
        
        out_res = output.cpu().max(1)[1]

        if epoch == 0:
            for k in range(len(iname)):
                constant_change[iname[k]] = torch.zeros(tar[k].size()).int()
                out_temp[iname[k]] = out_res[k]
        else:
            for k in range(len(iname)):
                constant_or = ( out_temp[iname[k]] == out_res[k] ).int()
                constant_change[iname[k]] = constant_change[iname[k]] + constant_or
                out_temp[iname[k]] = out_res[k]    

constant_change = {}
out_temp = {}
correct_change = {}

def main():

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus

    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)
    batch_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnet34",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,                      # model output channels (number of classes in your dataset)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train_logger = utils.Logger(os.path.join(args.save_folder, 'train.log'))
    criterion = nn.CrossEntropyLoss(reduction='mean')
    model = nn.DataParallel(model)
    model.to(device)

    dataloader_iterator = iter(trainloader)
    end = time.time()
    best_iou_100 = 0
    best_iou_150 = 0
    for epoch in range(200):


        interval_loss = 0
        count_k = 0

        constant_for_dataset(model, trainloader, unlabeled_loader,epoch, device)

        model.train()
        jif = 0
        for (images2, labels2,iname2) in unlabeled_loader:

            try:
                images, labels, images1, labels1, iname = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(trainloader)
                images, labels, images1, labels1, iname = next(dataloader_iterator)
                # print('start cumu: ')
                constant_for_dataset(model, trainloader, unlabeled_loader,epoch, device)

            images = images.to(device, dtype=torch.float32)
            labels = labels.to(device, dtype=torch.long)
            img_rank = torch.concat([ images1, images2 ])
            label_rank = torch.concat([ labels1, labels2 ])
            iname_rank = np.concatenate( [iname, iname2] )
            optimizer.zero_grad()
            outputs = model(images)

            correct_loss = rank_loss_function_correct(model, images1, labels1, iname, device)
            constant_loss = rank_loss_function_constant(model, img_rank, label_rank, iname_rank, device)
            cls_loss = criterion(outputs, labels)
            loss = cls_loss + 0.5 * correct_loss + args.con_weight * constant_loss

            loss.backward()
            optimizer.step()
            np_loss = loss.item()

            interval_loss += np_loss
            count_k = count_k + 1

            total_losses.update(loss.item(), images.size(0))
            cls_losses.update(cls_loss.item(), images.size(0))
            ranking_losses.update(0.5 * correct_loss.item() + args.con_weight * constant_loss.item(), images2.size(0))

            if (jif+1) % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                    'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                    'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                    'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'.format(
                    epoch, jif, len(unlabeled_loader), batch_time=batch_time,
                    loss=total_losses, cls_loss=cls_losses,
                    rank_loss=ranking_losses))

            jif = jif + 1
            train_logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg])
            batch_time.update(time.time() - end)
            end = time.time()


        if epoch >= 100:
            cur_iou = val_fun(val_loader, model)
            if cur_iou[1] >  best_iou_100:
                torch.save(model.state_dict(),
                            os.path.join(args.save_folder, 'model_best.pth'))  
                best_iou_100 = cur_iou[1]
        # if epoch >= 150:
        #     if cur_iou[1] >  best_iou_150:
        #         torch.save(model.state_dict(),
        #                     os.path.join(args.save_folder, 'model_best_150.pth'))  
        #         best_iou_150 = cur_iou[1]


    torch.save(model.state_dict(),
                os.path.join(args.save_folder, 'model_'+ str(epoch) + '.pth'))

if __name__ == "__main__":
    main()