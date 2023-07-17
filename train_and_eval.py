from model import resnet
from model import densenet_BC
from model import vgg
import torchvision.transforms as transforms
# import crl_utils
import metrics
import utils
# import train
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_folder', type=str, required=True )
parser.add_argument('--gpus', type=str, required=True )
parser.add_argument('--label_size', type = int, required=True)

args = parser.parse_args()

class cifar10_dataset_labeled(Dataset):
    def __init__(self, img_file, label_file,  train=True, train_transforms=None, test_transforms = None):  
        super(cifar10_dataset_labeled, self).__init__()  
        
        self.img_file = np.load(img_file) 
        self.label_file = np.load(label_file)

        self.train_transforms = train_transforms
        self.test_transforms = test_transforms
        self.train = train
        
            
    def __len__(self):
        
        if self.train == True:
            return args.label_size
        else:
            return 10000
        
    def __getitem__(self, idx):
        
        if self.train == True:
            
            label = self.label_file[idx]
            # label = torch.tensor(label)
            image = Image.fromarray( self.img_file[idx].reshape(3,32,32).transpose(1,2,0) )
            image = self.train_transforms(image) 
            
        else:
            label = self.label_file[idx]
            # label = torch.tensor(label)
            image = Image.fromarray( self.img_file[idx].reshape(3,32,32).transpose(1,2,0) )
            image = self.test_transforms(image)
            
        return image, label, idx

class cifar10_dataset_unlabeled(Dataset):
    def __init__(self, img_file, label_file, train_transforms=None):  
        super(cifar10_dataset_unlabeled, self).__init__()  
        
        self.img_file = np.load(img_file) 
        self.label_file = np.load(label_file)
        self.train_transforms = train_transforms
        
            
    def __len__(self):
        

        return 50000 - args.label_size
        
    def __getitem__(self, idx):
            
        label = self.label_file[idx]
        # label = torch.tensor(label)
        image = Image.fromarray( self.img_file[idx].reshape(3,32,32).transpose(1,2,0) )
        image = self.train_transforms(image) 
            
        return image, label, idx + args.label_size

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = cifar10_dataset_labeled(
    '/home/superlc117/semiseg/cifar10/cifar10_data/img_labeled_' + str(args.label_size) + '.npy', '/home/superlc117/semiseg/cifar10/cifar10_data/ann_labeled_' + str(args.label_size) + '.npy', train=True, train_transforms=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=10)

testset = cifar10_dataset_labeled(
    '/home/superlc117/semiseg/cifar10/cifar10_data/img_test.npy', '/home/superlc117/semiseg/cifar10/cifar10_data/ann_test.npy', train=False, test_transforms=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=10)

unlabeled_trainset = cifar10_dataset_unlabeled('/home/superlc117/semiseg/cifar10/cifar10_data/img_unlabeled_' + str(args.label_size) + '.npy', '/home/superlc117/semiseg/cifar10/cifar10_data/ann_unlabeled_' + str(args.label_size) + '.npy', train_transforms=transform_train)
unlabeled_loader = torch.utils.data.DataLoader(
    unlabeled_trainset, batch_size=128, shuffle=True, num_workers=10)

class History_consistent(object):
    def __init__(self, n_data):
        self.consistency = np.zeros((n_data))

        self.temp = np.zeros((n_data)) + 100
        self.max_correctness = 1

    # correctness update
    def consistency_update(self, data_idx, output):
        probs = torch.nn.functional.softmax(output, dim=1)
        confidence, classes = probs.max(dim=1)
        data_idx = data_idx.cpu().numpy()

        self.consistency[data_idx] += (classes.cpu().numpy() == self.temp[data_idx]).astype(int)
        self.temp[data_idx] = classes.cpu().numpy()

    # max correctness update
    # get target & margin
    def get_target_margin(self, data_idx1):

        out = self.consistency[data_idx1]

        return (out - np.min(out)) / ( np.max(out) - np.min(out) + 1e-10 )

class History_correct(object):
    def __init__(self, n_data):
        self.correctness = np.zeros((n_data))

        self.max_correctness = 1

    # correctness update
    def correctness_update(self, data_idx, correctness):


        data_idx = data_idx.cpu().numpy()

        self.correctness[data_idx] += correctness.cpu().numpy()

    # max correctness update
    # get target & margin
    def get_target_margin(self, data_idx1):

        out = self.correctness[data_idx1]

        return (out - np.min(out)) / ( np.max(out) - np.min(out) + 1e-10 )

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

def combine_cumu(history, labelhistory,labelloader, unlabelloader,model):

    for img, tar, idx in labelloader:
        img, tar= img.cuda(), tar.cuda()
        out = model(img)        
        history.consistency_update(idx, out)
        prec, correct = utils.accuracy(out, tar)
        labelhistory.correctness_update(idx, correct)
    
    for img, tar, idx in unlabelloader:
        img, tar= img.cuda(), tar.cuda()
        out = model(img)        
        history.consistency_update(idx, out)    



def train_function(labeledloader, unlabeledloader, model, criterion_cls, optimizer, epoch, history, labelhistory, logger):
    batch_time = utils.AverageMeter()
    data_time = utils.AverageMeter()
    total_losses = utils.AverageMeter()
    top1 = utils.AverageMeter()
    cls_losses = utils.AverageMeter()
    ranking_losses = utils.AverageMeter()
    end = time.time()

    model.train()
    combine_cumu(history,labelhistory, labeledloader, unlabeledloader, model)

    dataloader_iterator = iter(labeledloader)
    for i, (uninput, untarget, unidx) in enumerate(unlabeledloader):
        data_time.update(time.time() - end)

        try:
            input, target, idx = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(labeledloader)
            input, target, idx = next(dataloader_iterator)
            combine_cumu(history,labelhistory, labeledloader, unlabeledloader, model)



        input, target = input.cuda(), target.cuda()
        uninput = uninput.cuda()
        # compute output
        output = model(input)
        unoutput = model(uninput)

        # compute ranking target value normalize (0 ~ 1) range
        # max(softmax)

        conf = F.softmax(output, dim=1)
        confidence, _ = conf.max(dim=1)
        unconfidence,_ = F.softmax(unoutput, dim=1).max(dim=1)

        finconfidence = torch.cat((confidence,unconfidence))
        rank_target_corr = labelhistory.get_target_margin(idx)
        rank_target = history.get_target_margin(idx)
        unrank_target = history.get_target_margin(unidx)

        finrank_target = np.concatenate( (rank_target, unrank_target) )
        rank_target_corr = torch.tensor(rank_target_corr).cuda()
        finrank_target = torch.tensor(finrank_target).cuda()
        # print(rank_target)
        # ranking loss
        ranking_loss = CRL_loss_base(finrank_target, finconfidence, 1)
        rank_corr = CRL_loss_base(rank_target_corr, confidence, 1)
        # print(ranking_loss)
        # total loss
        cls_loss = criterion_cls(output, target)

        loss = cls_loss + 0.5 * ranking_loss + 0.5 * rank_corr

        # compute gradient and do optimizer step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # record loss and accuracy
        prec, correct = utils.accuracy(output, target)
        total_losses.update(loss.item(), input.size(0))
        cls_losses.update(cls_loss.item(), input.size(0))
        ranking_losses.update( 0.5 * ranking_loss.item() + 0.5 * rank_corr.item(), input.size(0))
        top1.update(prec.item(), input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i % 10 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Cls Loss {cls_loss.val:.4f} ({cls_loss.avg:.4f})\t'
                  'Rank Loss {rank_loss.val:.4f} ({rank_loss.avg:.4f})\t'
                  'Prec {top1.val:.2f}% ({top1.avg:.2f}%)'.format(
                   epoch, i, len(unlabeledloader), batch_time=batch_time,
                   data_time=data_time, loss=total_losses, cls_loss=cls_losses,
                   rank_loss=ranking_losses,top1=top1))

        # correctness count update
        # history.correctness_update(idx, correct)
    # max correctness update

    logger.write([epoch, total_losses.avg, cls_losses.avg, ranking_losses.avg, top1.avg])

def main():
    # set GPU ID
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    cudnn.benchmark = True

    # check save path
    save_path = args.save_folder
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # make dataloader
    # train_loader, test_loader = trainloader,testloader


    num_class = 10

    # set num_classes
    model_dict = {
        "num_classes": num_class,
    }

    # set model

    model = resnet.resnet110(**model_dict).cuda()


    # set criterion
    cls_criterion = nn.CrossEntropyLoss().cuda()

    # set optimizer (default:sgd)
    optimizer = optim.SGD(model.parameters(),
                          lr=0.1,
                          momentum=0.9,
                          weight_decay=0.0001,
                          nesterov=False)

    # set scheduler
    scheduler = MultiStepLR(optimizer,
                            milestones=[150,250],
                            gamma=0.1)

    # make logger
    train_logger = utils.Logger(os.path.join(save_path, 'train.log'))
    result_logger = utils.Logger(os.path.join(save_path, 'result.log'))

    # make History Class
    correctness_history = History_correct(args.label_size)
    
    consistency_history = History_consistent(50000)

    # start Train
    for epoch in range(1, 300 + 1):
        scheduler.step()
        train_function(trainloader,
                    unlabeled_loader,
                    model,
                    cls_criterion,
                    optimizer, 
                    epoch,
                    consistency_history,
                    correctness_history,
                    train_logger,
                    )

        if epoch % 30 == 0:
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'model_' + str(epoch) + '.pth'))            
            acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics(testloader, model)
            print('acc: ')
            print(acc)
        # save model
        if epoch == 300:
            torch.save(model.state_dict(),
                       os.path.join(save_path, 'model.pth'))
    # finish train

    # calc measure
    acc, aurc, eaurc, aupr, fpr, ece, nll, brier = calc_metrics(testloader, model)
    print('acc: ')
    print(acc)
    # result write
    result_logger.write([acc, aurc*1000, eaurc*1000, aupr*100, fpr*100, ece*100, nll*10, brier*100])


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

# AUPR ERROR
def calc_fpr_aupr(softmax, correct):
    softmax = np.array(softmax)
    correctness = np.array(correct)
    softmax_max = np.max(softmax, 1)

    fpr, tpr, thresholds = metrics.roc_curve(correctness, softmax_max)
    idx_tpr_95 = np.argmin(np.abs(tpr - 0.95))
    fpr_in_tpr_95 = fpr[idx_tpr_95]

    aupr_err = metrics.average_precision_score(-1 * correctness + 1, -1 * softmax_max)

    print("AUPR {0:.2f}".format(aupr_err*100))
    print('FPR {0:.2f}'.format(fpr_in_tpr_95*100))

    return aupr_err, fpr_in_tpr_95

# ECE
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

# NLL & Brier Score
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

# Calc NLL
def calc_nll(log_softmax, label):
    out = torch.zeros_like(label, dtype=torch.float)
    for i in range(len(label)):
        out[i] = log_softmax[i][label[i]]

    return -out.sum()/len(out)

# Calc coverage, risk
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

# Calc aurc, eaurc
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


def calc_metrics(data_loader, model):
    model.eval()

    list_softmax = []
    list_correct = []
    list_logit = []
    label_list = []
    list_onehot = []
    with torch.no_grad():
        for inputs, targets,idx_list in data_loader:
            inputs, targets = inputs.cuda(), targets
            label_list.extend(targets)
            list_onehot.extend( F.one_hot(targets, num_classes=10).data.numpy() )
            outputs = model(inputs)
            list_softmax.extend(F.softmax(outputs).cpu().data.numpy())
            pred = outputs.data.max(1, keepdim=True)[1]
            for i in outputs:
                list_logit.append(i.cpu().data.numpy())
            for j in range(len(pred)):
                if pred[j] == targets[j]:
                    cor = 1
                else:
                    cor = 0
                list_correct.append(cor)
    list_onehot = np.array(list_onehot)
    aurc, eaurc = calc_aurc_eaurc(list_softmax, list_correct)
    aupr, fpr = calc_fpr_aupr(list_softmax, list_correct)
    ece = calc_ece(list_softmax, label_list, bins=15)
    nll, brier = calc_nll_brier(list_softmax, list_logit, label_list, list_onehot)
    return np.mean(list_correct),aurc, eaurc, aupr, fpr, ece, nll, brier


if __name__ == "__main__":
    main()