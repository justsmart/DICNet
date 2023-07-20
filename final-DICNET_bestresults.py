# -*- coding: utf-8 -*-

from __future__ import print_function, division
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import numpy as np
from sklearn.cluster import KMeans
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
from model import DICNet
from sklearn.preprocessing import StandardScaler, MinMaxScaler, normalize, scale
import scipy.io
import h5py
import math
import copy
from loss import Loss
# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="5"

from measure import *


def wmse_loss(input, target, weight, reduction='mean'):
    ret = (torch.diag(weight).mm(target - input)) ** 2
    ret = torch.mean(ret)
    return ret


def do_metric(y_prob, label):
    y_predict = y_prob > 0.5
    ranking_loss = 1 - compute_ranking_loss(y_prob, label)
    # print(ranking_loss)
    one_error = compute_one_error(y_prob, label)
    # print(one_error)
    coverage = compute_coverage(y_prob, label)
    # print(coverage)
    hamming_loss = 1 - compute_hamming_loss(y_predict, label)
    # print(hamming_loss)
    precision = compute_average_precision(y_prob, label)
    # print(precision)
    macro_f1 = compute_macro_f1(y_predict, label)
    # print(macro_f1)
    micro_f1 = compute_micro_f1(y_predict, label)
    # print(micro_f1)
    auc = compute_auc(y_prob, label)
    auc_me = mlc_auc(y_prob, label)
    return np.array([hamming_loss, one_error, coverage, ranking_loss, precision, auc, auc_me, macro_f1, micro_f1])


def train_DIC(mul_X, mul_X_val, WE,WE_val,yv_label, device,args):
    # return None, torch.randn(9, 1)
    model = DICNet(
        n_stacks=4,
        n_input=args.n_input,
        n_z=args.Nlabel,
        Nlabel=args.Nlabel).to(device)
    loss_model = Loss(args.alpha, device)
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Module):
            for mm in m.modules():

                if isinstance(mm, nn.Linear):
                    nn.init.xavier_uniform_(mm.weight)
                    nn.init.constant_(mm.bias, 0.0)
    num_X = mul_X[0].shape[0]
    num_X_val = mul_X_val[0].shape[0]
    print(num_X, num_X_val)
    optimizer = SGD(model.parameters(), lr=args.lrkl, momentum=args.momentumkl)
    # optimizer = Adam(model.parameters(), lr=args.lrkl)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    total_loss = 0
    ytest_Lab = np.zeros([mul_X_val[0].shape[0], args.Nlabel])
    ap_loss = []
    best_value_result = [0] * 10
    best_value_epoch = 0
    best_train_model = copy.deepcopy(model)
    for epoch in range(int(args.maxiter)):
        model.train()
        total_loss_last = total_loss
        total_loss = 0
        ytest_Lab_last = np.copy(ytest_Lab)
        index_array = np.arange(num_X)
        if args.AE_shuffle == True:
            np.random.shuffle(index_array)
        for batch_idx in range(int(np.ceil(num_X / args.batch_size))):
            idx = index_array[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X)]
            mul_X_batch = []
            for iv, X in enumerate(mul_X):
                mul_X_batch.append(X[idx].to(device))

            we = WE[idx].to(device)
            sub_target = Inc_label[idx].to(device)
            fan_sub_target = fan_Inc_label[idx].to(device)
            sub_obrT = obrT[idx].to(device)
            optimizer.zero_grad()

            x_bar_list, target_pre, fusion_z, individual_zs = model(mul_X_batch, we)

            loss_Cont = 0
            for i in range(len(individual_zs)):
                for j in range(i + 1, len(individual_zs)):
                    loss_Cont += loss_model.contrast_loss(individual_zs[i], individual_zs[j], we[:, i], we[:, j])


            loss_CL = torch.mean(torch.abs((sub_target.mul(torch.log(target_pre + 1e-10)) \
                                            + fan_sub_target.mul(torch.log(1 - target_pre + 1e-10))).mul(sub_obrT)))
            loss_AE = 0
            for iv, x_bar in enumerate(x_bar_list):
                loss_AE += wmse_loss(x_bar, mul_X_batch[iv], we[:, iv])
            fusion_loss = loss_CL + args.gamma * loss_AE  + loss_Cont * args.beta
            # print('all:',fusion_loss.item())
            total_loss += fusion_loss.item()
            fusion_loss.backward()
            optimizer.step()
        # scheduler.step()
        yp_prob = test_DIC(model, mul_X_val, WE_val, args, device)
        # yp_prob = np.delete(yp_prob, ind_00, axis=0)

        value_result = do_metric(yp_prob, yv_label)
        ap_loss.append([value_result[4],total_loss])
        total_loss = total_loss / (batch_idx + 1)
        print("semi_epoch {} loss={:.4f} hamming loss={:.4f} AP={:.4f} AUC={:.4f} auc_me={:.4f}"
              .format(epoch, total_loss, value_result[0], value_result[4], value_result[5], value_result[6]))
        if best_value_result[4] < value_result[4]:
            best_value_result = value_result
            best_train_model = copy.deepcopy(model)
            best_value_epoch = epoch

            # torch.save(model)
        ytest_Lab = yp_prob > 0.5
        del yp_prob
        delta_y = np.sum(ytest_Lab != ytest_Lab_last).astype(np.float32) / ytest_Lab.shape[0] / ytest_Lab.shape[1]
        if epoch > 100 and ( (best_value_result[4]-value_result[4]>0.03) or
                best_value_result[4] < args.min_AP or (abs(total_loss_last - total_loss) < 1e-5 or delta_y < args.tol)):
            print('Training stopped: epoch=%d, best_epoch=%d, best_AP=%.7f, min_AP=%.7f,total_loss=%.7f' % (
                epoch, best_value_epoch, best_value_result[4], args.min_AP, total_loss))
            break

    return best_train_model, best_value_result,ap_loss


def test_DIC(model, mul_X_test, WE_test, args, device):
    model.eval()
    num_X_test = mul_X_test[0].shape[0]
    tmp_q = torch.zeros([num_X_test, args.Nlabel]).to(device)
    index_array_test = np.arange(num_X_test)
    for batch_idx in range(int(np.ceil(num_X_test / args.batch_size))):
        idx = index_array_test[batch_idx * args.batch_size: min((batch_idx + 1) * args.batch_size, num_X_test)]
        mul_X_test_batch = []
        for iv, X in enumerate(mul_X_test):
            mul_X_test_batch.append(X[idx].to(device))

        we = WE_test[idx].to(device)
        _, linshi_q, _, _ = model(mul_X_test_batch, we)
        tmp_q[idx] = linshi_q
        del linshi_q
    # del x0,x1,x2,x3,x4,x5,we  
    # update target distribution p
    yy_pred = tmp_q.data.cpu().numpy()
    yy_pred = np.nan_to_num(yy_pred)
    return yy_pred


def filterparam(file_path):
    params = []
    if os.path.exists(file_path):
        file_handle = open(file_path, mode='r')
        lines = file_handle.readlines()
        lines = lines[1:] if len(lines)>1 else []
        params = [(float(line.split(' ')[-5]),float(line.split(' ')[-3]), float(line.split(' ')[-2]), float(line.split(' ')[-1])) for line in lines]
    return params


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--Nlabel', default=7, type=int)
    parser.add_argument('--maxiter', default=300, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--dataset', type=str, default='corel5k_six_view')
    parser.add_argument('--dataPath', type=str, default='data/corel5k')
    # parser.add_argument('--pretrain_path_basis', type=str, default='pascal07/iaprtc_six_view')
    parser.add_argument('--MaskRatios', type=float, default=0.5)
    parser.add_argument('--LabelMaskRatio', type=float, default=0.5)
    parser.add_argument('--TraindataRatio', type=float, default=0.7)
    parser.add_argument('--AE_shuffle', type=bool, default=True)
    parser.add_argument('--min_AP', default=0., type=float)
    parser.add_argument('--tol', default=1e-7, type=float)
    args = parser.parse_args()

    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    file_path = 'final-DIC-' + args.dataset + '_BS_' + str(args.batch_size) + '_VMR_' + str(
        args.MaskRatios) + '_LMR_' + str(args.LabelMaskRatio) + '_TR_' + str(
        args.TraindataRatio) + '-best_AP' + '.txt'
    existed_params = filterparam(file_path)

    device = torch.device("cuda" if args.cuda else "cpu")
    Pre_fnum = 5
    pre_momae = [0.9]  # [0.98,0.9,0.95]
    pre_lrkl = [0.1]  # [0.007,0.01,0.0005,0.05,0.1,0.005,0.001]
    pre_alpha = [0.5] # alpha is the tau in the paper
    pre_beta = [1e-3] #1e-3 is recommended for contrastive loss on all datasets
    pre_gamma = [1e-1]  # 
    best_AUC_me = 0
    best_AUC_mac = 0
    best_AP = 0

    data = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '.mat')

    X = data['X'][0]

    view_num = X.shape[0]
    label = data['label']
    label = np.array(label, 'float32')
    for momae in pre_momae:
        args.momentumkl = momae
        for lrkl in pre_lrkl:
            args.lrkl = lrkl
            for gamma in pre_gamma:
                args.gamma = gamma
                for alpha in pre_alpha:
                    args.alpha = alpha
                    for beta in pre_beta:
                        args.beta = beta

                        if args.lrkl >= 0.01:
                            args.momentumkl = 0.90

                        if (args.lrkl,args.alpha,args.beta,args.gamma) in existed_params:
                            print('existed param! lr:{} alpha:{} beta:{} gamma:{}'.format(args.lrkl,args.alpha,args.beta,args.gamma))
                            continue
                        print(args)
                        hm_loss = np.zeros(Pre_fnum)
                        one_error = np.zeros(Pre_fnum)
                        coverage = np.zeros(Pre_fnum)
                        rk_loss = np.zeros(Pre_fnum)
                        AP_score = np.zeros(Pre_fnum)

                        mac_auc = np.zeros(Pre_fnum)
                        auc_me = np.zeros(Pre_fnum)
                        mac_f1 = np.zeros(Pre_fnum)
                        mic_f1 = np.zeros(Pre_fnum)

                        for fnum in range(Pre_fnum):
                            mul_X = [None] * view_num

                            datafold = scipy.io.loadmat(args.dataPath + '/' + args.dataset + '_MaskRatios_' + str(
                                args.MaskRatios) + '_LabelMaskRatio_' +
                                                        str(args.LabelMaskRatio) + '_TraindataRatio_' + str(
                                args.TraindataRatio) + '.mat')
                            folds_data = datafold['folds_data']
                            folds_label = datafold['folds_label']
                            folds_sample_index = datafold['folds_sample_index']
                            del datafold
                            Ndata, args.Nlabel = label.shape
                            # training data, val data and test data
                            indexperm = np.array(folds_sample_index[0, fnum], 'int32')
                            train_num = math.ceil(Ndata * args.TraindataRatio)
                            # train_index = indexperm[0,0:train_num]-1   #matlab generates the index from '1' to 'Nsample', but python needs from '0' to 'Nsample-1'
                            remain_num = Ndata-train_num
                            val_num = math.ceil(remain_num*0.5)
                            test_index = indexperm[0, train_num:indexperm.shape[1]] - 1
                            print('val_num',val_num)
                            print('remain_index',len(test_index))
                            val_index = indexperm[0, train_num:train_num+val_num] - 1
                            print('val_index',len(val_index))
                            rtest_index = indexperm[0, train_num+val_num:indexperm.shape[1]] - 1
                            print('rtest_index',len(rtest_index))
                            # incomplete data index    
                            WE = np.array(folds_data[0, fnum], 'int32')
                            # incomplete label construction
                            obrT = np.array(folds_label[0, fnum], 'int32')  # incomplete label index

                            if label.min() == -1:
                                label = (label + 1) * 0.5
                            Inc_label = label * obrT  # incomplete label matrix
                            fan_Inc_label = 1 - Inc_label
                            # incomplete data construction 
                            for iv in range(view_num):
                                mul_X[iv] = np.copy(X[iv])
                                mul_X[iv] = mul_X[iv].astype(np.float32)
                                WEiv = WE[:, iv]
                                ind_1 = np.where(WEiv == 1)
                                ind_1 = (np.array(ind_1)).reshape(-1)
                                ind_0 = np.where(WEiv == 0)
                                ind_0 = (np.array(ind_0)).reshape(-1)
                                mul_X[iv][ind_1, :] = StandardScaler().fit_transform(mul_X[iv][ind_1, :])
                                mul_X[iv][ind_0, :] = 0
                                clum = abs(mul_X[iv]).sum(0)
                                ind_11 = np.array(np.where(clum != 0)).reshape(-1)
                                new_X = np.copy(mul_X[iv][:, ind_11])

                                mul_X[iv] = torch.Tensor(np.nan_to_num(np.copy(new_X)))
                                del new_X, ind_0, ind_1, ind_11, clum
                            
                            WE = torch.Tensor(WE)
                            mul_X_test = [xiv[test_index] for xiv in mul_X]
                            mul_X_val = [xiv[val_index] for xiv in mul_X]
                            mul_X_rtest = [xiv[rtest_index] for xiv in mul_X]
                            WE_test = WE[test_index]
                            WE_val = WE[val_index]
                            WE_rtest = WE[rtest_index]
                            obrT = torch.Tensor(obrT)
                            Inc_label = torch.Tensor(Inc_label)
                            fan_Inc_label = torch.Tensor(fan_Inc_label)
                            # args.n_input = [X0.shape[1],X1.shape[1],X2.shape[1],X3.shape[1],X4.shape[1],X5.shape[1]]
                            args.n_input = [xiv.shape[1] for xiv in mul_X]
                            yv_label = np.copy(label[val_index])
                            yt_label = np.copy(label[test_index])
                            yrt_label = np.copy(label[rtest_index])
                            # clum = abs(yv_label).sum(1)
                            # ind_00 = np.array(np.where(clum == 0)).reshape(-1)
                            # yv_label = np.delete(yv_label, ind_00, axis=0)
                            # clum = abs(yrt_label).sum(1)
                            # ind_00 = np.array(np.where(clum == 0)).reshape(-1)
                            # yrt_label = np.delete(yrt_label, ind_00, axis=0)

                            model, _,ap_loss = train_DIC(mul_X, mul_X_val, WE, WE_val, yv_label, device,args)
                            yp_prob = test_DIC(model, mul_X_rtest, WE_rtest, args, device)
                            #test in the test set
                            # yp_prob = np.delete(yp_prob, ind_00, axis=0)
                            value_result = do_metric(yp_prob, yrt_label)
                            
                            

                            print(
                                "final:hamming-loss" + ' ' + "one-error" + ' ' + "coverage" + ' ' + "ranking-loss" + ' ' + "average-precision" + ' ' + "macro-auc" + ' ' + "auc_me" + ' ' + "macro_f1" + ' ' + "micro_f1")
                            print(value_result)

                            hm_loss[fnum] = value_result[0]
                            one_error[fnum] = value_result[1]
                            coverage[fnum] = value_result[2]
                            rk_loss[fnum] = value_result[3]
                            AP_score[fnum] = value_result[4]
                            mac_auc[fnum] = value_result[5]
                            auc_me[fnum] = value_result[6]
                            mac_f1[fnum] = value_result[7]
                            mic_f1[fnum] = value_result[8]
                        if AP_score.mean() > best_AP:
                            best_AP = AP_score.mean()

                        file_handle = open(file_path, mode='a')
                        if os.path.getsize(file_path) == 0:
                            file_handle.write(
                                'mean_AP std_AP mean_hamming_loss std_hamming_loss mean_one_error std_one_error mean_coverage std_coverage mean_ranking_loss std_ranking_loss mean_AUC std_AUC mean_AUCme std_AUCme mean_macro_f1 std_macro_f1 mean_micro_f1 std_micro_f1 lrkl momentumKL alphakl betakl gammakl\n')

                        file_handle.write(str(AP_score.mean()) + ' ' +
                                          str(AP_score.std()) + ' ' +
                                          str(hm_loss.mean()) + ' ' +
                                          str(hm_loss.std()) + ' ' +
                                          str(one_error.mean()) + ' ' +
                                          str(one_error.std()) + ' ' +
                                          str(coverage.mean()) + ' ' +
                                          str(coverage.std()) + ' ' +
                                          str(rk_loss.mean()) + ' ' +
                                          str(rk_loss.std()) + ' ' +
                                          str(mac_auc.mean()) + ' ' +
                                          str(mac_auc.std()) + ' ' +
                                          str(auc_me.mean()) + ' ' +
                                          str(auc_me.std()) + ' ' +
                                          str(mac_f1.mean()) + ' ' +
                                          str(mac_f1.std()) + ' ' +
                                          str(mic_f1.mean()) + ' ' +
                                          str(mic_f1.std()) + ' ' +
                                          str(args.lrkl) + ' ' +
                                          str(args.momentumkl) + ' ' +
                                          str(args.alpha) + ' ' +
                                          str(args.beta) + ' ' +
                                          str(args.gamma)
                                          )

                        file_handle.write('\n')
                        file_handle.close()
