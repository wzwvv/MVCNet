'''
=================================================
coding:utf-8
@Time:      2025/4/21 19:22
@File:      MVCNet_CV.py
@Author:    Ziwei Wang
@Function:
=================================================
'''
import math
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import gc
import sys
from utils.data_augment import data_aug
from utils.network import backbone_net_ifnet, encoder, projector
from utils.LogRecord import LogRecord
from utils.dataloader import read_mi_within_tar_CV
from utils.utils import fix_random_seed, cal_acc_comb, data_loader_within
from models.Conformer import Conformer
from info_nce import InfoNCE
from utils.contrastive_loss import NTXentLoss, SupConLoss
from utils.alg_utils import EA
from sklearn.model_selection import KFold


import warnings
warnings.filterwarnings('ignore')
def train_target(args):
    X, y = read_mi_within_tar_CV(args)
    kf = KFold(n_splits=5, shuffle=False)  # Important, cannot shuffle for MI classification
    acc_cv = []
    for train_index, test_index in kf.split(X):
        X_src, X_tar = X[train_index], X[test_index]
        y_src, y_tar = y[train_index], y[test_index]
        if args.align:
            X_src = EA(X_src)
            X_tar = EA(X_tar)
        print("训练集大小:", X_src.shape, "测试集大小:", X_tar.shape)
        dset_loaders = data_loader_within(X_src, y_src, X_tar, y_tar, args)

        # network
        if args.backbone == 'EEGNet':
            netF, netC = backbone_net(args, return_type='xy')
        elif args.backbone == 'deep':
            netF, netC = backbone_net_deep(args, return_type='xy')
        elif args.backbone == 'shallow':
            netF, netC = backbone_net_shallow(args, return_type='xy')
        elif args.backbone == 'IFNet':
            netF, netC = backbone_net_ifnet(args, return_type='xy')
        elif args.backbone == 'FBCNet':
            netF = backbone_net_fbcnet(args, return_type='xy')
        elif args.backbone == 'ADFCNN':
            netF, netC = backbone_net_adfcnn(args, return_type='xy')
        elif args.backbone == 'Conformer':
            netF = backbone_net_conformer(args, return_type='xy')
        elif args.backbone == 'FBMSNet':
            netF, netC = backbone_net_fbmsnet(args, return_type='xy')

        if args.data_env != 'local':
            if args.backbone == 'FBCNet' or args.backbone == 'Conformer':
                netF = netF.cuda()
                base_network = netF
                optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
            else:
                netF, netC = netF.cuda(), netC.cuda()
                base_network = nn.Sequential(netF, netC)
                optimizer_f = optim.Adam(netF.parameters(), lr=args.lr)
                optimizer_c = optim.Adam(netC.parameters(), lr=args.lr)

        if args.class_num == 2:
            class_weight = torch.tensor([1., args.weight], dtype=torch.float32).cuda()  # class imbalance
            criterion = nn.CrossEntropyLoss(weight=class_weight)
        else:
            criterion = nn.CrossEntropyLoss()


        max_iter = args.max_epoch * len(dset_loaders["source"])
        interval_iter = max_iter // args.max_epoch
        args.max_iter = max_iter
        iter_num = 0
        base_network.train()

        if args.encoder == 'Transformer':
            netE = encoder(args, nhead=2, nlayer=1)  # 头数影响不大，层数1-->5(78-->77)
        elif args.encoder == 'Conformer':
            netE = Conformer(args, emb_size=40, depth=6, chn=args.chn, n_classes=args.class_num)
        netP = projector(args)
        base_network.train()

        netE, netP = netE.cuda(), netP.cuda()
        netE.train()
        netP.train()

        # contrastive loss
        """NTXentLoss: normalized temperature-scaled cross entropy loss. From SimCLR"""
        device = torch.device("cuda:0")
        scl_criterion = SupConLoss(temperature=args.Context_Cont_temperature)
        """infoNCE loss in Contrastive Predictive Coding"""
        contrastive_loss = InfoNCE(negative_mode='unpaired')  # negative_mode='unpaired' is the default value (query, positive_key, negative_keys)

        optimizer_e = optim.Adam(netE.parameters(), lr=args.lr)
        optimizer_p = optim.Adam(netP.parameters(), lr=args.lr)

        while iter_num < max_iter:
            try:
                inputs_source, labels_source = next(iter_source)
            except:
                iter_source = iter(dset_loaders["source"])
                inputs_source, labels_source = next(iter_source)

            if inputs_source.size(0) == 1:
                continue

            iter_num += 1


            if args.aug:
                # View 1
                if 'multi' in args.augmethod1:
                    flag_aug1 = [True, False, False, False, False, False]
                elif 'noise' in args.augmethod1:
                    flag_aug1 = [False, True, False, False, False, False]
                elif 'flip' in args.augmethod1:
                    flag_aug1 = [False, False, True, False, False, False]
                elif 'freq' in args.augmethod1:
                    flag_aug1 = [False, False, False, True, False, False]
                elif 'cr' in args.augmethod1:
                    flag_aug1 = [False, False, False, False, True, False]
                elif 'hs' in args.augmethod1:
                    flag_aug1 = [False, False, False, False, False, True]
                # print('flag_aug1:', flag_aug1)
                if 'hs' in args.augmethod1 or 'cr' in args.augmethod1:
                    EEGData_Train1 = np.array(inputs_source.squeeze().cpu())
                    EEGLabel_Train1 = np.array(labels_source.cpu())
                    aug_out1 = data_aug(args, EEGData_Train1, EEGLabel_Train1, EEGData_Train1.shape[1], flag_aug1)
                    aug_inputs_source1, aug_labels_source1 = aug_out1
                else:
                    EEGData_Train1 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                    EEGLabel_Train1 = np.array(labels_source.cpu())
                    aug_out1 = data_aug(args, EEGData_Train1, EEGLabel_Train1, EEGData_Train1.shape[1], flag_aug1)
                    aug_inputs_source1, aug_labels_source1 = aug_out1
                    aug_inputs_source1 = np.transpose(aug_inputs_source1, (0, 2, 1))
                if args.data_env != 'local':
                    aug_inputs_source1, aug_labels_source1 = torch.from_numpy(aug_inputs_source1).to(
                        torch.float32), torch.from_numpy(aug_labels_source1).long()
                    aug_inputs_source1 = aug_inputs_source1.cuda()
                    aug_labels_source1 = aug_labels_source1.cuda()
                if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                    aug_inputs_source1 = aug_inputs_source1.unsqueeze_(3)
                    aug_inputs_source1 = aug_inputs_source1.permute(0, 3, 1, 2)

                # View 2
                if 'multi' in args.augmethod2:
                    flag_aug2 = [True, False, False, False, False, False]
                elif 'noise' in args.augmethod2:
                    flag_aug2 = [False, True, False, False, False, False]
                elif 'flip' in args.augmethod2:
                    flag_aug2 = [False, False, True, False, False, False]
                elif 'freq' in args.augmethod2:
                    flag_aug2 = [False, False, False, True, False, False]
                elif 'cr' in args.augmethod2:
                    flag_aug2 = [False, False, False, False, True, False]
                elif 'hs' in args.augmethod2:
                    flag_aug2 = [False, False, False, False, False, True]
                # print('flag_aug2:', flag_aug2)
                if 'hs' in args.augmethod2 or 'cr' in args.augmethod2:
                    EEGData_Train2 = np.array(inputs_source.squeeze().cpu())
                    EEGLabel_Train2 = np.array(labels_source.cpu())
                    aug_out2 = data_aug(args, EEGData_Train2, EEGLabel_Train2, EEGData_Train2.shape[1], flag_aug2)
                    aug_inputs_source2, aug_labels_source2 = aug_out2
                else:
                    EEGData_Train2 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                    EEGLabel_Train2 = np.array(labels_source.cpu())
                    aug_out2 = data_aug(args, EEGData_Train2, EEGLabel_Train2, EEGData_Train2.shape[1], flag_aug2)
                    aug_inputs_source2, aug_labels_source2 = aug_out2
                    aug_inputs_source2 = np.transpose(aug_inputs_source2, (0, 2, 1))
                if args.data_env != 'local':
                    aug_inputs_source2, aug_labels_source2 = torch.from_numpy(aug_inputs_source2).to(
                        torch.float32), torch.from_numpy(aug_labels_source2).long()
                    aug_inputs_source2 = aug_inputs_source2.cuda()
                    aug_labels_source2 = aug_labels_source2.cuda()
                if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                    aug_inputs_source2 = aug_inputs_source2.unsqueeze_(3)
                    aug_inputs_source2 = aug_inputs_source2.permute(0, 3, 1, 2)

                # View 3
                if 'multi' in args.augmethod3:
                    flag_aug3 = [True, False, False, False, False, False]
                elif 'noise' in args.augmethod3:
                    flag_aug3 = [False, True, False, False, False, False]
                elif 'flip' in args.augmethod3:
                    flag_aug3 = [False, False, True, False, False, False]
                elif 'freq' in args.augmethod3:
                    flag_aug3 = [False, False, False, True, False, False]
                elif 'cr' in args.augmethod3:
                    flag_aug3 = [False, False, False, False, True, False]
                elif 'hs' in args.augmethod3:
                    flag_aug3 = [False, False, False, False, False, True]
                # print('flag_aug3:', flag_aug3)
                if 'hs' in args.augmethod3 or 'cr' in args.augmethod3:
                    EEGData_Train3 = np.array(inputs_source.squeeze().cpu())
                    EEGLabel_Train3 = np.array(labels_source.cpu())
                    aug_out3 = data_aug(args, EEGData_Train3, EEGLabel_Train3, EEGData_Train3.shape[1], flag_aug3)
                    aug_inputs_source3, aug_labels_source3 = aug_out3
                else:
                    EEGData_Train3 = np.array(inputs_source.squeeze().cpu().swapaxes(1, 2))
                    EEGLabel_Train3 = np.array(labels_source.cpu())
                    aug_out3 = data_aug(args, EEGData_Train3, EEGLabel_Train3, EEGData_Train3.shape[1], flag_aug3)
                    aug_inputs_source3, aug_labels_source3 = aug_out3
                    aug_inputs_source3 = np.transpose(aug_inputs_source3, (0, 2, 1))
                if args.data_env != 'local':
                    aug_inputs_source3, aug_labels_source3 = torch.from_numpy(aug_inputs_source3).to(
                        torch.float32), torch.from_numpy(aug_labels_source3).long()
                    aug_inputs_source3 = aug_inputs_source3.cuda()
                    aug_labels_source3 = aug_labels_source3.cuda()
                if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone:
                    aug_inputs_source3 = aug_inputs_source3.unsqueeze_(3)
                    aug_inputs_source3 = aug_inputs_source3.permute(0, 3, 1, 2)
            #
            if 'ADFCNN' in args.backbone or 'Conformer' in args.backbone:
                inputs_source = inputs_source.unsqueeze_(3)
                inputs_source = inputs_source.permute(0, 3, 1, 2)
                aug_inputs_source1 = aug_inputs_source1.unsqueeze_(3)
                aug_inputs_source1 = aug_inputs_source1.permute(0, 3, 1, 2)
                aug_inputs_source2 = aug_inputs_source2.unsqueeze_(3)
                aug_inputs_source2 = aug_inputs_source2.permute(0, 3, 1, 2)
                aug_inputs_source3 = aug_inputs_source3.unsqueeze_(3)
                aug_inputs_source3 = aug_inputs_source3.permute(0, 3, 1, 2)

            features_source, outputs_source = base_network(inputs_source)
            features_source_aug1, outputs_source_aug1 = base_network(aug_inputs_source1)
            features_source_aug2, outputs_source_aug2 = base_network(aug_inputs_source2)
            features_source_aug3, outputs_source_aug3 = base_network(aug_inputs_source3)
            classifier_loss = criterion(outputs_source, labels_source)
            classifier_loss1 = criterion(outputs_source_aug1, aug_labels_source1)
            classifier_loss2 = criterion(outputs_source_aug2, aug_labels_source2)
            classifier_loss3 = criterion(outputs_source_aug3, aug_labels_source3)
            classifier_loss = classifier_loss1 + classifier_loss2 + classifier_loss3 + classifier_loss

            if 'EEGNet' in args.backbone or 'deep' in args.backbone or 'shallow' in args.backbone or 'Conformer' in args.backbone or 'ADFCNN' in args.backbone:
                inputs_source = inputs_source.squeeze()
                aug_inputs_source1 = aug_inputs_source1.squeeze()
                aug_inputs_source2 = aug_inputs_source2.squeeze()
                aug_inputs_source3 = aug_inputs_source3.squeeze()
            if 'BNCI2014001' in args.data_name or args.data_name == 'BNCI2014002' or args.data_name == 'BNCI2015001' or args.data_name == 'Zhou2016':
                raw = inputs_source[:, :, :-1]
                aug_a1 = aug_inputs_source1[:, :, :-1]
                aug_a2 = aug_inputs_source2[:, :, :-1]
                aug_a3 = aug_inputs_source3[:, :, :-1]
            else:
                raw = inputs_source
                aug_a1 = aug_inputs_source1
                aug_a2 = aug_inputs_source2
                aug_a3 = aug_inputs_source3

            h_raw = netE(raw)
            h_a1 = netE(aug_a1)
            h_a2 = netE(aug_a2)
            h_a3 = netE(aug_a3)

            h_raw = h_raw.reshape(h_raw.shape[0], -1)
            h_a1 = h_a1.reshape(h_a1.shape[0], -1)
            h_a2 = h_a2.reshape(h_a2.shape[0], -1)
            h_a3 = h_a3.reshape(h_a3.shape[0], -1)

            z_raw = netP(h_raw)
            z_a1 = netP(h_a1)
            z_a2 = netP(h_a2)
            z_a3 = netP(h_a3)

            if features_source.shape[0] >= args.batch_size:
                b_s = args.batch_size
            else:
                b_s = features_source.shape[0]
            nt_xent_criterion_cvc = NTXentLoss(device=device, batch_size=b_s * 2,
                                               temperature=args.Context_Cont_temperature,
                                               use_cosine_similarity=args.Context_Cont_use_cosine_similarity)  # device, 256, 0.2, True TODO

            nt_xent_criterion_cmc = NTXentLoss(device=device, batch_size=b_s * 4,
                                               temperature=args.Context_Cont_temperature,
                                               use_cosine_similarity=args.Context_Cont_use_cosine_similarity)  # device, 256, 0.2, True TODO

            cvc_feas_raw = torch.cat((features_source, z_raw))
            cvc_feas_v1 = torch.cat((features_source_aug1, z_a1))
            cvc_feas_v2 = torch.cat((features_source_aug2, z_a2))
            cvc_feas_v3 = torch.cat((features_source_aug3, z_a3))
            cmc_feas_b1 = torch.cat((features_source, features_source_aug1, features_source_aug2, features_source_aug3))
            cmc_feas_b2 = torch.cat((z_raw, z_a1, z_a2, z_a3))

            # Calculate the CVC and CMC losses
            loss_cvc = (nt_xent_criterion_cvc(cvc_feas_raw, cvc_feas_v1) + nt_xent_criterion_cvc(cvc_feas_raw, cvc_feas_v2) + nt_xent_criterion_cvc(
                cvc_feas_raw, cvc_feas_v3)) / 3
            loss_cmc = nt_xent_criterion_cmc(cmc_feas_b1, cmc_feas_b2)
            classifier_loss = classifier_loss + args.lamda1 * loss_cvc + args.lamda2 * loss_cmc

            optimizer_f.zero_grad()
            optimizer_e.zero_grad()
            optimizer_p.zero_grad()
            if args.backbone != 'FBCNet' and args.backbone != 'Conformer':
                optimizer_c.zero_grad()
            classifier_loss.backward()
            optimizer_f.step()
            optimizer_e.step()
            optimizer_p.step()
            if args.backbone != 'FBCNet' and args.backbone != 'Conformer':
                optimizer_c.step()

            if iter_num % interval_iter == 0 or iter_num == max_iter:
                base_network.eval()
                netE.eval()
                netP.eval()
                acc_t_te, _ = cal_acc_comb(dset_loaders["target-online"], base_network, args=args)  # TODO target-online
                log_str = 'Task: {}, Iter:{}/{}; Acc = {:.2f}%'.format(args.task_str, int(iter_num // len(dset_loaders["source"])), int(max_iter // len(dset_loaders["source"])), acc_t_te)
                args.log.record(log_str)
                # print(log_str)

                base_network.train()
                netE.train()
                netP.train()

        print('Test Acc = {:.2f}%'.format(acc_t_te))
        acc_cv.append(acc_t_te)
        print('saving model...')

        if not os.path.exists('./runs/' + str(args.data_name) + '/'):
            os.makedirs('./runs/' + str(args.data_name) + '/')
        if args.align:
            torch.save(base_network.state_dict(),
                       './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '.ckpt')
        else:
            torch.save(base_network.state_dict(),
                       './runs/' + str(args.data_name) + '/' + str(args.backbone) + '_S' + str(args.idt) + '_seed' + str(args.SEED) + '_noEA' + '.ckpt')

        gc.collect()
        if args.data_env != 'local':
            torch.cuda.empty_cache()
    print(acc_cv)
    return np.mean(acc_cv)


if __name__ == '__main__':
    cpu_num = 8
    torch.set_num_threads(cpu_num)
    data_name_list = ['BNCI2014001']  # 'BNCI2014001', 'Zhou2016', 'MI1-7', 'BNCI2014002', 'BNCI2015001'
    dct = pd.DataFrame(columns=['dataset', 'avg', 'std', 's0', 's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13'])

    for data_name in data_name_list:
        # N: number of subjects, chn: number of channels
        weight = 1
        if data_name == 'BNCI2014001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 22, 2, 1001, 250, 144, 248, 1000, 22000  # 248 in egn, 2440 in shallow
        if data_name == 'BNCI2014002': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 14, 15, 2, 2561, 512, 100, 640, 2560, 38400  # 640 in egn, 6600 in shallow
        if data_name == 'BNCI2014004': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 3, 2, 1126, 250, 120, 280, 1126, 3378  # 280 in egn, 2760 in shallow
        if data_name == 'BNCI2015001': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 12, 13, 2, 2561, 512, 200, 640, 2560, 33280  # 640 in egn, 6600 in shallow
        if data_name == 'BNCI2014001-4': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 9, 22, 4, 1001, 250, 288, 248, 1000, 22000
        if data_name == 'MI1-7': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 7, 59, 2, 750, 250, 200, 184, 750, 44250  # 184 in egn, 1760 in shallow
        if data_name == 'BNCI2014008': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, weight = 'ERP', 8, 8, 2, 256, 256, 4200, 64, 3.5
        if data_name == 'BNCI2015003': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, weight = 'ERP', 10, 8, 2, 206, 256, 2520, 64, 9
        if data_name == 'Zhou2016': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 4, 14, 2, 1251, 250, -1, 312, 1250, 17500  # 312/3120/13800, 256 in deep
        if data_name == 'Zhou2016_3': paradigm, N, chn, class_num, time_sample_num, sample_rate, trial_num, feature_deep_dim, dim_e, dim_p = 'MI', 4, 14, 3, 1251, 250, -1, 312, 1250, 17500  # 312/3120/13800, 256 in deep
        F1, D, F2 = 4, 2, 8

        if 'BNCI2014008' in data_name:
            F1, D, F2 = 8, 4, 16
            feature_deep_dim = 128
        args = argparse.Namespace(feature_deep_dim=feature_deep_dim, trial_num=trial_num, dim_e=dim_e, dim_p=dim_p,
                                  time_sample_num=time_sample_num, sample_rate=sample_rate,
                                  N=N, chn=chn, class_num=class_num, paradigm=paradigm, data_name=data_name,
                                  F1=F1, D=D, F2=F2, weight=weight)

        args.backbone = 'IFNet'  # EEGNet, shallow, deep, FBCNet, ADFCNN, Conformer, IFNet
        args.encoder = 'Transformer'  #, Transformer
        if args.encoder == 'Conformer':
            if data_name == 'Zhou2016':
                args.dim_p = 3080
            elif data_name == 'BNCI2014002':
                args.dim_p = 6600
            elif data_name == 'BNCI2015001':
                args.dim_p = 6600
            elif data_name == 'MI1-7':
                args.dim_p = 1760

        args.method = args.backbone + '_' + data_name
        # data augmentation
        if args.backbone == 'IFNet':
            args.embed_dims = 64  # IFNet
        args.aug = True  # TODO choose augmentation or not
        args.augmethod1 = 'flip'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.augmethod2 = 'freq'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.augmethod3 = 'cr'  # TODO: flip multi freq noise cr hs (the number of augmethod can be changed)
        args.freq_method = 'shift'  # shift surr
        args.freq_mode = 0.1  # [0.1, 0.2, 0.3, 0.4, 0.5]
        args.mult_mode = 0.2  # [0.005, 0.01, 0.05, 0.1, 0.2]
        args.noise_mode = 0.25  # [0.25, 0.5, 1, 2, 4]

        # Contrastive Loss Settings
        args.Context_Cont_temperature = 0.2
        args.Context_Cont_use_cosine_similarity = True
        args.subject = False
        args.perclass = False
        args.lamda1 = float(sys.argv[2])
        args.lamda2 = float(sys.argv[3])
        args.augsettings = 'pos'
        if args.backbone == 'IFNet':
            if 'BNCI2014001' in data_name:
                args.patch_size = 125
                args.feature_deep_dim = 512
            elif data_name == 'Zhou2016':
                args.patch_size = 125
                args.feature_deep_dim = 640
            elif data_name == 'MI1-7':
                args.patch_size = 125
                args.feature_deep_dim = 384
            elif data_name == 'BNCI2015001' or data_name == 'BNCI2014002':
                args.patch_size = 128
                args.feature_deep_dim = 1280
        if args.backbone == 'ADFCNN':
            if 'BNCI2014001' in data_name:
                args.feature_deep_dim = 552
            elif data_name == 'Zhou2016':
                args.feature_deep_dim = 696
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 408
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 1440
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1440
        if args.backbone == 'FBCNet':
            if 'BNCI2014001' in data_name:
                args.nBands = 22
                args.feature_deep_dim = 192
            elif data_name == 'Zhou2016':
                args.nBands = 7
                args.feature_deep_dim = 192
            elif data_name == 'MI1-7':
                args.nBands = 2
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2014002':
                args.nBands = 3
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2015001':
                args.nBands = 3
                args.feature_deep_dim = 192
        if args.backbone == 'FBMSNet':
            if 'BNCI2014001' in data_name:
                args.in_chans = 9
                args.feature_deep_dim = 192
            elif data_name == 'Zhou2016':
                args.in_chans = 14
                args.feature_deep_dim = 36
            elif data_name == 'MI1-7':
                args.in_chans = 59
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2014002':
                args.in_chans = 3
                args.feature_deep_dim = 192
            elif data_name == 'BNCI2015001':
                args.in_chans = 3
                args.feature_deep_dim = 192
        if args.backbone == 'shallow':
            if 'BNCI2014001' in data_name:
                args.feature_deep_dim = 2440
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3120  # 312/3120/13800
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 1480  # 312/3120/13800
                args.dim_e = 640
                args.dim_p = 9600
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1480  # 312/3120/13800
                args.dim_e = 640
                args.dim_p = 8320
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1760
                # args.dim_e = 640
                # args.dim_p = 8320
        if args.backbone == 'Conformer':
            if 'BNCI2014001' in data_name:
                args.feature_deep_dim = 2440
                args.dim_p = 22000  # 2440
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3080
                args.dim_e = 220
                args.dim_p = 3080
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 6600
                args.dim_p = 6600
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 6600
                args.dim_p = 6600
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1760
                args.dim_p = 1760
        elif args.backbone == 'deep':
            if 'BNCI2014001' in data_name:
                args.feature_deep_dim = 10800
                # args.dim_e = 0
                # args.dim_p = 0
            if data_name == 'Zhou2016':
                args.feature_deep_dim = 3400
                args.dim_e = 416
                args.dim_p = 5824
            elif data_name == 'MI1-7':
                args.feature_deep_dim = 1400
                args.dim_e = 250
                args.dim_p = 14750
            elif data_name == 'BNCI2015001':
                args.feature_deep_dim = 1400
                args.dim_e = 256
                args.dim_p = 3328
            elif data_name == 'BNCI2014002':
                args.feature_deep_dim = 1400
                args.dim_e = 256
                args.dim_p = 3840
        args.projector_dim1 = args.feature_deep_dim * 4
        args.projector_dim2 = args.feature_deep_dim
        # whether to use EA
        args.align = True  # TODO
        args.dropoutRate = 0.25
        # learning rate
        args.lr = 0.001
        # train batch size
        # if args.aug:
        #     args.batch_size = 64
        # else:
        args.batch_size = 32  # TODO
        # training epochs
        args.max_epoch = 100

        # GPU device id
        try:
            device_id = str(sys.argv[1])
            os.environ["CUDA_VISIBLE_DEVICES"] = device_id
            args.data_env = 'gpu' if torch.cuda.device_count() != 0 else 'local'
        except:
            args.data_env = 'local'

        total_acc = []

        # train multiple randomly initialized models
        #for s in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]:
        for s in [1, 2, 3, 4, 5]:
            args.SEED = s

            fix_random_seed(args.SEED)
            torch.backends.cudnn.deterministic = True

            args.data = data_name
            print(args.data)
            print(args.method)
            print(args.SEED)
            print(args)

            args.local_dir = './data/' + str(data_name) + '/'
            args.result_dir = './logs/'
            my_log = LogRecord(args)
            my_log.log_init()
            my_log.record('=' * 50 + '\n' + os.path.basename(__file__) + '\n' + '=' * 50)

            sub_acc_all = np.zeros(N)
            for idt in range(N):
                args.idt = idt
                source_str = 'Except_S' + str(idt)
                target_str = 'S' + str(idt)
                args.task_str = source_str + '_2_' + target_str
                info_str = '\n========================== Transfer to ' + target_str + ' =========================='
                print(info_str)
                my_log.record(info_str)
                args.log = my_log
                if args.data_name == 'Zhou2016':
                    sbj_num = [119, 100, 100, 90]
                    args.nsamples = math.ceil(sbj_num[idt] / 2 * 0.8)  # 80%训练，20%测试
                elif args.data_name == 'BNCI2014001-4':
                    args.nsamples = math.ceil(args.trial_num / 4 * 0.8)  # 80%训练，20%测试
                else:
                    args.nsamples = math.ceil(args.trial_num / 2 * 0.8)  # 80%训练，20%测试
                sub_acc_all[idt] = train_target(args)
            print('Sub acc: ', np.round(sub_acc_all, 3))
            print('Avg acc: ', np.round(np.mean(sub_acc_all), 3))
            total_acc.append(sub_acc_all)

            acc_sub_str = str(np.round(sub_acc_all, 3).tolist())
            acc_mean_str = str(np.round(np.mean(sub_acc_all), 3).tolist())
            args.log.record("\n==========================================")
            args.log.record(acc_sub_str)
            args.log.record(acc_mean_str)

        args.log.record('\n' + '#' * 20 + 'final results' + '#' * 20)
        print(str(total_acc))
        args.log.record(str(total_acc))
        subject_mean = np.round(np.average(total_acc, axis=0), 5)
        total_mean = np.round(np.average(np.average(total_acc)), 5)
        total_std = np.round(np.std(np.average(total_acc, axis=1)), 5)

        print(subject_mean)
        print(args.method)
        print(total_mean)
        print(total_std)

        args.log.record(str(subject_mean))
        args.log.record(str(total_mean))
        args.log.record(str(total_std))

        result_dct = {'dataset': data_name, 'avg': total_mean, 'std': total_std}
        for i in range(len(subject_mean)):
            result_dct['s' + str(i)] = subject_mean[i]
