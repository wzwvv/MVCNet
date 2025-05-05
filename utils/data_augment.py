# -*- coding: utf-8 -*-
# @Time    : 2021/6/25 21:01
# @Author  : wenzhang
# @File    : data_augment.py
import random

import numpy as np
import torch
from scipy.signal import hilbert
from braindecode.augmentation import FTSurrogate, SmoothTimeMask, ChannelsDropout, FrequencyShift

def CD(args, X):
    transform = ChannelsDropout(
        probability=1.,  # defines the probability of actually modifying the input
        p_drop=0.2
    )
    X_tr, _ = transform.operation(torch.as_tensor(X).float(), None, 0.2)  # drop prob
    return X_tr.numpy()

def freqShift(args, X):
    transform = FrequencyShift(
        probability=1.,  # defines the probability of actually modifying the input
        sfreq=args.sample_rate,
        max_delta_freq=2.  # the frequency shifts are sampled now between -2 and 2 Hz
    )
    X_tr, _ = transform.operation(torch.as_tensor(X).float(), None, 2., args.sample_rate)  # shift 10Hz
    return X_tr.numpy()

def freqSur(args, X):
    transform = FTSurrogate(
        probability=1.,  # defines the probability of actually modifying the input
    )
    X_tr, _ = transform.operation(torch.as_tensor(X).float(), None, 0.5, False)  # 空间信息很重要时，设为false，0-1之间相位扰动
    return X_tr.numpy()


def Augmentation_left_right_half_sample_label_perclass(X, y, left_mat, right_mat, middle_mat):
    """
    初级，aug时不考虑标签；之后aug时将标签考虑进去
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    num_samples, num_channels, num_timesamples = X.shape
    llist = [i for i in y if i == 0]
    rlist = [i for i in y if i == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedL2R = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2L = np.zeros((len(rlist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        kr = random.choice([ele for ele in rlen])
        LL2R = np.concatenate((Xl_left[i, :, :], Xl_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        LR2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        LL2R = np.take(LL2R, real_list, axis=-2)  # channel 维度重排序 1
        LR2L = np.take(LR2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = LR2L
        transformedL2R[i, :, :] = LL2R
    for i in range(len(rlist)):
        kl = random.choice([ele for ele in llen])
        kr = random.choice([ele for ele in rlen if ele != i])
        RL2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        RR2L = np.concatenate((Xl_left[kl, :, :], Xr_middle[i, :, :], Xr_right[i, :, :]), axis=0)  # 右拼0类左-->0
        RL2R = np.take(RL2R, real_list, axis=-2)  # channel 维度重排序 1
        RR2L = np.take(RR2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedR2L[i, :, :] = RR2L
        transformedR2R[i, :, :] = RL2R
    # transformedLX = np.concatenate((transformedL2L, transformedR2L), axis=0)  # 0
    # transformedRX = np.concatenate((transformedL2R, transformedR2R), axis=0)  # 1
    y_la = np.zeros((len(llist)))
    y_ra = np.ones((len(rlist)))
    return Xl, transformedL2L, transformedL2R, Xr, transformedR2L, transformedR2R, y_la, y_ra


def Augmentation_left_right_half_sample_label(args, X, y, left_mat, right_mat, middle_mat):
    """
    初级，aug时不考虑标签；之后aug时将标签考虑进去
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    if 'halfsample' in args.method:
        X = X.cpu()
    num_samples, num_channels, num_timesamples = X.shape
    llist = [i for i in y if i == 0]
    rlist = [i for i in y if i == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedL2R = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2L = np.zeros((len(rlist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        kr = random.choice([ele for ele in rlen])
        L2R = np.concatenate((Xl_left[i, :, :], Xl_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        L2R = np.take(L2R, real_list, axis=-2)  # channel 维度重排序 1
        L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = L2L
        transformedL2R[i, :, :] = L2R
    for i in range(len(rlist)):
        kl = random.choice([ele for ele in llen])
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        R2L = np.concatenate((Xl_left[kl, :, :], Xr_middle[i, :, :], Xr_right[i, :, :]), axis=0)  # 右拼0类左-->0
        R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        R2L = np.take(R2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedR2L[i, :, :] = R2L
        transformedR2R[i, :, :] = R2R
    transformedLX = np.concatenate((transformedL2L, transformedR2L), axis=0)  # 0
    transformedRX = np.concatenate((transformedL2R, transformedR2R), axis=0)  # 1
    y_la = np.zeros((num_samples))
    y_ra = np.ones((num_samples))
    return transformedLX, transformedRX, y_la, y_ra


def Augmentation_left_right_half_sample(args, X, y, left_mat, right_mat, middle_mat, seed):
    """
    初级，aug时不考虑标签；之后aug时将标签考虑进去
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'halfsample' in args.method:
        X = X.cpu()
    num_samples, num_channels, num_timesamples = X.shape
    # llist = [i for i in y if y == 0]
    # rlist = [i for i in y if y == 1]
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedL2R = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2L = np.zeros((len(rlist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        # kr = random.choice([ele for ele in rlen])
        # L2R = np.concatenate((Xl_left[i, :, :], Xl_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        # L2R = np.take(L2R, real_list, axis=-2)  # channel 维度重排序 1
        L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = L2L
    for i in range(len(rlist)):
        # kl = random.choice([ele for ele in llen])
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        # R2L = np.concatenate((Xl_left[kl, :, :], Xr_middle[i, :, :], Xr_right[i, :, :]), axis=0)  # 右拼0类左-->0
        R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        # R2L = np.take(R2L, real_list, axis=-2)  # channel 维度重排序 0
        # transformedR2L[i, :, :] = R2L
        transformedR2R[i, :, :] = R2R
    # transformedLX = np.concatenate((transformedL2L, transformedR2L), axis=0)  # 0
    # transformedRX = np.concatenate((transformedL2R, transformedR2R), axis=0)  # 1
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    return transformedL2L, transformedR2R, y_la, y_ra


def Augmentation_left_right_half_sample_sort(args, X, y, left_mat, right_mat, middle_mat, seed):
    """
    考虑ori和aug的对应顺序
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_samples, num_channels, num_timesamples = X.shape
    llist = [i for i in y if i == 0]
    rlist = [i for i in y if i == 1]
    X_left = X[:, left_mat, :]
    X_right = X[:, right_mat, :]
    X_middle = X[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedX = np.zeros((num_samples, num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(num_samples):
        if i in llist:
            kl = random.choice([ele for ele in llen if ele != i])
            L2L = np.concatenate((X_left[kl, :, :], X_middle[i, :, :], X_right[i, :, :]), axis=0)  # 右拼0类左-->0
            L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
            transformedX[i, :, :] = L2L
        elif i in rlist:
            kr = random.choice([ele for ele in rlen if ele != i])
            R2R = np.concatenate((X_left[i, :, :], X_middle[i, :, :], X_right[kr, :, :]), axis=0)  # 左拼1类右-->1
            R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
            transformedX[i, :, :] = R2R
    return transformedX


def Augmentation_left_right_half_sample_new(args, X, y, left_mat, right_mat, middle_mat, seed):
    """
    -------
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'halfsample' in args.method:
        X = X.cpu()
    num_samples, num_channels, num_timesamples = X.shape
    # llist = [i for i in y if y == 0]
    # rlist = [i for i in y if y == 1]
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    # transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    # transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    transformedL2L = []
    transformedR2R = []
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[kl, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        # L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        # transformedL2L[i, :, :] = L2L
        transformedL2L.append(L2L)
    for i in range(len(rlist)):
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[kr, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        # R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        # transformedR2R[i, :, :] = R2R
        transformedR2R.append(R2R)
    transformedL2L = np.array(transformedL2L)
    transformedR2R = np.array(transformedR2R)
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    X = np.concatenate((X[:, left_mat, :], X[:, middle_mat, :], X[:, right_mat, :]), axis=1)  # 右拼0类左-->0
    return X, transformedL2L, transformedR2R, y_la, y_ra


def Augmentation_left_right_half_sample_subject(args, X, y, left_mat, right_mat, middle_mat, seed):
    """
    初级，aug时不考虑标签；考虑subject的ID，按照subject的ID进行增强
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if 'halfsample' in args.method:
        X = X.cpu()
    trial_num = args.trial_num
    num_samples, num_channels, num_timesamples = X.shape
    # llist = [i for i in y if y == 0]
    # rlist = [i for i in y if y == 1]
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        t = int(i / (trial_num / 2))  # 仅拼接当前subject的半样本
        kl = random.choice([ele for ele in range(int(trial_num / 2) * t, int(trial_num / 2) * (t + 1)) if ele != i])
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = L2L
    for i in range(len(rlist)):
        t = int(i / (trial_num / 2))  # 仅拼接当前subject的半样本
        kr = random.choice([ele for ele in range(int(trial_num / 2) * t, int(trial_num / 2) * (t + 1)) if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        transformedR2R[i, :, :] = R2R
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    return transformedL2L, transformedR2R, y_la, y_ra


def data_aug(args, data, labels, size, flag_aug):
    mult_flag, noise_flag, neg_flag, freq_mod_flag, cr_flag, hs_flag = flag_aug[0], flag_aug[1], flag_aug[2], flag_aug[
        3], flag_aug[4], flag_aug[5]

    n_channels = data.shape[2]
    data_out = data  # 1 raw features
    labels_out = labels

    if mult_flag:  # 2 features
        mult_data_add, labels_mult = data_mult_f(args, data, labels, size, n_channels=n_channels)
        # data_out = np.concatenate([data_out, mult_data_add], axis=0)
        # labels_out = np.append(labels_out, labels_mult)
        data_out = mult_data_add
        labels_out = labels_mult
    if noise_flag:  # 1 features
        noise_data_add, labels_noise = data_noise_f(args, data, labels, size, n_channels=n_channels)
        # data_out = np.concatenate([data_out, noise_data_add], axis=0)
        # labels_out = np.append(labels_out, labels_noise)
        data_out = noise_data_add
        labels_out = labels_noise
    if neg_flag:  # 1 features
        neg_data_add, labels_neg = data_neg_f(data, labels, size, n_channels=n_channels)
        # neg_data_add = CD(args, data)
        # labels_neg = labels
        data_out = neg_data_add
        labels_out = labels_neg
        # data_out = np.concatenate([data_out, neg_data_add], axis=0)
        # labels_out = np.append(labels_out, labels_neg)
    if freq_mod_flag:  # 2 features
        # freq_data_add, labels_freq = freq_mod_f(args, data, labels, size, n_channels=n_channels)  # version 1
        # freq_data_add = freqShift(args, data)
        if args.freq_method == 'surr':
            freq_data_add = freqSur(args, data)
            labels_freq = labels
        elif args.freq_method == 'shift':
            freq_data_add, labels_freq = freq_mod_f(args, data, labels, size, n_channels=n_channels)
        elif args.freq_method == 'large':
            freq_data_add, labels_freq = data_mult_f2(args, data, labels, size, n_channels=n_channels)
        # data_out = np.concatenate([data_out, freq_data_add], axis=0)
        # labels_out = np.append(labels_out, labels_freq)
        data_out = freq_data_add
        labels_out = labels_freq
    if cr_flag:
        cr_data, labels_cr = cr_transform(args, data, labels)
        data_out = cr_data
        labels_out = labels_cr
    if hs_flag:
        hs_data_l, labels_hs_l, hs_data_r, labels_hs_r = hs_transform(args, data, labels)
        hs_data = np.concatenate((hs_data_l, hs_data_r), axis=0)
        labels_hs = np.concatenate((labels_hs_l, labels_hs_r), axis=0)
        data_out = hs_data
        labels_out = labels_hs

    # 最终输出data格式为
    # raw 144, mult_add 144, mult_reduce 144, noise 144, neg 144, freq1 144, freq2 144
    return data_out, labels_out


def data_noise_f(args, data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    noise_mod_val = args.noise_mode  # [0.25, 0.5, 1, 2, 4]
    noise_mod_val = 1  # [0.25, 0.5, 1, 2, 4]
    # print("noise mod: {}".format(noise_mod_val))
    for i in range(len(labels)):
        if labels[i] >= 0:
            stddev_t = np.std(data[i])
            rand_t = np.random.rand(data[i].shape[0], data[i].shape[1])
            rand_t = rand_t - 0.5
            to_add_t = rand_t * stddev_t / noise_mod_val
            data_t = data[i] + to_add_t
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def cr_transform(args, X, y):
    """
    Parameters
    ----------
    X: torch tensor of shape (num_samples, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """
    if 'BNCI2014001' in args.data_name:
        left_mat = [1, 2, 6, 7, 8, 13, 14, 18]
        right_mat = [5, 4, 12, 11, 10, 17, 16, 20]
    elif args.data_name == 'MI1-7':
        left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49, 50, 55, 57]
        right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53, 52, 56, 58]
    elif args.data_name == 'BNCI2014004':
        left_mat = [0]
        right_mat = [2]
    elif args.data_name == 'BNCI2014002':
        left_mat = [0, 3, 4, 5, 6, 12]
        right_mat = [2, 11, 10, 9, 8, 14]
    elif args.data_name == 'BNCI2015001':
        left_mat = [0, 3, 4, 5, 10]
        right_mat = [2, 9, 8, 7, 12]
    elif args.data_name == 'Zhou2016':
        left_mat = [0, 2, 5, 8, 11]
        right_mat = [1, 4, 7, 10, 13]
    num_samples, num_channels, num_timesamples = X.shape
    transformedX = np.zeros((num_samples, num_channels, num_timesamples))
    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, ch, :] = X[:, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, ch, :] = X[:, left_mat[ind], :]
        else:
            transformedX[:, ch, :] = X[:, ch, :]
    if args.data_name == 'Zhou2016' or args.data_name == 'MI1-7' or args.data_name == 'BNCI2014001' or 'BNCI2014004' in args.data_name:
        labels = 1 - y
    elif args.data_name == 'BNCI2014001-4':
        # 交换 0 和 1 的标签
        swapped_labels = y.copy()
        swapped_labels[y == 0] = -1  # 临时标记为 -1
        swapped_labels[y == 1] = 0  # 1 → 0
        swapped_labels[swapped_labels == -1] = 1  # -1 → 1
        labels = swapped_labels
    else:
        labels = y
    return transformedX, labels


def Aug_channel_reflection_transform(X, left_mat, right_mat):
    """

    Parameters
    ----------
    X: torch tensor of shape (num_samples, num_channels, num_timesamples)
    left_mat: numpy array of shape (a, ), where a is the number of left brain channels, in order
    right_mat: numpy array of shape (b, ), where b is the number of right brain channels, in order

    Returns
    -------
    transformedX: transformed signal of torch tensor of shape (num_samples, num_channels, num_timesamples)
    """

    num_samples, num_channels, num_timesamples = X.shape
    transformedX = torch.zeros((num_samples, num_channels, num_timesamples))
    for ch in range(num_channels):
        if ch in left_mat:
            ind = left_mat.index(ch)
            transformedX[:, ch, :] = X[:, right_mat[ind], :]
        elif ch in right_mat:
            ind = right_mat.index(ch)
            transformedX[:, ch, :] = X[:, left_mat[ind], :]
        else:
            transformedX[:, ch, :] = X[:, ch, :]

    return transformedX


def data_mult_f(args, data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    mult_mod = args.mult_mode  # [0.005, 0.01, 0.05, 0.1, 0.2]
    # print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'neg':
            # print(data[i])
            data_t = data[i] * (1 + mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'pos':
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels

def data_mult_f2(args, data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    mult_mod = args.mult_mode2  # [0.005, 0.01, 0.05, 0.1, 0.2]
    # print("mult mod: {}".format(mult_mod))
    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'neg':
            # print(data[i])
            data_t = data[i] * (1 + mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'pos':
            data_t = data[i] * (1 - mult_mod)
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels

def data_neg_f(data, labels, size, n_channels=22):
    # Returns: data double the size of the input over time, with new data
    # being a reflection along the amplitude

    new_data = []
    new_labels = []
    for i in range(len(labels)):
        if labels[i] >= 0:
            data_t = -1 * data[i]
            # data_t = data_t - np.min(data_t)  # TODO
            new_data.append(data_t)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_mod_f(args, data, labels, size, n_channels=22):
    new_data = []
    new_labels = []
    # print(data.shape)
    freq_mod = args.freq_mode  # [0.1, 0.2, 0.3, 0.4, 0.5]
    # print("freq mod: {}".format(freq_mod))
    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'neg':
            low_shift = freq_shift(data[i], -freq_mod, num_channels=n_channels)
            new_data.append(low_shift)
            new_labels.append(labels[i])

    for i in range(len(labels)):
        if labels[i] >= 0 and args.augsettings == 'pos':
            high_shift = freq_shift(data[i], freq_mod, num_channels=n_channels)
            new_data.append(high_shift)
            new_labels.append(labels[i])

    new_data_ar = np.array(new_data).reshape([-1, size, n_channels])
    new_labels = np.array(new_labels)

    return new_data_ar, new_labels


def freq_shift(x, f_shift, dt=1 / 250, num_channels=22):
    shifted_sig = np.zeros((x.shape))
    len_x = len(x)
    padding_len = 2 ** nextpow2(len_x)
    padding = np.zeros((padding_len - len_x, num_channels))
    with_padding = np.vstack((x, padding))
    hilb_T = hilbert(with_padding, axis=0)
    t = np.arange(0, padding_len)
    shift_func = np.exp(2j * np.pi * f_shift * dt * t)
    for i in range(num_channels):
        shifted_sig[:, i] = (hilb_T[:, i] * shift_func)[:len_x].real

    return shifted_sig


def nextpow2(x):
    return int(np.ceil(np.log2(np.abs(x))))


def hs_transform(args, X, y):
    """
    考虑ori和aug的对应顺序
    Parameters
    ----------
    X: input original EEG signals
    y: the corresponding labels

    Returns
    X_la: left aug samples;
    X_ra: right aug samples;
    y_la, y_ra: the corresponding labels
    -------
    """
    seed = args.SEED
    if 'BNCI2014001' in args.data_name:
        left_mat = [1, 2, 6, 7, 8, 13, 14, 18]
        middle_mat = [0, 3, 9, 15, 19, 21]
        right_mat = [5, 4, 12, 11, 10, 17, 16, 20]
    elif args.data_name == 'MI1-7':
        left_mat = [0, 2, 3, 4, 9, 10, 11, 16, 17, 18, 19, 24, 25, 26, 27, 33, 34, 35, 36, 41, 42, 43, 48, 49,
                    50, 55, 57]
        middle_mat = [5, 12, 28, 44, 51]
        right_mat = [1, 8, 7, 6, 15, 14, 13, 23, 22, 21, 20, 32, 31, 30, 29, 40, 39, 38, 37, 47, 46, 45, 54, 53,
                     52, 56, 58]
    elif args.data_name == 'BNCI2014004':
        left_mat = [0]
        middle_mat = [1]
        right_mat = [2]
    elif args.data_name == 'BNCI2014002':
        left_mat = [0, 3, 4, 5, 6, 12]
        right_mat = [2, 11, 10, 9, 8, 14]
        middle_mat = [1, 7, 13]
    elif args.data_name == 'BNCI2015001':
        left_mat = [0, 3, 4, 5, 10]
        middle_mat = [1, 6, 11]
        right_mat = [2, 9, 8, 7, 12]
    elif args.data_name == 'Zhou2016':
        left_mat = [0, 2, 5, 8, 11]
        middle_mat = [3, 6, 9, 12]
        right_mat = [1, 4, 7, 10, 13]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    num_samples, num_channels, num_timesamples = X.shape
    # llist = [i for i in y if i == 0]
    # rlist = [i for i in y if i == 1]
    llist = [i for i in range(len(y)) if y[i] == 0]
    rlist = [i for i in range(len(y)) if y[i] == 1]
    Xl = X[llist, :, :]
    Xr = X[rlist, :, :]
    Xl_left = Xl[:, left_mat, :]
    Xl_right = Xl[:, right_mat, :]
    Xl_middle = Xl[:, middle_mat, :]
    Xr_left = Xr[:, left_mat, :]
    Xr_right = Xr[:, right_mat, :]
    Xr_middle = Xr[:, middle_mat, :]
    llen = list(range(0, len(llist)))
    rlen = list(range(0, len(rlist)))
    transformedL2L = np.zeros((len(llist), num_channels, num_timesamples))
    transformedL2R = np.zeros((len(llist), num_channels, num_timesamples))
    transformedR2L = np.zeros((len(rlist), num_channels, num_timesamples))
    transformedR2R = np.zeros((len(rlist), num_channels, num_timesamples))
    clist = left_mat + middle_mat + right_mat
    real_list = [clist.index(h) for h in range(0, num_channels)]
    for i in range(len(llist)):
        kl = random.choice([ele for ele in llen if ele != i])
        # kr = random.choice([ele for ele in rlen])
        # L2R = np.concatenate((Xl_left[i, :, :], Xl_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        L2L = np.concatenate((Xl_left[kl, :, :], Xl_middle[i, :, :], Xl_right[i, :, :]), axis=0)  # 右拼0类左-->0
        # L2R = np.take(L2R, real_list, axis=-2)  # channel 维度重排序 1
        L2L = np.take(L2L, real_list, axis=-2)  # channel 维度重排序 0
        transformedL2L[i, :, :] = L2L
    for i in range(len(rlist)):
        # kl = random.choice([ele for ele in llen])
        kr = random.choice([ele for ele in rlen if ele != i])
        R2R = np.concatenate((Xr_left[i, :, :], Xr_middle[i, :, :], Xr_right[kr, :, :]), axis=0)  # 左拼1类右-->1
        # R2L = np.concatenate((Xl_left[kl, :, :], Xr_middle[i, :, :], Xr_right[i, :, :]), axis=0)  # 右拼0类左-->0
        R2R = np.take(R2R, real_list, axis=-2)  # channel 维度重排序 1
        # R2L = np.take(R2L, real_list, axis=-2)  # channel 维度重排序 0
        # transformedR2L[i, :, :] = R2L
        transformedR2R[i, :, :] = R2R
    # transformedLX = np.concatenate((transformedL2L, transformedR2L), axis=0)  # 0
    # transformedRX = np.concatenate((transformedL2R, transformedR2R), axis=0)  # 1
    y_la = np.zeros((transformedL2L.shape[0]))
    y_ra = np.ones((transformedR2R.shape[0]))
    return transformedL2L, y_la, transformedR2R, y_ra  # 只生成1倍增强数据
