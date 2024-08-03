'''
Based on https://github.com/haemyleemasson/voxelwise_encoding
by Hae-Yeoun Lee-Masson, this script is part of the voxelwise encoding analysis.
'''
import os
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.model_selection import KFold
import time
import argparse

from bridge_tools import bridge, bridge_test, prediction, bridge_results_nii, clean_image

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--basepath', type=str )
    args = args.parse_args()

    fmripath = os.path.join(args.basepath, 'fmri_data')

    model_name = ['full']
    model = model_name[0]  # TODO: figure out what this is used for
    print(f'model: {model}')
    groupdir = os.path.join(fmripath, f'results/group/{model}/')
    os.makedirs(groupdir, exist_ok=True)

    featuredir = os.path.join(args.basepath, 'annotations')
    name_feature = ['hue', 'saturation', 'pixel', 'layer5']

    processed_annotations = os.path.join(args.basepath, 'processed_annotations')
    feature1 = np.load(os.path.join(processed_annotations, 'all_features.npy'))
    feature2 = np.load(os.path.join(processed_annotations, 'layer5.npy'))

    nfeature = len(name_feature)
    feature1 = feature1[:, :2]
    feature2 = feature2[:, :2]
    X1 = normalize(feature1, axis=0)
    X2 = normalize(feature2, axis=0)

    # Subject level analysis, voxel-wise encoding with ridge regression
    alphas = np.logspace(1, 4, 10)
    ratios = np.logspace(-2, 2, 15)

    for subj in range(1, 2):
        start_time = time.time()
        print(f'subject: {subj}')

        savedir = os.path.join(fmripath, f'results/sub{subj}/{model}/')
        os.makedirs(savedir, exist_ok=True)
        filename = 'sub-1_task-500daysofsummer_bold_blur_censor_ica_sample'
        mask = None
        data_clean = clean_image(filename, subj, mask, fmripath)
        numvoxel = data_clean.shape[1]
        y = data_clean.copy()
        tr_num = y.shape[0]
        niteration = 10
        nfold = 5

        r_iterated = np.zeros((numvoxel, niteration))
        weights_iterated = np.zeros((nfeature, numvoxel, niteration))
        b_iterated = np.zeros((X1.shape[1] + X2.shape[1], numvoxel, niteration))

        kf = KFold(n_splits=niteration)
        for iter_num, (train_index, test_index) in enumerate(kf.split(y)):
            X1_train, X1_test = X1[train_index], X1[test_index]
            X2_train, X2_test = X2[train_index], X2[test_index]
            y_train, y_test = y[train_index], y[test_index]

            Lmat1 = np.zeros((numvoxel, len(alphas) * len(ratios), nfold), dtype='single')
            Lmat2 = np.zeros((numvoxel, len(alphas) * len(ratios), nfold), dtype='single')

            inner_kf = KFold(n_splits=nfold)
            for fold, (inner_train_index, inner_test_index) in enumerate(inner_kf.split(y_train)):
                print(f'iteration: {iter_num + 1}, fold: {fold + 1}')

                X1_inner_train, X1_inner_test = X1_train[inner_train_index], X1_train[inner_test_index]
                X2_inner_train, X2_inner_test = X2_train[inner_train_index], X2_train[inner_test_index]
                y_inner_train, y_inner_test = y_train[inner_train_index], y_train[inner_test_index]

                b_banded, lamb1, lamb2 = bridge(X1_inner_train, X2_inner_train, y_inner_train, alphas, ratios, 3)
                print('training done in the inner loop')

                SSE1_all = np.zeros((numvoxel, len(ratios), len(alphas)), dtype='single')
                SSE2_all = np.zeros((numvoxel, len(ratios), len(alphas)), dtype='single')

                for i in range(b_banded.shape[3]):
                    for k in range(b_banded.shape[2]):
                        y1_hat = np.dot(X1_inner_test, b_banded[:X1_inner_test.shape[1], :, k, i])
                        y2_hat = np.dot(X2_inner_test, b_banded[X1_inner_test.shape[1]:, :, k, i])
                        y_true = y_inner_test

                        SSE1_all[:, k, i] = np.sum((y_true - y1_hat) ** 2, axis=0)
                        SSE2_all[:, k, i] = np.sum((y_true - y2_hat) ** 2, axis=0)

                SSE1_all_reshape = SSE1_all.reshape((SSE1_all.shape[0], -1))
                SSE2_all_reshape = SSE2_all.reshape((SSE2_all.shape[0], -1))
                Lmat1[:, :, fold] = SSE1_all_reshape
                Lmat2[:, :, fold] = SSE2_all_reshape
                print('testing done in the inner loop')

            # Lambda1 = np.array([lamb1[np.argmin(np.mean(Lmat1[v, :, :], axis=1))] for v in range(numvoxel)])
            # Lambda2 = np.array([lamb2[np.argmin(np.mean(Lmat2[v, :, :], axis=1))] for v in range(numvoxel)])
            Lambda1 = lamb1[np.unravel_index((np.argmin((np.mean(Lmat1, axis=2)), axis=1)),
                                             lamb1.shape)]  # TODO: I think this is right, but should verify
            Lambda2 = lamb2[np.unravel_index((np.argmin((np.mean(Lmat2, axis=2)), axis=1)), lamb2.shape)]
            print('selecting best hyperparameters done')

            b = bridge_test(X1_train, X2_train, y_train, Lambda1, Lambda2)
            print('training for testing done')

            X = np.hstack([X1, X2])
            y_true = y_test
            R, R_features = prediction(b, X[test_index], y_true, name_feature, featuredir)
            print('prediction done')

            r_iterated[:, iter_num] = R
            weights_iterated[:, :, iter_num] = R_features
            b_iterated[:, :, iter_num] = b
            print('testing done')

        r_mean = np.mean(r_iterated, axis=1)
        weight_mean = np.mean(weights_iterated, axis=2)
        b_mean = np.mean(b_iterated, axis=2)

        group_r, group_weights, group_weights_all = bridge_results_nii(model, subj, r_mean, b_mean, weight_mean,
                                                                       savedir, name_feature)

        np.save(os.path.join(groupdir, 'group_r.npy'), group_r)
        np.save(os.path.join(groupdir, 'group_weights.npy'), group_weights)
        np.save(os.path.join(groupdir, 'group_weights_all.npy'), group_weights_all)

        duration = round((time.time() - start_time) / 60)
        print(f'duration: {duration} mins')
