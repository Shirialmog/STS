import os
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import time
import argparse
import logging

from voxelwise_encoding.utils import clean_image, save_group_nii, save_as_nii
from models_config import models_config_dict

def concat_features(features_list, single_features_dir):
    processed_annotations = []
    for item in features_list:
        feature = np.load(os.path.join(single_features_dir, f'{item}.npy'))
        processed_annotations.append(feature)
    processed_annotations = np.concatenate(processed_annotations, axis=1)
    return processed_annotations


def main(data_path, annotations_path, mask_path, model, group_dir, original_data_shape, num_subjects, alphas):
    feature_names = models_config_dict[model]
    features = concat_features(feature_names, annotations_path)

    num_features = len(feature_names)
    X = normalize(features, axis=0)

    r_nifti_group = np.zeros([num_subjects, original_data_shape[0], original_data_shape[1], original_data_shape[2]])
    weight_nifti_group = np.zeros(
        [num_subjects, num_features, original_data_shape[0], original_data_shape[1], original_data_shape[2]])

    for subj in range(1, num_subjects + 1):
        print(f'subject: {subj}')
        save_dir = os.path.join(args.results_dir, f'sub{subj}/{model}/')
        os.makedirs(save_dir, exist_ok=True)
        filename = f'sub-{subj}_task-500daysofsummer_bold_blur_censor_ica.nii.gz'
        fmri_path = os.path.join(data_path, f'sub{subj}/derivatives', filename)

        if mask_path:
            mask = mask_path
        else:
            mask = None
        data_clean, masked_indices, original_data_shape, img_affine = clean_image(fmri_path, subj, mask,
                                                                                  args.results_dir)
        data_clean = data_clean[:5469] # in case the data is a frame or a few frames longer..
        num_voxels = data_clean.shape[1]
        y = data_clean.copy()

        # split to train and test, 80 20
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # fit ridge regression
        logging.info('Fitting ridge regression')
        ridge_results = RidgeCV(alphas=alphas)
        ridge_results.fit(X_train, y_train)
        ridge_coef = ridge_results.coef_

        # predict
        logging.info('Predicting')
        y_pred = ridge_results.predict(X_test)

        # get correlation per voxel
        logging.info('Calculating correlation per voxel')
        r = np.zeros(num_voxels)
        for i in range(num_voxels):
            r[i] = np.corrcoef(y_test[:, i], y_pred[:, i])[0, 1]

        # get weights ( r per feature)
        logging.info('Calculating weights')
        weights = np.zeros([num_features, num_voxels])
        for i in range(num_features):
            feature_coef = ridge_coef.copy()
            # make all other features 0
            for j in range(num_features):
                if j != i:
                    feature_coef[:, j] = 0
            # predict ridge with only one feature
            y_pred = np.dot(X_test, feature_coef.T)

            for v in range(num_voxels):
                weights[i, v] = np.corrcoef(y_test[:, v], y_pred[:, v])[0, 1]

        # use masked indices to convert r_mean, weight_mean, and b_mean to 3D arrays
        r_nifti = np.zeros((original_data_shape))
        weight_nifti = np.zeros([num_features, original_data_shape[0], original_data_shape[1], original_data_shape[2]])

        r_nifti[masked_indices[0], masked_indices[1], masked_indices[2]] = r
        weight_nifti[:, masked_indices[0], masked_indices[1], masked_indices[2]] = weights
        save_as_nii(model, subj, r_nifti, weight_nifti, save_dir, feature_names, img_affine)

        r_nifti_group[subj - 1, :, :, :] = r_nifti
        weight_nifti_group[subj - 1, :, :, :, :] = weight_nifti
    r_mean = np.mean(r_nifti_group, axis=0)
    weight_mean = np.mean(weight_nifti_group, axis=0)
    save_group_nii(model, r_mean, weight_mean, group_dir, feature_names, img_affine)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--fmri_data_path', type=str)
    args.add_argument('--annotations_path', type=str)
    args.add_argument('--isc_mask_path', type=str, required=False)
    args.add_argument('--results_dir', type=str)
    args.add_argument('--model', type=str, default='full', help='full, social, social_plus_llava, llava_features, llava_only_social')
    args = args.parse_args()

    start_time = time.time()
    model = args.model
    print(f'model type: {model}')
    group_dir = os.path.join(args.results_dir, f'group/{model}/')
    os.makedirs(group_dir, exist_ok=True)

    alphas = np.logspace(1, 4, 10)
    original_data_shape = [64, 76, 64]
    num_subjects = 3

    main(args.fmri_data_path, args.annotations_path, args.isc_mask_path, model, group_dir, original_data_shape, num_subjects, alphas)

    duration = round((time.time() - start_time) / 60)
    print(f'duration: {duration} mins')
