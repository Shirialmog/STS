from scipy.linalg import svd
from numpy.linalg import inv
from scipy.stats import pearsonr
from tqdm import tqdm
from nilearn import image


def clean_image(filename, subj, mask, fmripath):
    """
    Read fMRI data, mask the image with intersubject correlation map (r>0.25),
    clean the data (remove NaN, Inf, columns with constant values), save the nii file.
    """
    data_path = os.path.join(fmripath, filename + '.nii.gz')
    npy_path = os.path.join(fmripath, f'fmri_s{subj}_short_32.npy')

    if os.path.isfile(npy_path):
        print(f'all files exist for the subject {subj}.\nskip the operation.')
        data_clean = np.load(npy_path)
    else:
        original_data = load_fmri_data(data_path)
        masked_data = apply_mask(original_data, mask)
        data_clean = remove_useless_data(masked_data)
        np.save(npy_path, data_clean)
        # TODO: figure out how to recreate the nii file to visualize the data after cleaning. its a different shape now.
        print(f'saving npy files for the subject {subj}.')

    return data_clean


def load_fmri_data(filepath):
    img = image.load_img(filepath)
    data = img.get_fdata()
    return data


def apply_mask(data, mask):
    # TODO SHIRI: verify this function. is this returning the correct shape?
    if mask is not None:
        mask_data = nib.load(mask).get_fdata()
        data = data * (mask_data > 0.25)
    # else:
    #     data = data.reshape(-1, data.shape[-1])
    #     data = data.T

    return data


def remove_useless_data(data):
    # TODO: FIX THIS
    """
    Clean the data by removing voxels with NaN values, inf values, or constant values.
    """

    cleaned_data =  data[:, np.std(data, axis=0) > 0]
    cleaned_data = cleaned_data[:,~np.isnan(cleaned_data).any(axis=0)]
    cleaned_data = cleaned_data[:,~np.isinf(cleaned_data).any(axis=0)]

    return cleaned_data


def normalize(data):
    """
    Z-score normalization (mean=0, std=1)
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    return (data - mean) / std


def bridge(X1_training, X2_training, y_training, alphas, ratios, method):
    """
    Modified version of Tikhonov regression function (tikreg) in Python
    with a polar grid search using ratio and alpha
    method 1 -> standard Tikhonov regression; method 2-> standard form solution;
    method 3 -> Singular value decomposition (the fastest solution!)
    Citation: Nunez-Elizalde AO, Huth AG, and Gallant, JL (2019).
    Voxelwise encoding models with non-spherical multivariate normal priors. NeuroImage.
    """
    X1 = normalize(X1_training)
    X2 = normalize(X2_training)
    Ytrain = normalize(y_training)

    n1, f1 = X1.shape
    n2, f2 = X2.shape
    n, v = Ytrain.shape

    if n1 != n:
        raise ValueError('Input size mismatch for X1 and Ytrain.')
    if n2 != n:
        raise ValueError('Input size mismatch for X2 and Ytrain.')

    angle = np.arctan(ratios)
    b_banded = np.zeros((f1 + f2, v, len(angle), len(alphas)))
    lamb1 = np.zeros((len(angle), len(alphas)))
    lamb2 = np.zeros((len(angle), len(alphas)))

    for k, alpha in enumerate(alphas):
        solution = np.zeros((f1 + f2, v, len(angle)))
        lambda1 = np.zeros(len(angle))
        lambda2 = np.zeros(len(angle))

        for i, ang in enumerate(angle):
            lambda_one = np.cos(ang) * alpha
            lambda_two = np.sin(ang) * alpha
            bands = np.concatenate((np.ones(f1) * lambda_one, np.ones(f2) * lambda_two))
            C = np.diag(bands)
            Cinv = inv(C)

            if method == 1:
                # Tikhonov banded ridge (computationally very expensive)
                Xjoint = np.concatenate((X1, X2), axis=1)
                LH = inv((Xjoint.T @ Xjoint) + (C.T @ C))
                XTY = Xjoint.T @ Ytrain
                solution[:, :, i] = LH @ XTY

            elif method == 2:
                # Standard scaled banded ridge (computationally less expensive)
                A = np.concatenate((X1 / (lambda_one / alpha), X2 / (lambda_two / alpha)), axis=1)
                LH = inv((A.T @ A) + (alpha ** 2) * np.eye(A.shape[1]))
                RH = A.T @ Ytrain
                solution_standard = (LH @ RH) * alpha
                solution[:, :, i] = Cinv @ solution_standard

            else:
                # Banded ridge with SVD (fastest)
                A = np.concatenate((X1 / (lambda_one / alpha), X2 / (lambda_two / alpha)), axis=1)
                U, S, Vt = svd(A, full_matrices=False)
                UTY = U.T @ Ytrain
                D = np.diag(S / (S ** 2 + alpha ** 2))
                solution_svd = (Vt.T @ D @ UTY) * alpha
                solution[:, :, i] = Cinv @ solution_svd

            lambda1[i] = lambda_one
            lambda2[i] = lambda_two

        b_banded[:, :, :, k] = solution
        lamb1[:, k] = lambda1
        lamb2[:, k] = lambda2

    return b_banded, lamb1, lamb2


def bridge_test(X1, X2, y, Lambda1, Lambda2):
    """
    Banded ridge regression test
    Created by Haemy Lee Masson July/2020
    """
    Ytrain = normalize(y)
    X1 = normalize(X1)
    X2 = normalize(X2)
    Ytrain = normalize(Ytrain)

    n1, f1 = X1.shape  # n: observations, f: features
    n2, f2 = X2.shape
    n, v = Ytrain.shape  # n: observations, v: voxels

    if n1 != n:
        raise ValueError('Input size mismatch for X1 and Ytrain.')
    if n2 != n:
        raise ValueError('Input size mismatch for X2 and Ytrain.')

    b = np.zeros((f1 + f2, y.shape[1]))

    for voxel in range(y.shape[1]):
        lambda_one = Lambda1[voxel]
        lambda_two = Lambda2[voxel]
        bands = np.concatenate((np.ones(f1) * lambda_one, np.ones(f2) * lambda_two))
        C = np.diag(bands)

        # Tikhonov banded ridge
        Xjoint = np.concatenate((X1, X2), axis=1)
        LH = np.linalg.inv((Xjoint.T @ Xjoint) + (C.T @ C))
        XTY = Xjoint.T @ Ytrain[:, voxel]
        b[:, voxel] = LH @ XTY

    return b


def prediction(b, outerX, y_true, name_feature, featuredir):
    """
    Correlation between estimated y and true y.
    Made by Haemy Lee Masson July/2020
    """
    num_voxel = b.shape[1]
    y_hat = np.dot(outerX, b)  # predicted bold signal
    R = np.zeros(num_voxel, dtype=np.float32)

    # Compute correlation for each voxel
    print('Computing correlation for each voxel')
    for v in tqdm(range(num_voxel)):
        R[v] = pearsonr(y_true[:, v], y_hat[:, v])[0]

    k = 1
    R_features = np.zeros((len(name_feature), num_voxel))
    weight = np.zeros(num_voxel, dtype=np.float32)

    for i, feature_name in enumerate(name_feature):
        feature_path = f"{featuredir}/{feature_name}.npy"
        A = np.load(feature_path)
        A_size = A.shape[0] * A.shape[1]

        if A_size > 6000:  # 1976
            y_hat = np.dot(outerX[:, k:k + A.shape[1]], b[k:k + A.shape[1], :])
            k += A.shape[1]  # 1921
        else:
            y_hat = np.outer(outerX[:, k], b[k, :])  # TODO: check this
            k += 1  # 1976
        print('For feature:', feature_name, 'computing correlation for each voxel')
        for v in tqdm(range(num_voxel)):
            weight[v] = pearsonr(y_true[:, v], y_hat[:, v])[0]

        R_features[i, :] = weight

    return R, R_features


import numpy as np
import nibabel as nib
import os


def bridge_results_nii(model, subj, r_mean, b_mean, weight_mean, savedir, name_feature):
    """
    Save data to NIfTI format.
    """
    # Update data_clean.samples with r_mean and save to NIfTI
    save_nifti(r_mean, os.path.join(savedir, f"{model}_r_sub{subj}.nii"))
    group_r = r_mean

    group_weights_all = b_mean

    save_weights = np.zeros_like(weight_mean)

    # Save each feature's weights to a separate NIfTI file
    for i, feature_name in enumerate(name_feature):
        encoding_weights = weight_mean[i, :]
        save_nifti(encoding_weights, os.path.join(savedir, f"{feature_name}_sub{subj}.nii"))
        save_weights[i, :] = encoding_weights

    group_weights = save_weights

    return group_r, group_weights, group_weights_all


def save_nifti(data, filename):
    """
    Helper function to save the data in NIfTI format.
    """
    affine = np.eye(4)  # Assuming an identity matrix for affine transformation; update as needed
    nifti_img = nib.Nifti1Image(data.reshape((64, 76, 64)), affine)
    nib.save(nifti_img, filename)
