import nibabel as nib
import numpy as np
from nilearn import plotting

def plot_voxelwise_encoding_results_on_surface(results_file_path: str, model: str, feature: str):
    nii = nib.load(results_file_path)
    data = nii.get_fdata()
    # replace nan with 0, as nans are voxels that are outside the ISC mask
    data = np.nan_to_num(data, nan=0.0)
    # for visualization purposes, only plot positive correlations
    data[data < 0] = 0
    img = nib.Nifti1Image(data, affine=nii.affine)

    plotting.view_img_on_surf(img, surf_mesh='fsaverage', title=f'{model} - {feature}',
                              symmetric_cmap=False, cmap=plotting.cm.black_red, vmax=np.max(data)).open_in_browser()


path = r"C:\Users\shiri\Documents\School\Master\Research\STS\500_days_of_summer\voxelwise_encoding\simple_ridge\group\llava_features\face.nii"
plot_voxelwise_encoding_results_on_surface(path, model='llava_features', feature='face')