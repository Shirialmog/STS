<h1> Voxelwise Encoding Using Ridge Regression</h1>

The script takes the following inputs:
- 'fmri_data_path': path to the preprocessed fMRI data. This should be a 4D Nifti file. Here we used data from the 500 days of summer movie, which can be downloaded from the OpenNeuro repository (https://openneuro.org/datasets/ds002837/versions/2.0.0). Specifically, we used the pre-processed data in the derivatives folder of each subject, and the file called 'sub-{subj}_task-500daysofsummer_bold_blur_censor_ica.nii.gz'.
- 'annotations_path': path to folder containing annotations for each feature. Each feature should be saved as a separate numpy file, where each tr is represented by a row. i.e. the length of the numpy array should be the same as the number of TRs in the fMRI data. 
- 'isc_mask_path': path to the ISC mask. This should be a 3D Nifti file. More information on how to create this mask will be provided in the future. If not provided, the script will use the whole brain mask. This will take longer to run, and results will be less significant.
- 'results_dir': directory where the results will be saved.
- 'model': model refers to the collection of features that will be used for the encoding. See models_config.py for possible combinations, and add your own if needed.