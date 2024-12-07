<h1> Voxelwise Encoding Using Ridge Regression</h1>

The script takes the following inputs:
- fmri_data_path: path to the preprocessed fMRI data. This should be a 4D Nifti file. Here we used data from the 500 days of summer movie, which can be downloaded from the OpenNeuro repository (https://openneuro.org/datasets/ds002837/versions/2.0.0). Specifically, we used the pre-processed data in the derivatives folder of each subject, and the file called 'sub-{subj}_task-500daysofsummer_bold_blur_censor_ica.nii.gz'. 
- annotations_path: path to folder containing annotations for each feature. Each feature should be saved as a separate numpy file, and the name of the file should be the name of the feature. For example 'faces.npy'. The length of the numpy array should be the same as the number of TRs in the fMRI data. 
- isc_mask_path: path to the ISC mask. This should be a 3D Nifti file. More information on how to create this mask will be provided in the future. If mask is not provided, the script will use the whole brain mask. This will take longer to run, and results will be less significant.
- results_dir: directory where the results will be saved.
- model: refers to the collection of features that will be used for the encoding. See models_config.py for possible combinations, and add your own if needed.

 
<h2> Folder Hierarchy: </h2>
```
- fmri_data_path
    - sub1
        - derivatives
            - sub-1_task-500daysofsummer_bold_blur_censor_ica.nii.gz
    - sub2
        - derivatives
            - sub-2_task-500daysofsummer_bold_blur_censor_ica.nii.gz
    ...
- annotations_path
    - faces.npy
    - objects.npy
    - ...
- isc_mask_path
    - isc_mask.nii.gz
```

<h2> Plotting Results on Surface </h2>
![surface_plot_example](voxelwise_encoding/surface_plot_example.png)

The script 'voxelwise_encoding_ridge.py' will save nibabel files with the encoding results. To plot the results, you can use the script 'plot_nii_as_surf.py'. This script will plot the results on the surface of the brain, and open in a browser. You can then right click the plot and save it as an html file.



