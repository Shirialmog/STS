import os
import numpy as np
import scipy.io as sio
import argparse

def save_annotations_in_seconds(annotations, path):
    # each annotation represents 3 seconds
    events = []
   # add first 40 seconds of no annotations, 13 frames
    for i in range(13):
        item = [0] * annotations.shape[1]
        events.append(item)
        events.append(item)
        events.append(item)
        # TODO: rewrite this
    for i in range(len(annotations)):
        item = annotations[i]
        events.append(item)
        events.append(item)
        events.append(item)
    for i in range(88):
        item = [0] * annotations.shape[1]
        events.append(item)
        events.append(item)
        events.append(item)

    events = np.array(events)
    # save the annotations in seconds
    np.save(path, events)
    return events


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--basepath', type=str)
    args = args.parse_args()

    featuredir = os.path.join(args.basepath, 'features') # original annotations
    new_annotation_dir = os.path.join(args.basepath, 'annotations') # folder for annotations in numpy format, in seconds
    os.makedirs(new_annotation_dir, exist_ok=True)

    for feature_name in os.listdir(featuredir):
        feature_path = os.path.join(featuredir, feature_name)
        feature = sio.loadmat(feature_path)
        if feature_name == 'layer5.mat':
            feature = feature['layer_pca_norm']
        else:
            feature = feature[feature_name.split('.')[0]]

        print(f"{feature_path}: {feature.shape}")
        # save annotations in seconds
        save_annotations_in_seconds(feature, os.path.join(new_annotation_dir, f"{feature_name.split('.')[0]}.npy"))

    # combine all features except layer5, needed for encoding script
    processed_annotations = os.path.join(args.basepath, 'processed_annotations')
    os.makedirs(processed_annotations, exist_ok=True)
    all_features = []
    for feature_name in os.listdir(new_annotation_dir):
        if feature_name != 'layer5.npy':
            feature = np.load(os.path.join(new_annotation_dir, feature_name))
            all_features.append(feature)
    all_features = np.concatenate(all_features, axis=1)
    print(f"all_features: {all_features.shape}")
    # save annotations in seconds
    np.save(os.path.join(processed_annotations, 'all_features.npy'), all_features)
    np.save(os.path.join(processed_annotations, 'layer5.npy'), np.load(os.path.join(new_annotation_dir, 'layer5.npy')))

