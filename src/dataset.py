import os
import glob
import logging
import pydicom
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def show_dcm_info(dataset, file_path):
    print("Filename.........:", file_path)
    print("Storage type.....:", dataset.SOPClassUID)
    print()

    pat_name = dataset.PatientName
    display_name = pat_name.family_name + ", " + pat_name.given_name
    print("Patient's name......:", display_name)
    print("Patient id..........:", dataset.PatientID)
    print("Patient's Age.......:", dataset.PatientAge)
    print("Patient's Sex.......:", dataset.PatientSex)
    print("Modality............:", dataset.Modality)
    print("Body Part Examined..:", dataset.BodyPartExamined)
    print("View Position.......:", dataset.ViewPosition)
    
    if 'PixelData' in dataset:
        rows = int(dataset.Rows)
        cols = int(dataset.Columns)
        print("Image size.......: {rows:d} x {cols:d}, {size:d} bytes".format(
            rows=rows, cols=cols, size=len(dataset.PixelData)))
        if 'PixelSpacing' in dataset:
            print("Pixel spacing....:", dataset.PixelSpacing)

def plot_pixel_array(dataset, figsize=(10,10)):
    plt.figure(figsize=figsize)
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()

def plot_sample_images(image_glob, num_img=5, start=0, mask_df=None, mask_func=None):
    files = sorted(glob.glob(image_glob))[start:start+num_img]
    fig, ax = plt.subplots(nrows=1, ncols=num_img, sharey=True, figsize=(num_img*10,10))
    for q, file_path in enumerate(files):
        dataset = pydicom.dcmread(file_path)
        ax[q].imshow(dataset.pixel_array, cmap=plt.cm.bone)
        if mask_df is not None and mask_func is not None:
            img_id = os.path.splitext(os.path.basename(file_path))[0]
            if mask_df.loc[img_id, 1] != '-1':
                mask = mask_func(mask_df.loc[img_id, 1], 1024, 1024).T
                ax[q].set_title('See Marker')
                ax[q].imshow(mask, alpha=0.1, cmap="Reds")
            else:
                ax[q].set_title('Nothing to see')
        else:
            ax[q].set_title('Image')
    plt.show()

def load_train_images_masks(train_fns, df_full, rle2mask_func, im_height=1024, im_width=1024, im_chan=1):
    X_train = np.zeros((len(train_fns), im_height, im_width, im_chan), dtype=np.uint8)
    Y_train = np.zeros((len(train_fns), im_height, im_width, 1), dtype=bool)
    for n, _id in enumerate(train_fns):
        dataset = pydicom.read_file(_id)
        X_train[n] = np.expand_dims(dataset.pixel_array, axis=2)
        image_id = os.path.splitext(os.path.basename(_id))[0]
        try:
            mask_data = df_full.loc[image_id, ' EncodedPixels']
            if '-1' in str(mask_data):
                Y_train[n] = np.zeros((im_height, im_width, 1), dtype=bool)
            else:
                if isinstance(mask_data, str):
                    Y_train[n] = np.expand_dims(rle2mask_func(mask_data, im_height, im_width), axis=2)
                else:
                    Y_train[n] = np.zeros((im_height, im_width, 1), dtype=bool)
                    for x in mask_data:
                        Y_train[n] = Y_train[n] + np.expand_dims(rle2mask_func(x, im_height, im_width), axis=2)
        except KeyError:
            logger.warning(f"Key {image_id} without mask, assuming healthy patient.")
            Y_train[n] = np.zeros((im_height, im_width, 1), dtype=bool)
    return X_train, Y_train
