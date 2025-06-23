import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm_notebook
import pydicom
from PIL import Image
from src.model import Nest_Net
from mask_functions import rle2mask, mask2rle

def test_images_pred(test_fns, model_path='../outputs/unet.h5', img_size=512, orig_size=1024):
    pred_rle = []
    ids = []
    model = Nest_Net(img_size, img_size, 1)
    model.load_weights(model_path)
    for f in tqdm_notebook(test_fns):
        img = pydicom.read_file(f).pixel_array
        img = cv2.resize(img, (img_size, img_size))
        img = model.predict(img.reshape(1, img_size, img_size, 1))
        img = img.reshape(img_size, img_size)
        ids.append(os.path.splitext(os.path.basename(f))[0])
        pil_img = Image.fromarray((img.T * 255).astype(np.uint8)).resize((orig_size, orig_size))
        pil_img = np.asarray(pil_img)
        pred_rle.append(mask2rle(pil_img, orig_size, orig_size))
    return pred_rle, ids

def make_submission(test_fns, submission_path='submission.csv'):
    preds, ids = test_images_pred(test_fns)
    submission = pd.DataFrame({'ImageId': ids, 'EncodedPixels': preds})
    submission.to_csv(submission_path, index=False)
    return submission
