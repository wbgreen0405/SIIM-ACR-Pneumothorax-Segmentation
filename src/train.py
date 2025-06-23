import os
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping

from src.model import Nest_Net
from src.losses import bceloss
from src.dataset import load_train_images_masks

def train_model(train_fns, df_full, rle2mask_func, im_height=1024, im_width=1024, im_chan=1, batch_size=32, epochs=4):
    X_train, Y_train = load_train_images_masks(train_fns, df_full, rle2mask_func, im_height, im_width, im_chan)
    model = Nest_Net(im_height, im_width, im_chan)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[bceloss])
    checkpoint = ModelCheckpoint('../outputs/unet.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, verbose=1, mode='auto', min_delta=1e-4)
    early = EarlyStopping(monitor="val_loss", mode="min", patience=6)
    callbacks_list = [checkpoint, early, reduceLROnPlat]

    model.fit(X_train, Y_train, validation_split=.2, batch_size=batch_size, epochs=epochs, callbacks=callbacks_list)
    model.summary()
    return model
