import numpy as np
import tensorflow as tf
from src.losses import dice_coef, dice_loss, bce_dice_loss

def test_dice_coef_and_loss():
    y_true = tf.constant([[1, 0, 1, 0]], dtype=tf.float32)
    y_pred = tf.constant([[1, 0, 0, 0]], dtype=tf.float32)
    coef = dice_coef(y_true, y_pred)
    loss = dice_loss(y_true, y_pred)
    assert 0 <= coef <= 1
    assert 0 <= loss <= 1

def test_bce_dice_loss():
    y_true = tf.constant([[1., 0., 1., 0.]], dtype=tf.float32)
    y_pred = tf.constant([[0.8, 0.2, 0.6, 0.1]], dtype=tf.float32)
    loss = bce_dice_loss(y_true, y_pred)
    assert loss is not None
