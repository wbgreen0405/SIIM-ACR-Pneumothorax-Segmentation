# SIIM-pneumothorax-segmentation
<img src='img\header.png'>

Kaggle competition to identify pneumothorax disease on chest X-rays and if a present segment it. 

The dataset is provided via the Google Healthcare Cloud and consists of a set of training images and a set of test images, both in DICOM format, and a csv file of image labels in the form of run length encoded (RLE) segmentation masks. 

A description of the clinical problem and detailed analysis of the dataset is given in this notebook [SIIM_ACR_Pneumothorax_Competition_Data_Analysis.ipynb](SIIM_ACR_Pneumothorax_Competition_Data_Analysis.ipynb)


A deep learning model was constructed using a resnet34 architecture combined with a Unet and trained on approximately 10,000 X-rays and corresponding segmentation masks. The competition metric was the DICE co-efficient, a commonly used metric for image segmentation tasks. The code for the model is given in the notebook [SIIM_pneumothorax_model.ipynb](SIIM_pneumothorax_model.ipynb)

Example of an X=ray image from the validation set, the accompanying segmentation mask and prediction.

<img src='img\predictions.png'>

Predictions were made on a test set of around 1000 images and the results submitted to kaggle. Achieved place 413/1050 on the public leaderboard. 

Notebooks: 

1. [SIIM_ACR_Pneumothorax_Competition_Data_Analysis.ipynb](SIIM_ACR_Pneumothorax_Competition_Data_Analysis.ipynb) by: [Ekhtiar Syed](https://www.kaggle.com/ekhtiar)
2. [Unet w/ SE Resnet50 32x4d enconder](unet-with-se-resnext50-32x4d-encoder-for-stage-2.ipynb)
