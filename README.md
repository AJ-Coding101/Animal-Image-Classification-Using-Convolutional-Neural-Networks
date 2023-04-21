# Animal-Image-Classification-Using-Convolutional-Neural-Networks

### <u>Summary</u> ###

Using CNN, built and trained 2 models that can detect animal images and classify them accordingly. The models were deployed using Flask (My_FlaskApp.py).

Since we are dealing with a large dataset, we used DVC as well as Git for version control.

Google Colab(cnn_colab.ipynb) was used for training and saving the 2 models below of which ResNet152V2 was slightly more accurate:

- ResNet_Model.h5
- Inception_Model.h5

Dataset used: https://www.kaggle.com/datasets/antoreepjana/animals-detection-images-dataset (Saved as archive.zip)

 
Created 2 DVC remote repositories(origin and dagshub) which can be obtained after pull:

$ dvc remote list <br>
origin  gdrive://1cYGsOkejFhErWtg_L5d03cxQ-K9Z-iHl <br>
dagshub https://dagshub.com/AJ-Coding101/Animal-Image-Classification-Using-Convolutional-Neural-Networks.dvc

Trained h5 models were larger than allowed Git size of 100MB and thus pushed to DVC as well.