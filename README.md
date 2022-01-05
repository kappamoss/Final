# VRDL_Final_Project_The_Nature_Conservancy_Fisheries_Monitoring

The given training dataset includes 3777 images, each image contains a certain species of fish, our goal is to train a model which can classify images into the following 8 classes.

# Environment
GTX 2080 super

# Reproducing Submission (CNN+Kfold+data aug)  
Download the dataset from https://www.kaggle.com/c/the-nature-conservancy-fisheries-monitoring.  
Set the download dataset's root according to your file path.  
Run the "trainbest.py" file and save the weight.h5 (or you can use my model weight.).  
Run the "trainbest.py" file to get the CSV file.  

Note: if you modify the k fold number from "trainbest.py" file, you have to modify the k fold number in "predictbest.py" file too.  
