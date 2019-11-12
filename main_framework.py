# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:01:56 2019

@author: samic
"""
from Plant_Animal_Classifier_Split import Plant_Animal_Classifier
# read in images


### plant vs animal classifier ###

# train classifier
felis_catus_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\felis_catus\\"
populus_trichocarpa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\populus_trichocarpa\\"
plant_animal = Plant_Animal_Classifier(felis_catus_dir, populus_trichocarpa_dir)
plant_animal.main_loop()

# run the classifier on a specific images
# testing_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\testing_images\\"
predictions = plant_animal.predict_using_trained_model(populus_trichocarpa_dir)
print(predictions)



# if it comes in as a plant, call plant-specific classifier

# if it comes in as an animal, call animal-specific classifier