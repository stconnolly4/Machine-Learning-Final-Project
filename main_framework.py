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
canis_familiaris_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\canis_familiaris\\"

plants_dir = [populus_trichocarpa_dir]
animals_dir = [felis_catus_dir, canis_familiaris_dir]

#felis_populus = Plant_Animal_Classifier({0: "cat", 1: "tree"}, felis_catus_dir, populus_trichocarpa_dir)
#felis_populus.main_loop()

#canis_populus = Plant_Animal_Classifier({0: "dog", 1: "tree"}, canis_familiaris_dir, populus_trichocarpa_dir)
#canis_populus.main_loop()

canis_felis = Plant_Animal_Classifier({0: "dog", 1: "cat"}, canis_familiaris_dir, felis_catus_dir)
canis_felis.main_loop()

plant_animal = Plant_Animal_Classifier({0: "plant", 1: "animal"}, plants_dir, animals_dir)
plant_animal.main_loop()


# run the classifier on a specific images
testing_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\testing_images\\"
predictions = plant_animal.predict_using_trained_model(testing_dir, plot=True)

# now loop through predictions, if it's an animal, call canis_felis

# if it comes in as a plant, call plant-specific classifier

# if it comes in as an animal, call animal-specific classifier