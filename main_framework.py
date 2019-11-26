# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:01:56 2019

@author: samic
"""
from Plant_Animal_Classifier_Split import Plant_Animal_Classifier
import pickle


### plant vs animal classifier ###

# train classifier
felis_catus_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\felis_catus\\"
canis_familiaris_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\canis_familiaris\\"
populus_trichocarpa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\populus_trichocarpa\\"
oryza_sativa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\oryza_sativa\\"
vitis_vinifera_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\vitis_vinifera\\"

plants_dir = [populus_trichocarpa_dir, oryza_sativa_dir]
animals_dir = [felis_catus_dir, canis_familiaris_dir, vitis_vinifera_dir]

#felis_populus = Plant_Animal_Classifier({0: "cat", 1: "tree"}, felis_catus_dir, populus_trichocarpa_dir)
#felis_populus.main_loop()

#canis_populus = Plant_Animal_Classifier({0: "dog", 1: "tree"}, canis_familiaris_dir, populus_trichocarpa_dir)
#canis_populus.main_loop()

#canis_felis = Plant_Animal_Classifier({0: "dog", 1: "cat"}, canis_familiaris_dir, felis_catus_dir)
#canis_felis.main_loop()

#populus_oryza = Plant_Animal_Classifier({0: "populus tricocharpa", 1: "oryza sativa"}, populus_trichocarpa_dir, oryza_sativa_dir)
#populus_oryza.main_loop()

#populus_vitis = Plant_Animal_Classifier({0: "populus tricocharpa", 1: "vitis vinifera"}, populus_trichocarpa_dir, vitis_vinifera_dir)
#populus_vitis.main_loop()

plant_animal = Plant_Animal_Classifier({0: "plant", 1: "animal"}, plants_dir, animals_dir)
plant_animal.main_loop()

plant_animal.save_pickle("plant_animal")

# run the classifier on a specific images
testing_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\testing_images\\"
predictions = plant_animal.predict_using_trained_model(testing_dir, plot=True)

# now loop through predictions, if it's an animal, call canis_felis

# if it comes in as a plant, call plant-specific classifier

# if it comes in as an animal, call animal-specific classifier