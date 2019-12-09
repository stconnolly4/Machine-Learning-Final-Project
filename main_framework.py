# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 13:01:56 2019

@author: samic
"""
from Plant_Animal_Classifier_Split import Plant_Animal_Classifier
from CNN_Classification import CNN_Classification
import numpy as np


#CNN_Classification({0: "monocot", 1: "dicot"}, monocots_dir, dicots_dir)



### plant vs animal classifier ###

# train classifier
# felis_catus_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\felis_catus\\"
# canis_familiaris_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\canis_familiaris\\"
# populus_trichocarpa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\populus_trichocarpa\\"
# oryza_sativa_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\oryza_sativa\\"
# vitis_vinifera_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\vitis_vinifera\\"
# arabidopsis_thaliana_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\arabidopsis_thaliana\\"
# carica_papaya_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\carica_papaya\\"
# selaginella_moellendorffii_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\selaginella_moellendorffii\\"
# medicago_truncatula_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\medicago_truncatula\\"
# sorghum_bicolor_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\sorghum_bicolor\\"

populus_trichocarpa_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\populus_trichocarpa\\"
oryza_sativa_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\oryza_sativa\\"
vitis_vinifera_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\vitis_vinifera\\"
arabidopsis_thaliana_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\arabidopsis_thaliana\\"
carica_papaya_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\carica_papaya\\"
selaginella_moellendorffii_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\selaginella_moellendorffii\\"
medicago_truncatula_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\medicago_truncatula\\"
sorghum_bicolor_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\sorghum_bicolor\\"
lycophytes_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Lycophytes\\"
non_lycophytes_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Non-Lycophytes\\"
monocots_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Monocots\\"
dicots_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Dicots\\"
dicot_1_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Dicot1\\"
dicot_2_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Dicot2\\"
non_vitis_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Non-Vitis\\"
plants_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\plants\\"

felis_catus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\gatto\\"
felis_catus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\felis_catus\\"
canis_familiaris_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\cane\\"
canis_familiaris_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\canis_familiaris\\"




# plants
# plants_dir = [sorghum_bicolor_dir, medicago_truncatula_dir, selaginella_moellendorffii_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, oryza_sativa_dir, vitis_vinifera_dir]

# 3 main groups
# lycophytes_dir = [selaginella_moellendorffii_dir]
# nonlycophytes_dir = [sorghum_bicolor_dir, medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, oryza_sativa_dir, vitis_vinifera_dir]
# monocots_dir = [sorghum_bicolor_dir, oryza_sativa_dir]
# dicots_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, vitis_vinifera_dir]

# classifiers
# lycophytes_nonlycophytes = Plant_Animal_Classifier({0: "lycophyte", 1: "non-lycophyte"}, lycophytes_dir, nonlycophytes_dir)
# #lycophytes_nonlycophytes.main_loop(True)
# lycophytes_nonlycophytes.main_loop(True)

# classifiers
# lycophytes_nonlycophytes = CNN_Classification({0: "lycophyte", 1: "non-lycophyte"}, lycophytes_dir, non_lycophytes_dir)
# #lycophytes_nonlycophytes.main_loop(True)
# lycophytes_nonlycophytes.main_loop()


# monocot_dicot = Plant_Animal_Classifier({0: "monocot", 1: "dicot"}, monocots_dir, dicots_dir)
# monocot_dicot.main_loop(True)

# # classifiers
monocot_dicot = CNN_Classification({0: "monocot", 1: "dicot"}, monocots_dir, dicots_dir)
#lycophytes_nonlycophytes.main_loop(True)
monocot_dicot.main_loop()

# within monocots
oryza_sorghum =  Plant_Animal_Classifier({0: "oryza", 1: "sorghum"}, oryza_sativa_dir, sorghum_bicolor_dir)
oryza_sorghum.main_loop(True)

# within dicots
# nonvitis_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir]
vitis_nonvitis = Plant_Animal_Classifier({0: "vitis", 1: "non-vitis"}, vitis_vinifera_dir, non_vitis_dir)
vitis_nonvitis.main_loop(True)

# dicot_1_dir = [carica_papaya_dir, arabidopsis_thaliana_dir]
# dicot_2_dir = [medicago_truncatula_dir, populus_trichocarpa_dir]
dicot1_dicot2 = Plant_Animal_Classifier({0: "dicot1", 1: "dicot2"}, dicot_1_dir, dicot_2_dir)
dicot1_dicot2.main_loop(True)

arabidopsis_carica = Plant_Animal_Classifier({0: "arabidopsis", 1: "carica"}, arabidopsis_thaliana_dir, carica_papaya_dir)
arabidopsis_carica.main_loop(True)

medicago_populus = Plant_Animal_Classifier({0: "medicago", 1: "populus"}, medicago_truncatula_dir, populus_trichocarpa_dir)
medicago_populus.main_loop(True)


animals_dir = [felis_catus_dir, canis_familiaris_dir]

#canis_felis = Plant_Animal_Classifier({0: "dog", 1: "cat"}, canis_familiaris_dir, felis_catus_dir)
#canis_felis.main_loop()
#
plant_animal = Plant_Animal_Classifier({0: "plant", 1: "animal"}, plants_dir, animals_dir)
plant_animal.main_loop()

#plant_animal.save_to_file_test('Plant_Animal.h5')
#from tensorflow import keras
#plant_animal = keras.models.load_model('Plant_Animal.h5')

#plant_animal.save_to_file_test('Plant_Animal.h5')
#from tensorflow import keras
#plant_animal_save = keras.models.load_model('Plant_Animal.h5')


# run the classifier on a specific images
# testing_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\testing_images\\"
testing_dir = "C:\\Users\\djenz\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\testing_images\\"

predictions = plant_animal.predict_using_trained_model(testing_dir, plot=True)
#
## now loop through predictions, if it's an animal, call canis_felis
#import os
#import shutil
#print(predictions)
#all_images_directory = [testing_dir + "{}".format(i) for i in os.listdir(testing_dir)]
#dst = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\What the machine thinks\\hold\\"
## print(all_images_directory)
#for i in range(len(predictions)):
#    if predictions[i] == "animal":
#        #print(all_images_directory[i])
#        src = all_images_directory[i]
#        shutil.copy(src, dst)
#    #if predictions[i] == "plant":
#        # TODO
#
#dog_cat = Plant_Animal_Classifier({0: "cat", 1: "dog"}, felis_catus_dir, canis_familiaris_dir)
#dog_cat.main_loop()
#dog_or_cat_predictions = dog_cat.predict_using_trained_model(dst, plot=True)
#
#print(dog_or_cat_predictions)
#
#all_images_directory_machine_think = [dst + "{}".format(i) for i in os.listdir(dst)]
#dst_cat = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\What the machine thinks\\cat\\"
#dst_dog = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\What the machine thinks\\dog\\"
#for i in range(len(dog_or_cat_predictions)):
#    if dog_or_cat_predictions[i] == "cat":
#        #print(all_images_directory[i])
#        src = all_images_directory_machine_think[i]
#        shutil.copy(src, dst_cat)
#    if dog_or_cat_predictions[i] == "dog":
#        src = all_images_directory_machine_think[i]
#        shutil.copy(src, dst_dog)

# if it comes in as a plant, call plant-specific classifier

# if it comes in as an animal, call animal-specific classifier