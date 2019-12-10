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
# lycophytes_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Lycophytes\\"
# non_lycophytes_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Non-Lycophytes\\"
# monocots_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Monocots\\"
# dicots_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Dicots\\"
# dicot_1_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Dicot1\\"
# dicot_2_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Dicot2\\"
# non_vitis_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\Non-Vitis\\"
# plants_dir = "C:\\Users\\samic\\Documents\\Photos for Machine Learning\\plants\\"




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

bos_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\bos\\"
canis_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\canis\\"
equus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\equus\\"
felis_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\felis\\"
gallus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\gallus\\"
nongallus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\nongallus\\"
# livestock_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\livestock\\"
loxodonota_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\loxodonta\\"
# nonequus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\nonequus\\"
nonsciurmorpha_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\nonsciurmorpha\\"
ovis_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\ovis\\"
# pets_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\pets\\"
sciurmorpha_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\animals\\sciuromorpha\\"

# animals
# canis_felis = Plant_Animal_Classifier({0: "canis", 1: "felis"}, canis_dir, felis_dir)
# canis_felis.main_loop()
# bos_ovis = Plant_Animal_Classifier({0: "bos", 1: "ovis"}, bos_dir, ovis_dir)
# bos_ovis.main_loop()

# livestock_dir = [bos_dir, ovis_dir]
# pets_dir = [canis_dir, felis_dir]
#
# livestock_pets = Plant_Animal_Classifier({0: "livestock", 1: "pets"}, livestock_dir, pets_dir)
# livestock_pets.main_loop(downsample_second_list=True)

# Equus vs. Non-Equus

# nonequus_dir = [ovis_dir, bos_dir, canis_dir, felis_dir]
# equus_nonequus = Plant_Animal_Classifier({0: "equus", 1: "nonequus"}, equus_dir, nonequus_dir)
# equus_nonequus.main_loop(downsample_second_list=True)

# Sciuromorpha vs. Non-Sciuromorpha
# nonsciuromorpha_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir]
# sciuromorpha_nonsciuromorpha = Plant_Animal_Classifier({0: "sciuromorpha", 1: "nonsciuromorpha"}, sciurmorpha_dir, nonsciuromorpha_dir)
# sciuromorpha_nonsciuromorpha.main_loop(downsample_second_list=True)

# #loxodonta vs non-loxodonta
# nonloxodonta_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir, sciurmorpha_dir]
# loxodonta_nonloxodonta = Plant_Animal_Classifier({0: "loxodonta", 1: "nonloxodonta"}, loxodonota_dir, nonloxodonta_dir)
# loxodonta_nonloxodonta.main_loop(downsample_second_list=True)

#gallus vs non-gallus
# nongallus_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir, sciurmorpha_dir, loxodonota_dir]
# gallus_nongallus = Plant_Animal_Classifier({0: "gallus", 1: "nongallus"}, gallus_dir, nongallus_dir)
# gallus_nongallus.main_loop(downsample_second_list=True)


# # Train the animals using CNN
# gallus_nongallus = CNN_Classification({0: "gallus", 1: "nongallus"}, gallus_dir, nongallus_dir)
# gallus_nongallus.main_loop()





# plant vs animal
# -------------------------------------------------------------------------------------------------------------------------
allplants_dir = [populus_trichocarpa_dir, medicago_truncatula_dir, arabidopsis_thaliana_dir, carica_papaya_dir,
                 vitis_vinifera_dir, oryza_sativa_dir, sorghum_bicolor_dir, selaginella_moellendorffii_dir]
allanimals_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir, sciurmorpha_dir, loxodonota_dir, gallus_dir]
animals_plants = Plant_Animal_Classifier({0: "plant", 1: "animal"}, allplants_dir, allanimals_dir)
animals_plants.main_loop(downsample_second_list=True)
testing_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\testing_images\\"
trial1 = animals_plants.predict_using_trained_model(testing_dir)
print(trial1)
if trial1[0] != "animal":
    lycophytes_dir = [selaginella_moellendorffii_dir]
    nonlycophytes_dir = [sorghum_bicolor_dir, medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir,
                         populus_trichocarpa_dir, oryza_sativa_dir, vitis_vinifera_dir]
    lycophytes_nonlycophytes = Plant_Animal_Classifier({0: "lycophyte", 1: "non-lycophyte"}, lycophytes_dir, nonlycophytes_dir)
    lycophytes_nonlycophytes.main_loop(True)
    trial2 = lycophytes_nonlycophytes.predict_using_trained_model(testing_dir)
    print("Trial 2:")
    print(trial2)
    if trial2[0] != "lycophyte":
        monocots_dir = [sorghum_bicolor_dir, oryza_sativa_dir]
        dicots_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, vitis_vinifera_dir]
        monocot_dicot = Plant_Animal_Classifier({0: "monocot", 1: "dicot"}, monocots_dir, dicots_dir)
        monocot_dicot.main_loop(downsample_second_list=True)
        trial3 = monocot_dicot.predict_using_trained_model(testing_dir)
        print("Trial 3:")
        print(trial3)
        if trial3[0] != "monocot":
            nonvitis_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir]
            vitis_nonvitis = Plant_Animal_Classifier({0: "vitis", 1: "non-vitis"}, vitis_vinifera_dir, non_vitis_dir)
            vitis_nonvitis.main_loop(downsample_second_list=True)
            trial4 = vitis_nonvitis.predict_using_trained_model(testing_dir)
            print("Trial 4:")
            print(trial4)
            if trial4[0] != "vitis":
                dicot_1_dir = [carica_papaya_dir, arabidopsis_thaliana_dir]
                dicot_2_dir = [medicago_truncatula_dir, populus_trichocarpa_dir]
                dicot1_dicot2 = Plant_Animal_Classifier({0: "dicot1", 1: "dicot2"}, dicot_1_dir, dicot_2_dir)
                dicot1_dicot2.main_loop(downsample_second_list=True)
                trial5 = dicot1_dicot2.predict_using_trained_model(testing_dir)
                print("Trial 5:")
                print(trial5)
                if trial5[0] != "dicot1":
                    medicago_populus = Plant_Animal_Classifier({0: "medicago", 1: "populus"}, medicago_truncatula_dir, populus_trichocarpa_dir)
                    medicago_populus.main_loop(True)
                    trial6 = medicago_populus.predict_using_trained_model(testing_dir)
                    print("Trial 6:")
                    print(trial6)
                    if trial6[0] != "medicago":
                        print("It's Black cottonwood")
                    else:
                        print("It's Medicago truncatula")
                else:
                    arabidopsis_carica = Plant_Animal_Classifier({0: "arabidopsis", 1: "carica"}, arabidopsis_thaliana_dir, carica_papaya_dir)
                    arabidopsis_carica.main_loop(downsample_second_list=True)
                    trial6 = arabidopsis_carica.predict_using_trained_model(testing_dir)
                    print("Trial 6:")
                    print(trial6)
                    if trial6[0] != "arabidopsis":
                        print("It's a Papaya!")
                    else:
                        print("It's Thale cress!")
            else:
                print("Its a common grape vine!")
        else:
            oryza_sorghum = Plant_Animal_Classifier({0: "oryza", 1: "sorghum"}, oryza_sativa_dir, sorghum_bicolor_dir)
            oryza_sorghum.main_loop(downsample_second_list=True)
            trial4 = oryza_sorghum.predict_using_trained_model(testing_dir)
            if trial4[0] != "oryza":
                print("It's Broom Corn!")
            else:
                print("It's Rice!")
    else:
        print("It's a Fern!")
else:
    nongallus_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir, sciurmorpha_dir, loxodonota_dir]
    gallus_nongallus = Plant_Animal_Classifier({0: "gallus", 1: "nongallus"}, gallus_dir, nongallus_dir)
    gallus_nongallus.main_loop(downsample_second_list=True)
    trial2 = gallus_nongallus.predict_using_trained_model(testing_dir)
    print("Trial 2:")
    print(trial2)
    if trial2[0] != "gallus":
        nonloxodonta_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir, sciurmorpha_dir]
        loxodonta_nonloxodonta = Plant_Animal_Classifier({0: "loxodonta", 1: "nonloxodonta"}, loxodonota_dir, nonloxodonta_dir)
        loxodonta_nonloxodonta.main_loop(downsample_second_list=True)
        trial3 = loxodonta_nonloxodonta.predict_using_trained_model(testing_dir)
        print("Trial 3:")
        print(trial3)
        if trial3[0] != "loxodonta":
            #Sciuromorpha vs. Non-Sciuromorpha
            nonsciuromorpha_dir = [ovis_dir, bos_dir, canis_dir, felis_dir, equus_dir]
            sciuromorpha_nonsciuromorpha = Plant_Animal_Classifier({0: "sciuromorpha", 1: "nonsciuromorpha"}, sciurmorpha_dir, nonsciuromorpha_dir)
            sciuromorpha_nonsciuromorpha.main_loop(downsample_second_list=True)
            trial4 = sciuromorpha_nonsciuromorpha.predict_using_trained_model(testing_dir)
            print("Trial4:")
            print(trial4)
            if trial4[0] != "sciuromorpha":
                nonequus_dir = [ovis_dir, bos_dir, canis_dir, felis_dir]
                equus_nonequus = Plant_Animal_Classifier({0: "equus", 1: "nonequus"}, equus_dir, nonequus_dir)
                equus_nonequus.main_loop(downsample_second_list=True)
                trial5 = equus_nonequus.predict_using_trained_model(testing_dir)
                print("Trial 5")
                print(trial5)
                if trial5[0] != "equus":
                    livestock_dir = [bos_dir, ovis_dir]
                    pets_dir = [canis_dir, felis_dir]
                    livestock_pets = Plant_Animal_Classifier({0: "livestock", 1: "pets"}, livestock_dir, pets_dir)
                    livestock_pets.main_loop(downsample_second_list=True)
                    trial6 = livestock_pets.predict_using_trained_model(testing_dir)
                    print("Trial 6:")
                    print(trial6)
                    if trial6[0] != "livestock":
                        canis_felis = Plant_Animal_Classifier({0: "canis", 1: "felis"}, canis_dir, felis_dir)
                        canis_felis.main_loop(downsample_second_list=True)
                        trial7 = canis_felis.predict_using_trained_model(testing_dir)
                        print("Trial 7:")
                        print(trial7)
                        if trial7[0] != "canis":
                            print("It's a Cat!")
                        else:
                            print("It's a dog!")
                    else:
                        bos_ovis = Plant_Animal_Classifier({0: "bos", 1: "ovis"}, bos_dir, ovis_dir)
                        bos_ovis.main_loop(downsample_second_list=True)
                        trial7 = bos_ovis.predict_using_trained_model(testing_dir)
                        print("Trial 7:")
                        print(trial7)
                        if trial7[0] != "bos":
                            print("Its a sheep!")
                        else:
                            print("It's a cow!")
                else:
                    print("It's a horse!")
            else:
                print("Its a squirrell!")
        else:
            print("Its an elephant!")
    else:
        print("It's a chicken!")

# -------------------------------------------------------------------------------------------------------------------------




# plants
# plants_dir = [sorghum_bicolor_dir, medicago_truncatula_dir, selaginella_moellendorffii_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, oryza_sativa_dir, vitis_vinifera_dir]
# 3 main groups
# lycophytes_dir = [selaginella_moellendorffii_dir]
# nonlycophytes_dir = [sorghum_bicolor_dir, medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, oryza_sativa_dir, vitis_vinifera_dir]
# monocots_dir = [sorghum_bicolor_dir, oryza_sativa_dir]
# dicots_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir, vitis_vinifera_dir]

# classifiers

# lycophytes_nonlycophytes = CNN_Classification({0: "lycophyte", 1: "non-lycophyte"}, lycophytes_dir, non_lycophytes_dir)
# lycophytes_nonlycophytes.main_loop()
# lycophytes_nonlycophytes = Plant_Animal_Classifier({0: "lycophyte", 1: "non-lycophyte"}, lycophytes_dir, nonlycophytes_dir)
# #lycophytes_nonlycophytes.main_loop(True)

# monocot_dicot = Plant_Animal_Classifier({0: "monocot", 1: "dicot"}, monocots_dir, dicots_dir)
# monocot_dicot.main_loop(True)
#monocot_dicot = CNN_Classification(["monocot", "dicot"], monocots_dir, dicots_dir)
#monocot_dicot.main_loop()

# within monocots
#oryza_sorghum =  Plant_Animal_Classifier({0: "oryza", 1: "sorghum"}, oryza_sativa_dir, sorghum_bicolor_dir)
#oryza_sorghum.main_loop(True)
#oryza_sorghum =  CNN_Classification({0: "oryza", 1: "sorghum"}, oryza_sativa_dir, sorghum_bicolor_dir)
#oryza_sorghum.main_loop()


# within dicots
# nonvitis_dir = [medicago_truncatula_dir, carica_papaya_dir, arabidopsis_thaliana_dir, populus_trichocarpa_dir]
#vitis_nonvitis = Plant_Animal_Classifier({0: "vitis", 1: "non-vitis"}, vitis_vinifera_dir, non_vitis_dir)
#vitis_nonvitis.main_loop(True)
# vitis_nonvitis = CNN_Classification({0: "vitis", 1: "non-vitis"}, vitis_vinifera_dir, non_vitis_dir)
# vitis_nonvitis.main_loop()

# dicot_1_dir = [carica_papaya_dir, arabidopsis_thaliana_dir]
# dicot_2_dir = [medicago_truncatula_dir, populus_trichocarpa_dir]
#dicot1_dicot2 = Plant_Animal_Classifier({0: "dicot1", 1: "dicot2"}, dicot_1_dir, dicot_2_dir)
#dicot1_dicot2.main_loop(True)
#dicot1_dicot2 = CNN_Classification({0: "dicot1", 1: "dicot2"}, dicot_1_dir, dicot_2_dir)
#dicot1_dicot2.main_loop()

#arabidopsis_carica = Plant_Animal_Classifier({0: "arabidopsis", 1: "carica"}, arabidopsis_thaliana_dir, carica_papaya_dir)
#arabidopsis_carica.main_loop(True)
#arabidopsis_carica = CNN_Classification({0: "arabidopsis", 1: "carica"}, arabidopsis_thaliana_dir, carica_papaya_dir)
#arabidopsis_carica.main_loop()
#
#medicago_populus = Plant_Animal_Classifier({0: "medicago", 1: "populus"}, medicago_truncatula_dir, populus_trichocarpa_dir)
#medicago_populus.main_loop(True)
# medicago_populus = CNN_Classification({0: "medicago", 1: "populus"}, medicago_truncatula_dir, populus_trichocarpa_dir)
# medicago_populus.main_loop()


# animals_dir = [felis_catus_dir, canis_familiaris_dir]

#canis_felis = Plant_Animal_Classifier({0: "dog", 1: "cat"}, canis_familiaris_dir, felis_catus_dir)
#canis_felis.main_loop()
#
# plant_animal = Plant_Animal_Classifier({0: "plant", 1: "animal"}, plants_dir, animals_dir)
# plant_animal.main_loop()