import Plant_Animal_Classifier_Split as PAC

# subject to change on different Machines
populus_trichocarpa_dir = "populus_trichocarpa\\"
felis_catus_dir = "felis_catus\\"


# Static Classifiers for class names
class_names = {0: "populus_trichocarpa",
               1: "felis_catus"}


# Dataset
# https://www.kaggle.com/alessiocorrado99/animals10#OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg
# My path to plants and animals folders.
plants_dir = "animals\\cane\\"
animals_dir = "animals\\farfalla\\"

# Static Classifiers for plant vs animal
class_namesa = {0: "cane",
                1: "farfalla"}

# Return the results
results = PAC.Plant_animal_Classifier(class_namesa, plants_dir, animals_dir)
results.main_loop()
