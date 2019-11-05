import Plant_Animal_Classifier_Split as PAC

# subject to change on different Machines
populus_trichocarpa_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\images final\\populus_trichocarpa\\"
felis_catus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\images final\\felis_catus\\"


# Static Classifiers for class names
class_names = {0: "populus_trichocarpa",
               1: "felis_catus"}

# Return the results
results = PAC.Plant_animal_Classifier(class_names, populus_trichocarpa_dir, felis_catus_dir)
results.main_loop()
