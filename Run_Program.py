from Plant_Animal_Classifier_Split import Plant_Animal_Classifier as PAC


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
class_namesa = {0: "tree",
                1: "cat"}

# Return the results
# results = PAC.Plant_Animal_Classifier(class_namesa, populus_trichocarpa_dir, felis_catus_dir)
results = PAC(class_namesa, populus_trichocarpa_dir, felis_catus_dir)
results.main_loop()

predictions, accuracy = results.predict_using_trained_model(populus_trichocarpa_dir)
print("The Accuracy is: " + str(accuracy))
print(predictions)
