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
dog_dir = "animals\\cane\\"
butterfly_dir = "animals\\farfalla\\"

# Static Classifiers for plant vs animal
class_namesa = {0: "dog",
                1: "butterfly"}

# Return the results
# results = PAC.Plant_Animal_Classifier(class_namesa, populus_trichocarpa_dir, felis_catus_dir)
# MUST BE PAC(class_namesa, classifierdir1, classifierdir2)
results = PAC(class_namesa, dog_dir, butterfly_dir)
results.main_loop()

# MUST BE (CLASSIFIERDIR1, TYPE1, TYPE2)
predictions, accuracy, type, type2 = results.predict_using_trained_model(dog_dir, "dog", "butterfly")
print("The Accuracy for " + type + ": " + str(accuracy))
print(predictions)
