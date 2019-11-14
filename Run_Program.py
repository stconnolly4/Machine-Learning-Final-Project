from Plant_Animal_Classifier_Split import Plant_Animal_Classifier as PAC
import random


# subject to change on different Machines
populus_trichocarpa_dir = "populus_trichocarpa\\"
felis_catus_dir = "felis_catus\\"


# Static Classifiers for class names
class_names = {0: "felis_catus",
               1: "populus_trichocarpa"}


# Dataset
# https://www.kaggle.com/alessiocorrado99/animals10#OIF-e2bexWrojgtQnAPPcUfOWQ.jpeg
# My path to plants and animals folders.
dog_dir = "animals\\cane\\"
butterfly_dir = "animals\\farfalla\\"
cat_dir = "animals\\gatto\\"

# Static Classifiers for plant vs animal
class_namesa = {0: "cat",
                1: "butterfly"}

# Return the results
results = PAC(class_names, felis_catus_dir, populus_trichocarpa_dir)
# MUST BE PAC(class_namesa, classifierdir1, classifierdir2)
# results = PAC(class_namesa, dog_dir, butterfly_dir)
results.main_loop()

# MUST BE (CLASSIFIERDIR1, TYPE1, TYPE2)
# predictions, accuracy, type, type2 = results.predict_using_trained_model(dog_dir, "cat", "butterfly")
# print("The Accuracy for " + type + ": " + str(accuracy))
# print(predictions)


predictions, accuracy, type, type2 = results.predict_using_trained_model(cat_dir, "cat", "tree")
print("The Accuracy for " + type + ": " + str(accuracy))
# print(predictions)

if accuracy > .70:
    print("Passing " + type + " into animal category tester")
    print("Using Dog vs cat for now.")
    results_animal = PAC(class_names, cat_dir, dog_dir)
    results_animal.main_loop()
    # MUST BE PAC(class_namesa, classifierdir1, classifierdir2)
    # results = PAC(class_namesa, dog_dir, butterfly_dir)
    results_animal.main_loop()

    predictions, accuracy, type, type2 = results.predict_using_trained_model(cat_dir, "cat", "dog")
    print("The Accuracy for " + type + ": " + str(accuracy))

elif accuracy < .30:
    print("Passing " + type + " into plant category tester")
    print("Using Tree vs ... for now.")
    results_plants = PAC(class_names, populus_trichocarpa_dir, dog_dir)
    results_plants.main_loop()
    # MUST BE PAC(class_namesa, classifierdir1, classifierdir2)
    # results = PAC(class_namesa, dog_dir, butterfly_dir)
    results_plants.main_loop()

    predictions, accuracy, type, type2 = results.predict_using_trained_model(cat_dir, "tree", "...")
    print("The Accuracy for " + type + ": " + str(accuracy))
else:
    print("Network is unsure if this is a plant or animal...")
    print("Passing into a random category...")
    randompass = random.randint(0,1)
    if randompass == 0:
        print("We will try this as an animal")
    else:
        print("We will try this as a plant.")
