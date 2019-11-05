from __future__ import absolute_import, division, print_function, unicode_literals

class Plant_animal_Classifier:
    def __init__(self, class_name, plant_image_dir, animal_image_dir):

        self.class_name = class_name
        self.Pimages = plant_image_dir
        self.Aimages = animal_image_dir


populus_trichocarpa_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\images final\\populus_trichocarpa\\"
felis_catus_dir = "C:\\Users\\djenz\\OneDrive - University of Vermont\\images final\\felis_catus\\"


class_names = {0: "populus_trichocarpa",
               1: "felis_catus"}

pic1 = Plant_animal_Classifier(class_names, populus_trichocarpa_dir, felis_catus_dir)


