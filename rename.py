import os
# importing os module
name = "dicot2"
# path = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\"+name+"\\"
path = "C:\\Users\\djenz\\OneDrive - University of Vermont\\Machine-Learning-Final-Project\\plants\\Dicot2\\"

i = 0
for filename in os.listdir(path):
    os.rename(os.path.join(path,filename), os.path.join(path, name+'_pic_'+str(i)+'.jpg'))
    i += 1