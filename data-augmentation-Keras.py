from keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(

    zca_whitening=False,

    rotation_range=40,

    width_shift_range=0.2,

    height_shift_range=0.2,

    shear_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True,

    fill_mode="nearest")

import os
from tqdm import tqdm
import cv2
from matplotlib import pyplot as plt

class_names = ["angry","fear","happy","neutral","sad","surprise"]
#class_names = ["surprise"]

class_names_label = {class_name:i for i, class_name in enumerate(class_names)}

#datasets = ['trainFER_data-augmentation-Keras_1','trainFER_data-augmentation-Keras_2','trainFER_data-augmentation-Keras_3']#資料夾
datasets = ['manual_FER_from FER64-2-copy']#資料夾

# Iterate through training and test sets
for dataset in datasets:
    a=0
    # Iterate through each folder corresponding to a category

    for folder in os.listdir(dataset):
        try:
            label = class_names_label[folder]
            
            
            # Iterate through each image in our folder
            for file in tqdm(os.listdir(os.path.join(dataset, folder))):
                # Get the path name of the image
                img_path = os.path.join(os.path.join(dataset, folder), file)
                #print(img_path)
                # Open and resize the img
                
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                plt.imshow(image)
                #print(image.shape)
                image = image.reshape((1,) + image.shape)
                #print(image.shape)
    
                i = 0
                for batch in datagen.flow(image, batch_size=10,
                                          save_to_dir="C:/Users/MSI/Desktop/作業/專題/CNN_face_recognition/"+ dataset +"/"+class_names[a],
                                          save_prefix= file[0:-4]+'-data-augmentation-Keras',
                                          save_format='png'):
                    plt.subplot(5,4,1 + i)
                    plt.axis("off")
                    
                    augImage = batch[0]
                    augImage = augImage.astype('float32')
                    augImage /= 255
                    plt.imshow(augImage)
                    
                    i += 1
                    if i > 14:#(19)
                        break  # otherwise the generator would loop indefinitely
            a+=1
        except:
            print("no "+folder)
print("done")
