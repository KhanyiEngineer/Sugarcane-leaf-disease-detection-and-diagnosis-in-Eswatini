import cv2
from matplotlib import pyplot as plt
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model

filepath = 'C:/Users/lenovo/AppData/Local/Programs/Python/Python310/Plant-Leaf-Disease-Prediction-master/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")


sugarcane_plant = cv2.imread("C://Users/lenovo//AppData//Local//Programs//Python//Python310//Plant-Leaf-Disease-Prediction-master//Dataset//test//Sugarcane___Red_Rot (2).jpeg")

# Assuming sugarcane_plant contains the file path to the image
print("C://Users//lenovo//AppData//Local//Programs//Python//Python310//Plant-Leaf-Disease-Prediction-master//Dataset//test//Sugarcane___Red_Rot (2).jpeg", sugarcane_plant)  # Debug print to check the file path

#if sugarcane_plant is None:
    #print("Error: Unable to load image")
sugarcane_plant = cv2.resize(sugarcane_plant, (128,128)) # load image
    #cv2.imshow(sugarcane_plant, test_image)
    #pixels.append(sugarcane_plant)
  
sugarcane_plant = img_to_array(sugarcane_plant)/255 # convert image to np array and normalize
sugarcane_plant = np.expand_dims(sugarcane_plant, axis = 0) # change dimention 3D to 4D
  
result = model.predict(sugarcane_plant) # predict diseased plant or not
 
pred = np.argmax(result, axis=1)
print(pred)
if pred==0:
    print( "Sugarcane - Rust Disease")
       
elif pred==1:
    print("Sugarcane - Smut Disease")
        
elif pred==2:
    print("Sugarcane - Healthy and Fresh")
    
elif pred==3:
      print("Sugarcane - Red Rot Disease")
       
elif pred==4:
    print("Sugarcane - Rust Disease")
        
elif pred==5:
    print("Sugarcane - Yellow Disease")




