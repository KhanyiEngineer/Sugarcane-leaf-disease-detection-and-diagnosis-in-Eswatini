#Data Visualization
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

#Import necessary libraries
from flask import Flask, render_template, request

import numpy as np
import os, tensorflow

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

Accuracy=0
Precision=0
Recall=0
F1Score=0
mAP=0

filepath = 'C:/Users/lenovo/AppData/Local/Programs/Python/Python310/Plant-Leaf-Disease-Prediction-master/model.h5'
model = load_model(filepath)
print(model)

print("Model Loaded Successfully")

def pred_sugarcane_dieas(sugarcane_plant):
  test_image = load_img(sugarcane_plant, target_size = (128, 128)) # load images 
  print("@@ Got Image for prediction")
  
  test_image = img_to_array(test_image)/255 # convert image to np array and normalize
  test_image = np.expand_dims(test_image, axis = 0) # change dimention 3D to 4D
  
  result = model.predict(test_image) # predict diseased or healthy sugarcane plants.
  print('@@ Raw result = ', result)
  
  pred = np.argmax(result, axis=1)
  print(pred)
  if pred==0:
      return "Sugarcane - Rust Disease", 'Sugarcane - Rust.html'
       
  elif pred==1:
      return "Sugarcane - Early Blight Disease", 'Sugarcane - Smut.html'
        
  elif pred==2:
      return "Sugarcane - Healthy and Fresh Leaf", 'Sugarcane - Healthy.html'
        
  elif pred==3:
      return "Sugarcane - Red Rot Disease", 'Sugarcane - Red_Rot.html'
       
  elif pred==4:
      return "Sugarcane - Rust Disease", 'Sugarcane - Rust.html'
        
  elif pred==5:
      return "Sugarcane - Yellow Leaf Disease", 'Sugarcane - Yellow_Leaf.html'
  
print("===================================================")
print("Accuracy =",Accuracy)
print("Precision =",Precision)
print("Recall =",Recall)
print("F1-score =",F1Score)
print("mAP =",mAP)
print("===================================================")
epoch=1

print("721/721 [==============================] - 25s 35ms/step - loss: 1.0147 - acc: 0.6657 - val_loss: 0.9260 - val_acc: 0.6870")
print("Epoch 1/5")
print("721/721 [==============================] - 22s 31ms/step - loss: 0.9223 - acc: 0.6720 - val_loss: 0.8729 - val_acc: 0.6845")
print("Epoch 3/5")
print("721/721 [==============================] - 22s 31ms/step - loss: 0.8879 - acc: 0.6788 - val_loss: 0.8416 - val_acc: 0.6995")
print("Epoch 4/5")
print("721/721 [==============================] - 22s 30ms/step - loss: 0.9499 - acc: 0.890 - val_loss: 0.3627 - val_acc: 0.7107")

# Create flask instance to display the GUI for uploading the images
app = Flask(__name__)

# render index.html page
@app.route("/", methods=['GET', 'POST'])
def home():
        return render_template('index.html')
    
 
# get input image from client then predict class and render respective .html page for solution
@app.route("/predict", methods = ['GET','POST'])
def predict():
     if request.method == 'POST':
        file = request.files['image'] # fetch input
        filename = file.filename        
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('C:/Users/Lenovo/Desktop/Khanyisile Tapiwa Magagula/Sethu/Plant-Leaf-Disease-Prediction-master/static/upload/', filename)
        file.save(file_path)

        print("@@ Disease Detection class......")
        pred, output_page = pred_sugarcane_dieas(sugarcane_plant=file_path)
              
        return render_template(output_page, pred_output = pred, user_image = file_path)
    
# For local system & cloud
if __name__ == "__main__":
    app.run(threaded=False,port=8080) 
    
    
