import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Enable eager execution
tf.compat.v1.enable_eager_execution()

# Configure GPU options
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)

# Define image augmentation
aug = ImageDataGenerator(
    rotation_range=20,
    zoom_range=0.15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.15,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Convolutional Neural Network model building
classifier = Sequential()

# Step 1 - Convolution layer
classifier.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(32, (3, 3), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Flattening layer
classifier.add(Flatten())

# Step 4 - Fully connected layer
classifier.add(Dense(units=128, activation='relu'))

# Output layer with 6 units (since you have 6 classes)
classifier.add(Dense(units=6, activation='softmax'))  # Use softmax activation for multi-class classification

# Compile the CNN
classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


# Load and augment training data
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
training_set = train_datagen.flow_from_directory(
    'C:/Users/lenovo/AppData/Local/Programs/Python/Python310/Plant-Leaf-Disease-Prediction-master/Dataset/train',
    target_size=(128, 128),
    batch_size=6,
    class_mode='categorical'
)

# Load and preprocess validation data
test_datagen = ImageDataGenerator(rescale=1./255)
valid_set = test_datagen.flow_from_directory(
    'C:/Users/lenovo/AppData/Local/Programs/Python/Python310/Plant-Leaf-Disease-Prediction-master/Dataset/val',
    target_size=(128, 128),
    batch_size=3,
    class_mode='categorical'
)

# Train the classifier
classifier.fit(
    training_set,
    steps_per_epoch=20,
    epochs=50,
    validation_data=valid_set
)

# Save the model
classifier_json = classifier.to_json()
with open("model1.json", "w") as json_file:
    json_file.write(classifier_json)
classifier.save_weights("my_model_weights.weights.h5")
classifier.save("model.h5")
print("Saved model to disk drive")
