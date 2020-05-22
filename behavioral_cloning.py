import cv2
import csv 
import numpy as np 
import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Read training images info
lines = []
with open('./recorded_data/driving_log.csv') as csvfile:
    csv_reader = csv.reader(csvfile)
    for line in csv_reader:
        lines.append(line)

# Read images and Steering measurement
images = []
steering = []
for line in lines:
    source_path = line[0]
    file_name = source_path.split('\\')[-1] # Note the path delimiter for windows 
    current_path = './recorded_data/IMG/' + file_name
    image = cv2.imread(current_path)
    images.append(image)
    measurement = line[3]
    steering.append(float(measurement)) 

X_train = np.array(images)
y_train = np.array(steering)


# Define simple CNN model,Input shape shall be (160, 320, 3)
model = Sequential()
model.add(Flatten(input_shape=(160, 320, 3)))
model.add(Dense(1))

# Compile the model 
model.compile(loss='mse', optimizer='adam')

# Model Training 
model.fit(X_train, y_train, epochs=10, validation_split=0.2, shuffle=True, verbose=2)

# Saving model
model.save('model_simple.h5')