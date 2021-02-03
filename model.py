import csv, cv2, math
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Conv2D, Dropout, Cropping2D, BatchNormalization, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

samples = []

# reading the csv file

with open('/opt/carnd_p3/data/driving_log.csv') as f:
    reader = csv.reader(f)
    next(reader)
    for line in reader:
        samples.append(line)
    
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

# defining generator for getting features and labels

def generator(samples, batch_size=32):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset+batch_size]
            images = []
            measurements = []
            for batch_sample in batch_samples:
                curr_sample = '/opt/carnd_p3/data/IMG/'+batch_sample[0].split('/')[-1]
                img = cv2.imread(curr_sample)
                image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                measurement = float(batch_sample[3])
                images.append(image)
                measurements.append(measurement)
                images.append(cv2.flip(image, 1))
                measurements.append(measurement*-1.0)
            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

batch_size = 64
train_generator = generator(train_samples, batch_size=batch_size)
validation_generator = generator(validation_samples, batch_size=batch_size)

ch, row, col = 3, 160, 320

# creating, compiling and fitting model

model = Sequential()
model.add(Lambda(lambda x: (x / 255.0) - 0.5, input_shape=(row, col, ch)))
model.add(Cropping2D(cropping=((70, 20), (0, 0))))
model.add(Conv2D(24, kernel_size=(5, 5), strides=(2, 2), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(36, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(48, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
#model.add(BatchNormalization())
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, steps_per_epoch=math.ceil(len(train_samples) / batch_size), \
                    validation_data=validation_generator, validation_steps=math.ceil(len(validation_samples) / batch_size),\
                    epochs=5, verbose=1)

model.save('model.h5')
