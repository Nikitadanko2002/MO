import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from subprocess import check_output


class myCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > 0.999):
            print('Good!')
            self.model.stop_training = True


train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
train_y = train['label'].astype('float32')
train_x = train.drop(['label'], axis=1).astype('int32')
test_x = test.astype('float32')
train_x = train_x.values.reshape(-1, 28, 28, 1)
train_x = train_x / 255.0
test_x = test_x.values.reshape(-1, 28, 28, 1)
test_x = test_x / 255.0
train_y = tf.keras.utils.to_categorical(train_y, 10)
print(train['label'].head())
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='Same'),
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.50),
    tf.keras.layers.Dense(10, activation='softmax')
])
callbacks = myCallBack()
Optimizer = tf.keras.optimizers.Adam(
    learning_rate=0.0005,
    beta_1=0.9,
    beta_2=0.999,
    epsilon=1e-07,
    name='Adam'
)
model.compile(optimizer=Optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(train_x, train_y, batch_size=50, epochs=3, callbacks=[callbacks])

results = model.predict(test_x)

results = np.argmax(results, axis=1)

results = pd.Series(results, name='Label')
submission = pd.concat([pd.Series(range(1, 28001), name="ImageId"), results], axis=1)
submission.to_csv("submission.csv", index=False)
