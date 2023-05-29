import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import math
from generators import *
from sklearn.metrics import accuracy_score

# Generate data
generator = Generator(train_percentage='100%', dimensions=2)
x_train, y_train, a, b = generator.generate(size=1000, min_range=-1000, max_range=1000, type='euklides')
x_test, y_test, c, d  = generator.generate(size=100, min_range=2000, max_range=4000, type='euklides')

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=512, activation='relu'))
model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dense(units=128, activation='relu'))
model.add(tf.keras.layers.Dense(units=1, activation='relu'))

model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')

# Train
history = model.fit(x_train, y_train, epochs=50)

losses = history.history['loss']
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

# Generate test data
test_x1, test_y1= (
    x_test[:, 0],
    x_test[:, 1],
)
predicted_distances = model.predict(x_test).flatten()

# Visualize the results
plt.scatter(test_x1, test_y1, c=predicted_distances)
plt.colorbar(label="Predicted Distance")
plt.xlabel("X1")
plt.ylabel("Y1")
plt.title("Euclidean Distance between Two Points")
plt.show()

max_diff = 0
for i, data in enumerate(x_test):
    real_euclidean = np.sqrt(
        (data[2:][0] - data[:2][0]) ** 2 + (data[2:][1] - data[:2][1]) ** 2
    )
    difference = np.round(np.abs(predicted_distances[i] - real_euclidean), 2)
    if difference > max_diff:
        max_diff = difference

    print(f"{data[:2]}, {data[2:]} -> {predicted_distances[i]} (Diff: ~{difference})")

print(f"Max difference: {max_diff}")