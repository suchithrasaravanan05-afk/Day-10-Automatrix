import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Make training repeatable
tf.keras.utils.set_random_seed(42)

# Generate training data for y = 2x
x_train = np.linspace(-10, 10, 200, dtype=np.float32)
y_train = 2 * x_train

# Build a one-neuron model
model = tf.keras.Sequential([
    tf.keras.Input(shape=(1,)),
    tf.keras.layers.Dense(units=1)
])

# Compile and train the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="mean_squared_error"
)
model.fit(x_train, y_train, epochs=200, verbose=0)

# Generate test data and predictions
x_test = np.linspace(-12, 12, 50, dtype=np.float32)
y_test = 2 * x_test
y_pred = model.predict(x_test, verbose=0).flatten()

# Plot predictions versus actual values
plt.figure(figsize=(8, 5))
plt.scatter(x_test, y_test, color="blue", label="Actual (y = 2x)", alpha=0.6)
plt.plot(x_test, y_pred, "r--", linewidth=2, label="TensorFlow predictions")
plt.title("TensorFlow Predictions vs Actual Values")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.grid(True)
plt.show()

# Print trained parameters for the ESP32
weight = float(model.layers[0].get_weights()[0][0][0])
bias = float(model.layers[0].get_weights()[1][0])

print("\n--- Model Insights ---")
print(f"Trained weight (target about 2): {weight:.6f}")
print(f"Trained bias (target about 0):  {bias:.6f}")

print("\n--- Copy these values into sketch.ino ---")
print(f"const float model_weight = {weight:.6f}f;")
print(f"const float model_bias   = {bias:.6f}f;")
