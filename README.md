# Day-10-Automatrix
Day 10 ESP32 project: train a tiny TensorFlow model in Google Colab to learn y = 2x, then run its weight-and-bias inference on an ESP32 in Wokwi.

# Day 10: First TensorFlow Model

This project trains a one-neuron TensorFlow model in Google Colab to learn the relationship `y = 2x`. It plots the model’s predictions against the actual values, then exports the trained weight and bias for use in an ESP32 simulation.

The ESP32 applies the trained linear model to sample inputs and prints the actual and predicted values to the Serial Monitor.

## Project files

- `python.py` — trains and evaluates the TensorFlow model and prints its weight and bias.
- `sketch.ino` — uses the exported parameters to run inference on the ESP32.
- `diagram.json` — configures the ESP32 board and Serial Monitor in Wokwi.

## Run the Colab model

1. Open `python.py` in Google Colab, or paste its contents into a notebook cell.
2. Run the cell.
3. View the prediction plot and the printed model parameters.

## Run the Wokwi simulation

1. Open the Wokwi project.
2. Paste `sketch.ino` and `diagram.json` into their matching tabs.
3. Set `model_weight` and `model_bias` in `sketch.ino` to the values printed by Colab.
4. Start the simulation and view the results in the Serial Monitor.
## Wokwi Simulation

[Run the simulation](https://wokwi.com/projects/476694878324111361)

## Example trained parameters

```cpp
const float model_weight = 2.000000f;
const float model_bias = -0.000000f;
