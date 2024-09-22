import numpy as np
import tensorflow as tf
print('tensorflow version: ' + tf.__version__)

from keras.models import Sequential
from keras.layers import Dense

def m1(epochs, input_neurons=10, output_neurons=1, enableIntNewron = False):
    a = np.random.randint(0, 100, size=(1000, 1))  # 1000 random integers between 0 and 100
    # Initialize b based on the condition
    b = np.where(a % 2 == 0, 2 * a + 1, 2 * a + 5)

    # Normalize the input data
    a = a / 100
    b = b / 100

    # Create the model
    model = Sequential()
    model.add(Dense(input_neurons, input_dim=1, activation='relu'))  # Specify the number of input neurons

    if(enableIntNewron):
        model.add(Dense(input_neurons, activation='relu'))

    # Output layer with configurable neurons
    model.add(Dense(output_neurons))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(a, b, epochs=epochs, batch_size=10, verbose=0)

    # Test the model
    test_a = np.array([[50], [51], [52], [53], [54], [55], [56], [57], [58], [59]])  # 10 example test inputs

    # Calculate actual_b based on the condition provided
    denormalized_a = test_a.flatten()  # Flatten to get a 1D array of a values
    actual_b = np.where(denormalized_a % 2 == 0, 2 * denormalized_a + 1, 2 * denormalized_a + 5)

    test_a = test_a / 100  # Normalize the input

    # Predict using the model
    predicted_b = model.predict(test_a)

    # Denormalize the predicted output to get the original scale
    predicted_b = predicted_b * 100

    total_accuracy = 0
    print(f"\nEpochs: {epochs}, Input Neurons: {input_neurons}, Output Neurons: {output_neurons}, enableIntNeuron: {enableIntNewron}")
    for i in range(len(predicted_b)):
        accuracy = 100 * (1 - abs(predicted_b[i][0] - actual_b[i]) / actual_b[i])  # Calculate accuracy percentage
        total_accuracy += accuracy  # Accumulate accuracy
        print(f"a = {denormalized_a[i]:.0f}, Predicted b: {predicted_b[i][0]:.2f}, Actual b: {actual_b[i]:.2f}, Accuracy: {accuracy:.2f}%")

    average_accuracy = total_accuracy / len(predicted_b)
    print(f"Average Accuracy: {average_accuracy:.2f}%")
    return average_accuracy

# Call the function with varying epoch values and different configurations
epoch_values_range =  range(10, 911, 150) # #[20] # Start from 10 to 1000 in increments of 50
accuracies = []

# Test with different configurations
for epoch_values in epoch_values_range:  # Example input neuron counts
    for input_neurons in [10, 50]:  # Example input neuron counts
        for output_neurons in [1]:  # Example output neuron counts
            for enableIntNewrons in [False, True]:  # Example output neuron counts
                avg_accuracy = m1(epoch_values, input_neurons=input_neurons, output_neurons=output_neurons, enableIntNewron=enableIntNewrons)
                accuracies.append((input_neurons, output_neurons, enableIntNewrons, avg_accuracy))

# Print the average accuracies for each configuration
print("\nInput Neurons vs Output Neurons vs enableIntNeuron Average Accuracy:")
for input_neurons, output_neurons, enableIntNewrons, accuracy in accuracies:
    print(f"Input Neurons: {input_neurons}, Output Neurons: {output_neurons}, enableIntNewrons: {enableIntNewrons}, Average Accuracy: {accuracy:.2f}%")
