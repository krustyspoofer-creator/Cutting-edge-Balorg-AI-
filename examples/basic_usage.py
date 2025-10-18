"""
Example usage of Balorg AI Framework.

This example demonstrates how to use the Balorg AI Framework to:
1. Create a neural network
2. Load and preprocess data
3. Train a model
4. Evaluate the model
"""

import numpy as np
from balorg.framework import Model, Config, Trainer
from balorg.nn import NeuralNetwork, DenseLayer
from balorg.preprocessing import DataLoader, normalize, one_hot_encode


def main():
    print("=" * 70)
    print("Balorg AI Framework - Example Usage")
    print("=" * 70)
    
    # Create a configuration
    print("\n1. Creating configuration...")
    config = Config(
        learning_rate=0.001,
        batch_size=32,
        epochs=10
    )
    print(f"Configuration: {config.to_dict()}")
    
    # Load and preprocess data
    print("\n2. Loading and preprocessing data...")
    data_loader = DataLoader(batch_size=32, shuffle=True, seed=42)
    (x_train, y_train), (x_test, y_test) = data_loader.load_dataset('mnist')
    
    # Normalize the data
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    # One-hot encode labels
    y_train_encoded = one_hot_encode(y_train, num_classes=10)
    y_test_encoded = one_hot_encode(y_test, num_classes=10)
    
    print(f"Training data shape: {x_train.shape}")
    print(f"Test data shape: {x_test.shape}")
    
    # Create a neural network
    print("\n3. Creating neural network...")
    model = NeuralNetwork(name="MNIST_Classifier")
    model.add_layer(DenseLayer(input_size=784, output_size=128, activation='relu'))
    model.add_layer(DenseLayer(input_size=128, output_size=64, activation='relu'))
    model.add_layer(DenseLayer(input_size=64, output_size=10, activation='softmax'))
    
    print("\nModel Architecture:")
    model.summary()
    
    # Compile the model
    print("\n4. Compiling the model...")
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    print("Model compiled successfully!")
    
    # Train the model
    print("\n5. Training the model...")
    history = model.fit(
        x_train, y_train_encoded,
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size'),
        validation_data=(x_test, y_test_encoded),
        verbose=1
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    
    # Evaluate the model
    print("\n6. Evaluating the model...")
    results = model.evaluate(x_test, y_test_encoded, verbose=1)
    
    # Make predictions
    print("\n7. Making predictions...")
    sample_predictions = model.predict(x_test[:5])
    print(f"Sample predictions shape: {sample_predictions.shape}")
    
    print("\n" + "=" * 70)
    print("Example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
