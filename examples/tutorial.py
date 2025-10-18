"""
Tutorial: Building Your First Neural Network with Balorg AI

This tutorial walks through the complete process of building, training,
and evaluating a neural network using the Balorg AI framework.
"""

import numpy as np
from balorg.framework import Config, Trainer
from balorg.nn import NeuralNetwork, DenseLayer
from balorg.preprocessing import DataLoader, normalize, one_hot_encode, train_test_split


def section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def tutorial():
    """Run the complete tutorial."""
    
    print("\n" + "=" * 70)
    print("  BALORG AI FRAMEWORK - COMPLETE TUTORIAL")
    print("=" * 70)
    print("\nThis tutorial will guide you through:")
    print("  1. Loading and preprocessing data")
    print("  2. Building a neural network architecture")
    print("  3. Configuring training parameters")
    print("  4. Training the model")
    print("  5. Evaluating model performance")
    print("  6. Making predictions")
    
    # ========================================================================
    section("STEP 1: Load and Inspect Data")
    # ========================================================================
    
    print("Loading MNIST dataset...")
    data_loader = DataLoader(batch_size=32, shuffle=True, seed=42)
    (x_train, y_train), (x_test, y_test) = data_loader.load_dataset('mnist')
    
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Input shape: {x_train[0].shape}")
    print(f"Number of classes: {len(np.unique(y_train))}")
    
    # ========================================================================
    section("STEP 2: Preprocess Data")
    # ========================================================================
    
    print("Normalizing pixel values to [0, 1]...")
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    
    print("Converting labels to one-hot encoding...")
    num_classes = 10
    y_train_encoded = one_hot_encode(y_train, num_classes=num_classes)
    y_test_encoded = one_hot_encode(y_test, num_classes=num_classes)
    
    print(f"Encoded label shape: {y_train_encoded[0].shape}")
    print(f"Example label: {y_train[0]} -> {y_train_encoded[0]}")
    
    print("\nSplitting training data into train/validation sets...")
    x_train, x_val, y_train_encoded, y_val_encoded = train_test_split(
        x_train, y_train_encoded,
        test_size=0.15,
        shuffle=True,
        random_state=42
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    
    # ========================================================================
    section("STEP 3: Build Neural Network Architecture")
    # ========================================================================
    
    print("Creating a 3-layer feedforward neural network...")
    print("Architecture:")
    print("  Input Layer:  784 neurons (28x28 pixels)")
    print("  Hidden Layer 1: 128 neurons (ReLU activation)")
    print("  Hidden Layer 2: 64 neurons (ReLU activation)")
    print("  Output Layer: 10 neurons (Softmax activation)")
    
    model = NeuralNetwork(name="MNIST_Classifier")
    model.add_layer(DenseLayer(input_size=784, output_size=128, activation='relu'))
    model.add_layer(DenseLayer(input_size=128, output_size=64, activation='relu'))
    model.add_layer(DenseLayer(input_size=64, output_size=10, activation='softmax'))
    
    print("\nModel Summary:")
    model.summary()
    
    # ========================================================================
    section("STEP 4: Configure Training Parameters")
    # ========================================================================
    
    print("Creating training configuration...")
    config = Config(
        learning_rate=0.001,
        batch_size=32,
        epochs=10,
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("Configuration:")
    for key, value in config.to_dict().items():
        print(f"  {key}: {value}")
    
    print("\nCompiling model with configuration...")
    model.compile(
        optimizer=config.get('optimizer'),
        loss=config.get('loss'),
        metrics=config.get('metrics')
    )
    print("Model compiled successfully!")
    
    # ========================================================================
    section("STEP 5: Train the Model")
    # ========================================================================
    
    print("Starting training process...")
    print(f"Training for {config.get('epochs')} epochs with batch size {config.get('batch_size')}")
    
    history = model.fit(
        x_train, y_train_encoded,
        epochs=config.get('epochs'),
        batch_size=config.get('batch_size'),
        validation_data=(x_val, y_val_encoded),
        verbose=1
    )
    
    print("\nTraining completed!")
    print(f"Final training loss: {history['loss'][-1]:.4f}")
    print(f"Final training accuracy: {history['accuracy'][-1]:.4f}")
    if 'val_loss' in history:
        print(f"Final validation loss: {history['val_loss'][-1]:.4f}")
        print(f"Final validation accuracy: {history['val_accuracy'][-1]:.4f}")
    
    # ========================================================================
    section("STEP 6: Evaluate Model Performance")
    # ========================================================================
    
    print("Evaluating model on test set...")
    results = model.evaluate(x_test, y_test_encoded, verbose=1)
    
    print("\nTest Results:")
    for metric, value in results.items():
        print(f"  {metric}: {value:.4f}")
    
    # ========================================================================
    section("STEP 7: Make Predictions")
    # ========================================================================
    
    print("Making predictions on sample data...")
    n_samples = 5
    sample_x = x_test[:n_samples]
    sample_y = y_test[:n_samples]
    
    predictions = model.predict(sample_x)
    predicted_classes = np.argmax(predictions, axis=1)
    
    print(f"\nPredictions for {n_samples} samples:")
    print("-" * 50)
    for i in range(n_samples):
        print(f"Sample {i+1}:")
        print(f"  True label: {sample_y[i]}")
        print(f"  Predicted label: {predicted_classes[i]}")
        print(f"  Confidence: {predictions[i][predicted_classes[i]]:.2%}")
        print()
    
    # ========================================================================
    section("TUTORIAL COMPLETE!")
    # ========================================================================
    
    print("Congratulations! You've completed the Balorg AI tutorial.")
    print("\nYou've learned how to:")
    print("  ✓ Load and preprocess data")
    print("  ✓ Build a neural network architecture")
    print("  ✓ Configure and compile a model")
    print("  ✓ Train a model with validation")
    print("  ✓ Evaluate model performance")
    print("  ✓ Make predictions on new data")
    
    print("\nNext steps:")
    print("  - Try adjusting the network architecture")
    print("  - Experiment with different hyperparameters")
    print("  - Add data augmentation for better generalization")
    print("  - Implement custom models using the Model base class")
    print("  - Explore the advanced_usage.py example")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    tutorial()
