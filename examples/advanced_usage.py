"""
Advanced example demonstrating data augmentation and custom training.

This example shows:
1. Custom model implementation
2. Data augmentation
3. Custom training loop with callbacks
"""

import numpy as np
from balorg.framework import Model, Trainer
from balorg.nn import NeuralNetwork, DenseLayer
from balorg.preprocessing import DataLoader, train_test_split, Scaler
from balorg.preprocessing.augmentation import DataAugmentor, random_noise


class CustomClassifier(NeuralNetwork):
    """Custom classifier with additional functionality."""
    
    def __init__(self, input_size, num_classes, hidden_sizes=[128, 64]):
        super().__init__(name="CustomClassifier")
        
        # Build architecture
        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.add_layer(DenseLayer(prev_size, hidden_size, activation='relu'))
            prev_size = hidden_size
        
        # Output layer
        self.add_layer(DenseLayer(prev_size, num_classes, activation='softmax'))


def main():
    print("=" * 70)
    print("Balorg AI Framework - Advanced Example")
    print("=" * 70)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    np.random.seed(42)
    x = np.random.randn(1000, 20)
    y = np.random.randint(0, 5, size=(1000,))
    
    # Split data
    print("\n2. Splitting data into train/validation/test sets...")
    x_temp, x_test, y_temp, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42
    )
    x_train, x_val, y_train, y_val = train_test_split(
        x_temp, y_temp, test_size=0.2, random_state=42
    )
    
    print(f"Training samples: {len(x_train)}")
    print(f"Validation samples: {len(x_val)}")
    print(f"Test samples: {len(x_test)}")
    
    # Standardize data
    print("\n3. Standardizing data...")
    scaler = Scaler(method='standard')
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_test = scaler.transform(x_test)
    
    # Set up data augmentation
    print("\n4. Setting up data augmentation...")
    augmentor = DataAugmentor(seed=42)
    augmentor.add_augmentation(lambda x: random_noise(x, noise_factor=0.01))
    
    # Augment training data
    x_train_aug, y_train = augmentor.augment_batch(x_train, y_train)
    print(f"Augmented training data shape: {x_train_aug.shape}")
    
    # Create custom model
    print("\n5. Creating custom model...")
    model = CustomClassifier(
        input_size=20,
        num_classes=5,
        hidden_sizes=[64, 32, 16]
    )
    
    print("\nModel Architecture:")
    model.summary()
    
    # Create trainer
    print("\n6. Setting up trainer...")
    trainer = Trainer(model, optimizer='adam', loss='categorical_crossentropy')
    
    # Train model
    print("\n7. Training model...")
    history = trainer.train(
        x_train_aug, y_train,
        epochs=5,
        batch_size=32,
        validation_data=(x_val, y_val),
        verbose=1
    )
    
    # Evaluate
    print("\n8. Evaluating model...")
    results = trainer.evaluate(x_test, y_test, verbose=1)
    
    print("\n" + "=" * 70)
    print("Advanced example completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()
