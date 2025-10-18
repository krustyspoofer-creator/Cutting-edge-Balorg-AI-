"""Tests for Neural Network Library."""

import pytest
import numpy as np
from balorg.nn import NeuralNetwork, DenseLayer, relu, sigmoid, tanh, softmax
from balorg.nn.layers import DropoutLayer, BatchNormLayer


def test_neural_network_initialization():
    """Test neural network initialization."""
    nn = NeuralNetwork(name="test_nn")
    assert nn.name == "test_nn"
    assert len(nn.layers) == 0


def test_add_layer():
    """Test adding layers to network."""
    nn = NeuralNetwork()
    layer = DenseLayer(input_size=10, output_size=5)
    nn.add_layer(layer)
    
    assert len(nn.layers) == 1
    assert nn.layers[0] == layer


def test_forward_pass():
    """Test forward pass through network."""
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(input_size=10, output_size=5, activation='relu'))
    
    x = np.random.randn(32, 10)
    output = nn.forward(x)
    
    assert output.shape == (32, 5)


def test_dense_layer_initialization():
    """Test dense layer initialization."""
    layer = DenseLayer(input_size=10, output_size=5, activation='relu')
    
    assert layer.input_size == 10
    assert layer.output_size == 5
    assert layer.weights.shape == (10, 5)


def test_dense_layer_forward():
    """Test dense layer forward pass."""
    layer = DenseLayer(input_size=10, output_size=5)
    x = np.random.randn(32, 10)
    output = layer.forward(x)
    
    assert output.shape == (32, 5)


def test_dense_layer_backward():
    """Test dense layer backward pass."""
    layer = DenseLayer(input_size=10, output_size=5)
    x = np.random.randn(32, 10)
    output = layer.forward(x)
    
    gradient = np.random.randn(32, 5)
    input_gradient = layer.backward(gradient)
    
    assert input_gradient.shape == (32, 10)
    assert layer.weights_gradient.shape == (10, 5)


def test_dropout_layer():
    """Test dropout layer."""
    layer = DropoutLayer(dropout_rate=0.5)
    x = np.random.randn(32, 10)
    
    layer.training = True
    output = layer.forward(x)
    assert output.shape == x.shape
    
    layer.training = False
    output = layer.forward(x)
    assert np.allclose(output, x)


def test_batch_norm_layer():
    """Test batch normalization layer."""
    layer = BatchNormLayer(num_features=10)
    x = np.random.randn(32, 10)
    
    output = layer.forward(x)
    assert output.shape == x.shape


def test_relu_activation():
    """Test ReLU activation."""
    x = np.array([-2, -1, 0, 1, 2])
    output = relu(x)
    expected = np.array([0, 0, 0, 1, 2])
    assert np.allclose(output, expected)


def test_sigmoid_activation():
    """Test sigmoid activation."""
    x = np.array([0])
    output = sigmoid(x)
    assert np.allclose(output, 0.5)


def test_tanh_activation():
    """Test tanh activation."""
    x = np.array([0])
    output = tanh(x)
    assert np.allclose(output, 0)


def test_softmax_activation():
    """Test softmax activation."""
    x = np.array([[1, 2, 3]])
    output = softmax(x)
    
    assert output.shape == x.shape
    assert np.allclose(np.sum(output), 1.0)


def test_network_summary():
    """Test network summary."""
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(input_size=10, output_size=5, activation='relu'))
    nn.add_layer(DenseLayer(input_size=5, output_size=2, activation='softmax'))
    
    # Should not raise an error
    nn.summary()


def test_get_set_weights():
    """Test getting and setting weights."""
    nn = NeuralNetwork()
    nn.add_layer(DenseLayer(input_size=10, output_size=5))
    
    weights = nn.get_weights()
    assert len(weights) == 1
    
    new_weights = [np.random.randn(10, 5)]
    nn.set_weights(new_weights)
    
    assert np.allclose(nn.layers[0].weights, new_weights[0])
