#!/usr/bin/env python3
"""
Test script to verify technical mode matches the problem statement
"""

from balorg_ai import TechnicalMode


def test_welcome_message():
    """Test that welcome message matches problem statement"""
    tech_mode = TechnicalMode()
    message = tech_mode.get_welcome_message()
    
    # Check for key phrases from problem statement
    assert "Activating technical mode" in message
    assert "transformer models and neural networks" in message
    assert "Mixed Precision Training" in message
    assert "float16 and float32" in message
    assert "Distributed Training" in message
    assert "Knowledge Distillation" in message
    assert "Pruning" in message
    assert "Quantization" in message
    
    print("✓ Welcome message matches problem statement")


def test_menu_options():
    """Test that menu options are correct"""
    tech_mode = TechnicalMode()
    
    # Test main menu
    assert '1' in TechnicalMode.MENU_OPTIONS
    assert '2' in TechnicalMode.MENU_OPTIONS
    assert '3' in TechnicalMode.MENU_OPTIONS
    
    assert "Model Architecture" in TechnicalMode.MENU_OPTIONS['1']
    assert "Optimization Techniques" in TechnicalMode.MENU_OPTIONS['2']
    assert "Applications" in TechnicalMode.MENU_OPTIONS['3']
    
    print("✓ Menu options are correct")


def test_query_processing():
    """Test query processing"""
    tech_mode = TechnicalMode()
    
    # Test main menu option 1
    response = tech_mode.process_query('1')
    assert "Model Architecture" in response
    assert "Transformer" in response or "transformer" in response
    
    # Test main menu option 2
    response = tech_mode.process_query('2')
    assert "Optimization" in response
    assert "Mixed Precision" in response or "mixed_precision" in response
    
    # Test main menu option 3
    response = tech_mode.process_query('3')
    assert "Application" in response
    assert "NLP" in response or "nlp" in response
    
    print("✓ Query processing works correctly")


def test_architecture_details():
    """Test architecture details"""
    tech_mode = TechnicalMode()
    
    # Test transformer details
    response = tech_mode.get_architecture_details('transformer')
    assert "Multi-Head" in response or "attention" in response.lower()
    assert "Transformer" in response
    
    # Test neural network details
    response = tech_mode.get_architecture_details('neural_network')
    assert "Neural Network" in response or "Fully Connected" in response
    
    print("✓ Architecture details are comprehensive")


def test_optimization_details():
    """Test optimization details"""
    tech_mode = TechnicalMode()
    
    # Test mixed precision
    response = tech_mode.get_optimization_details('mixed_precision')
    assert "Mixed Precision" in response
    assert "float16" in response or "FP16" in response
    
    # Test distributed training
    response = tech_mode.get_optimization_details('distributed')
    assert "Distributed" in response
    assert "GPU" in response or "parallel" in response.lower()
    
    # Test knowledge distillation
    response = tech_mode.get_optimization_details('distillation')
    assert "Distillation" in response or "distillation" in response.lower()
    assert "teacher" in response.lower()
    
    # Test pruning
    response = tech_mode.get_optimization_details('pruning')
    assert "Pruning" in response or "pruning" in response.lower()
    assert "weight" in response.lower()
    
    # Test quantization
    response = tech_mode.get_optimization_details('quantization')
    assert "Quantization" in response or "quantization" in response.lower()
    assert "INT8" in response or "precision" in response.lower()
    
    print("✓ Optimization details are comprehensive")


def test_application_details():
    """Test application details"""
    tech_mode = TechnicalMode()
    
    # Test NLP
    response = tech_mode.get_application_details('nlp')
    assert "Natural Language Processing" in response or "NLP" in response
    
    # Test computer vision
    response = tech_mode.get_application_details('cv')
    assert "Computer Vision" in response or "vision" in response.lower()
    
    # Test reinforcement learning
    response = tech_mode.get_application_details('rl')
    assert "Reinforcement Learning" in response or "RL" in response
    
    print("✓ Application details are comprehensive")


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("Testing Balorg AI Technical Mode")
    print("=" * 60 + "\n")
    
    test_welcome_message()
    test_menu_options()
    test_query_processing()
    test_architecture_details()
    test_optimization_details()
    test_application_details()
    
    print("\n" + "=" * 60)
    print("All tests passed! ✓")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
