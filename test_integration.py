#!/usr/bin/env python3
"""
Comprehensive integration test for Balorg AI
Tests all components work together correctly
"""

import sys


def test_imports():
    """Test all imports work correctly"""
    print("Testing imports...")
    try:
        from balorg_ai import (
            BalorgAI,
            TechnicalMode,
            MixedPrecisionTrainer,
            DistributedTrainer,
            KnowledgeDistillation,
            ModelPruner,
            ModelQuantizer
        )
        print("  ✓ All imports successful")
        return True
    except Exception as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_model_creation():
    """Test model creation"""
    print("Testing model creation...")
    try:
        from balorg_ai import BalorgAI
        
        # Test transformer
        transformer = BalorgAI(model_type='transformer', config={
            'num_layers': 6,
            'hidden_size': 512
        })
        assert transformer.model_type == 'transformer'
        
        # Test neural network
        nn = BalorgAI(model_type='neural_network', config={
            'layer_sizes': [256, 128, 64]
        })
        assert nn.model_type == 'neural_network'
        
        print("  ✓ Model creation successful")
        return True
    except Exception as e:
        print(f"  ✗ Model creation failed: {e}")
        return False


def test_optimization_techniques():
    """Test optimization techniques"""
    print("Testing optimization techniques...")
    try:
        from balorg_ai import (
            MixedPrecisionTrainer,
            DistributedTrainer,
            KnowledgeDistillation,
            ModelPruner,
            ModelQuantizer,
            BalorgAI
        )
        
        model = BalorgAI(model_type='transformer')
        
        # Test mixed precision
        mp = MixedPrecisionTrainer(opt_level='O1')
        assert mp.opt_level == 'O1'
        
        # Test distributed training
        dist = DistributedTrainer(world_size=4)
        assert dist.world_size == 4
        
        # Test knowledge distillation
        kd = KnowledgeDistillation(temperature=3.0)
        assert kd.temperature == 3.0
        
        # Test pruning
        pruner = ModelPruner(sparsity=0.5)
        assert pruner.sparsity == 0.5
        
        # Test quantization
        quantizer = ModelQuantizer(dtype='int8')
        assert quantizer.dtype == 'int8'
        
        print("  ✓ All optimization techniques work")
        return True
    except Exception as e:
        print(f"  ✗ Optimization techniques failed: {e}")
        return False


def test_technical_mode():
    """Test technical mode functionality"""
    print("Testing technical mode...")
    try:
        from balorg_ai import TechnicalMode
        
        tech_mode = TechnicalMode()
        
        # Test welcome message
        welcome = tech_mode.get_welcome_message()
        assert "Activating technical mode" in welcome
        assert "Mixed Precision Training" in welcome
        assert "Distributed Training" in welcome
        assert "Knowledge Distillation" in welcome
        assert "Pruning" in welcome
        assert "Quantization" in welcome
        
        # Test menu processing
        response = tech_mode.process_query('1')
        assert "Architecture" in response
        
        response = tech_mode.process_query('2')
        assert "Optimization" in response
        
        response = tech_mode.process_query('3')
        assert "Application" in response
        
        # Test detail retrieval
        details = tech_mode.get_architecture_details('transformer')
        assert len(details) > 0
        
        details = tech_mode.get_optimization_details('mixed_precision')
        assert len(details) > 0
        
        details = tech_mode.get_application_details('nlp')
        assert len(details) > 0
        
        print("  ✓ Technical mode works correctly")
        return True
    except Exception as e:
        print(f"  ✗ Technical mode failed: {e}")
        return False


def test_problem_statement_requirements():
    """Verify all problem statement requirements are met"""
    print("Verifying problem statement requirements...")
    try:
        from balorg_ai import TechnicalMode
        
        tech_mode = TechnicalMode()
        welcome = tech_mode.get_welcome_message()
        
        # Check all required text from problem statement
        required_phrases = [
            "Activating technical mode",
            "transformer models and neural networks",
            "Mixed Precision Training",
            "float16 and float32 precision",
            "balance accuracy and computational efficiency",
            "Distributed Training",
            "multiple GPUs or nodes",
            "accelerate model development",
            "Knowledge Distillation",
            "larger models to smaller ones",
            "improved efficiency",
            "Pruning",
            "redundant weights and connections",
            "reduce model size",
            "Quantization",
            "lower-precision data types",
            "Model Architecture",
            "transformer models, neural networks",
            "Optimization Techniques",
            "model performance and efficiency",
            "Applications",
            "natural language processing, computer vision, or reinforcement learning"
        ]
        
        missing = []
        for phrase in required_phrases:
            if phrase not in welcome:
                missing.append(phrase)
        
        if missing:
            print(f"  ✗ Missing phrases: {missing}")
            return False
        
        print("  ✓ All problem statement requirements met")
        return True
    except Exception as e:
        print(f"  ✗ Problem statement verification failed: {e}")
        return False


def test_menu_options():
    """Test menu options match problem statement"""
    print("Testing menu options...")
    try:
        from balorg_ai import TechnicalMode
        
        # Verify menu has 3 options
        assert len(TechnicalMode.MENU_OPTIONS) == 3
        assert '1' in TechnicalMode.MENU_OPTIONS
        assert '2' in TechnicalMode.MENU_OPTIONS
        assert '3' in TechnicalMode.MENU_OPTIONS
        
        # Verify option descriptions
        assert "Architecture" in TechnicalMode.MENU_OPTIONS['1']
        assert "Optimization" in TechnicalMode.MENU_OPTIONS['2']
        assert "Applications" in TechnicalMode.MENU_OPTIONS['3']
        
        print("  ✓ Menu options correct")
        return True
    except Exception as e:
        print(f"  ✗ Menu options failed: {e}")
        return False


def main():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("Balorg AI - Comprehensive Integration Test")
    print("=" * 70 + "\n")
    
    tests = [
        test_imports,
        test_model_creation,
        test_optimization_techniques,
        test_technical_mode,
        test_problem_statement_requirements,
        test_menu_options
    ]
    
    results = []
    for test in tests:
        result = test()
        results.append(result)
        print()
    
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Test Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All integration tests passed!")
        print("=" * 70)
        return 0
    else:
        print(f"✗ {total - passed} test(s) failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
