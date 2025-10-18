#!/usr/bin/env python3
"""
Example script demonstrating Balorg AI optimization techniques
"""

from balorg_ai import (
    BalorgAI,
    MixedPrecisionTrainer,
    DistributedTrainer,
    KnowledgeDistillation,
    ModelPruner,
    ModelQuantizer
)


def demonstrate_mixed_precision():
    """Demonstrate Mixed Precision Training"""
    print("\n" + "=" * 60)
    print("Mixed Precision Training Example")
    print("=" * 60)
    
    # Create a model
    model = BalorgAI(model_type='transformer', config={
        'num_layers': 12,
        'hidden_size': 768
    })
    
    # Initialize mixed precision trainer
    mp_trainer = MixedPrecisionTrainer(
        enabled=True,
        loss_scale=1024.0,
        opt_level='O1'
    )
    
    print("\nModel Configuration:")
    print(model.summary())
    
    print("\nMixed Precision Configuration:")
    for key, value in mp_trainer.get_config().items():
        print(f"  {key}: {value}")
    
    print("\nBenefits:")
    print("  - 2-3x faster training")
    print("  - 50% reduced memory consumption")
    print("  - Larger batch sizes possible")


def demonstrate_distributed_training():
    """Demonstrate Distributed Training"""
    print("\n" + "=" * 60)
    print("Distributed Training Example")
    print("=" * 60)
    
    # Initialize distributed trainer
    dist_trainer = DistributedTrainer(
        backend='nccl',
        world_size=4,
        rank=0
    )
    
    print("\nDistributed Configuration:")
    for key, value in dist_trainer.get_config().items():
        print(f"  {key}: {value}")
    
    print("\nBenefits:")
    print("  - Linear speedup with multiple GPUs")
    print("  - Train larger models (>100B parameters)")
    print("  - Faster experimentation")


def demonstrate_knowledge_distillation():
    """Demonstrate Knowledge Distillation"""
    print("\n" + "=" * 60)
    print("Knowledge Distillation Example")
    print("=" * 60)
    
    # Create teacher model
    teacher = BalorgAI(model_type='transformer', config={
        'num_layers': 12,
        'hidden_size': 768
    })
    
    # Create student model (smaller)
    student = BalorgAI(model_type='transformer', config={
        'num_layers': 6,
        'hidden_size': 384
    })
    
    # Initialize knowledge distillation
    kd = KnowledgeDistillation(
        temperature=3.0,
        alpha=0.5,
        teacher_model=teacher
    )
    
    print("\nTeacher Model:")
    print(teacher.summary())
    
    print("\nStudent Model:")
    print(student.summary())
    
    print("\nKnowledge Distillation Configuration:")
    for key, value in kd.get_config().items():
        print(f"  {key}: {value}")
    
    print("\nBenefits:")
    print("  - 10x model size reduction")
    print("  - 2-5x faster inference")
    print("  - 90-95% accuracy retention")


def demonstrate_pruning():
    """Demonstrate Model Pruning"""
    print("\n" + "=" * 60)
    print("Model Pruning Example")
    print("=" * 60)
    
    # Create model
    model = BalorgAI(model_type='neural_network', config={
        'layer_sizes': [512, 256, 128, 64],
        'activation': 'relu'
    })
    
    # Initialize pruner
    pruner = ModelPruner(
        pruning_method='magnitude',
        sparsity=0.5,
        structured=False
    )
    
    print("\nOriginal Model:")
    print(model.summary())
    
    print("\nPruning Configuration:")
    for key, value in pruner.get_config().items():
        print(f"  {key}: {value}")
    
    print("\nBenefits:")
    print("  - 50-90% size reduction")
    print("  - 2-5x faster inference")
    print("  - Lower memory usage")


def demonstrate_quantization():
    """Demonstrate Model Quantization"""
    print("\n" + "=" * 60)
    print("Model Quantization Example")
    print("=" * 60)
    
    # Create model
    model = BalorgAI(model_type='transformer', config={
        'num_layers': 6,
        'hidden_size': 512
    })
    
    # Initialize quantizer
    quantizer = ModelQuantizer(
        quantization_type='dynamic',
        dtype='int8',
        calibration_method='minmax'
    )
    
    print("\nOriginal Model:")
    print(model.summary())
    
    print("\nQuantization Configuration:")
    for key, value in quantizer.get_config().items():
        print(f"  {key}: {value}")
    
    print("\nBenefits:")
    print("  - 4x model size reduction (INT8)")
    print("  - 2-4x inference speedup")
    print("  - Edge device deployment")


def main():
    """Run all optimization demonstrations"""
    print("\n" + "=" * 60)
    print("Balorg AI - Optimization Techniques Demonstration")
    print("=" * 60)
    
    demonstrate_mixed_precision()
    demonstrate_distributed_training()
    demonstrate_knowledge_distillation()
    demonstrate_pruning()
    demonstrate_quantization()
    
    print("\n" + "=" * 60)
    print("Demonstration Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
