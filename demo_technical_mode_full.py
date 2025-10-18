#!/usr/bin/env python3
"""
Non-interactive demo of Balorg AI Technical Mode
Shows all features without requiring user input
"""

from balorg_ai import TechnicalMode


def demonstrate_technical_mode():
    """Demonstrate technical mode capabilities"""
    
    print("=" * 70)
    print("BALORG AI - TECHNICAL MODE DEMONSTRATION")
    print("=" * 70)
    
    tech_mode = TechnicalMode()
    
    # Show welcome message
    print("\n" + "=" * 70)
    print("1. WELCOME MESSAGE")
    print("=" * 70)
    print(tech_mode.get_welcome_message())
    
    # Demonstrate main menu navigation
    print("\n" + "=" * 70)
    print("2. EXPLORING MENU OPTION 1: MODEL ARCHITECTURE")
    print("=" * 70)
    response = tech_mode.process_query('1')
    print(response)
    
    # Show transformer details
    print("\n" + "=" * 70)
    print("3. TRANSFORMER MODEL DETAILS")
    print("=" * 70)
    print(tech_mode.get_architecture_details('transformer'))
    
    # Show neural network details
    print("\n" + "=" * 70)
    print("4. NEURAL NETWORK DETAILS")
    print("=" * 70)
    print(tech_mode.get_architecture_details('neural_network'))
    
    # Demonstrate optimization menu
    print("\n" + "=" * 70)
    print("5. EXPLORING MENU OPTION 2: OPTIMIZATION TECHNIQUES")
    print("=" * 70)
    response = tech_mode.process_query('2')
    print(response)
    
    # Show mixed precision details
    print("\n" + "=" * 70)
    print("6. MIXED PRECISION TRAINING DETAILS")
    print("=" * 70)
    print(tech_mode.get_optimization_details('mixed_precision'))
    
    # Show distributed training details
    print("\n" + "=" * 70)
    print("7. DISTRIBUTED TRAINING DETAILS")
    print("=" * 70)
    print(tech_mode.get_optimization_details('distributed'))
    
    # Show knowledge distillation details
    print("\n" + "=" * 70)
    print("8. KNOWLEDGE DISTILLATION DETAILS")
    print("=" * 70)
    print(tech_mode.get_optimization_details('distillation'))
    
    # Show pruning details
    print("\n" + "=" * 70)
    print("9. MODEL PRUNING DETAILS")
    print("=" * 70)
    print(tech_mode.get_optimization_details('pruning'))
    
    # Show quantization details
    print("\n" + "=" * 70)
    print("10. MODEL QUANTIZATION DETAILS")
    print("=" * 70)
    print(tech_mode.get_optimization_details('quantization'))
    
    # Demonstrate applications menu
    print("\n" + "=" * 70)
    print("11. EXPLORING MENU OPTION 3: APPLICATIONS")
    print("=" * 70)
    response = tech_mode.process_query('3')
    print(response)
    
    # Show NLP details
    print("\n" + "=" * 70)
    print("12. NATURAL LANGUAGE PROCESSING DETAILS")
    print("=" * 70)
    print(tech_mode.get_application_details('nlp'))
    
    # Show computer vision details
    print("\n" + "=" * 70)
    print("13. COMPUTER VISION DETAILS")
    print("=" * 70)
    print(tech_mode.get_application_details('cv'))
    
    # Show reinforcement learning details
    print("\n" + "=" * 70)
    print("14. REINFORCEMENT LEARNING DETAILS")
    print("=" * 70)
    print(tech_mode.get_application_details('rl'))
    
    print("\n" + "=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print("\nTo run interactive mode, use: python examples/technical_mode_demo.py")
    print("=" * 70)


def main():
    """Run the demonstration"""
    demonstrate_technical_mode()


if __name__ == "__main__":
    main()
