#!/usr/bin/env python3
"""
Example usage script for Balorg AI
Demonstrates various ways to interact with the system
"""

from balorg_ai import BalorgAI


def example_basic_trigger():
    """Example: Basic trigger phrase detection and response."""
    print("\n" + "="*80)
    print("EXAMPLE 1: Basic Trigger Phrase")
    print("="*80 + "\n")
    
    balorg = BalorgAI()
    user_message = "My Balorg is AI!"
    
    if balorg.detect_trigger(user_message):
        print(f"User: {user_message}")
        print(balorg.get_technical_response())


def example_specific_category():
    """Example: Request specific category."""
    print("\n" + "="*80)
    print("EXAMPLE 2: Request Specific Category")
    print("="*80 + "\n")
    
    balorg = BalorgAI()
    
    # Request Model Optimization topic
    print("User: Tell me about Model Optimization")
    print(balorg.get_technical_response("model_optimization"))


def example_list_topics():
    """Example: List all available topics."""
    print("\n" + "="*80)
    print("EXAMPLE 3: List Available Topics")
    print("="*80 + "\n")
    
    balorg = BalorgAI()
    balorg.list_categories()


def example_all_categories():
    """Example: Get responses from all categories."""
    print("\n" + "="*80)
    print("EXAMPLE 4: Sample Responses from All Categories")
    print("="*80 + "\n")
    
    balorg = BalorgAI()
    
    categories = [
        ("deep_learning_architectures", "Deep Learning"),
        ("model_optimization", "Model Optimization"),
        ("ai_applications", "AI Applications")
    ]
    
    for category_key, category_name in categories:
        print(f"\n--- {category_name} ---")
        response = balorg.get_technical_response(category_key)
        # Print just the header for brevity
        lines = response.split('\n')
        for line in lines[:8]:
            print(line)
        print("... (content truncated for example)")
        print()


def main():
    """Run all examples."""
    print("\n" + "="*80)
    print("BALORG AI - USAGE EXAMPLES")
    print("="*80)
    
    # Run examples
    example_basic_trigger()
    example_specific_category()
    example_list_topics()
    example_all_categories()
    
    print("\n" + "="*80)
    print("To run Balorg AI interactively, execute: python balorg_ai.py")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
