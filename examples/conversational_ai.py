"""Example: Using Balorg AI for conversation.

This example demonstrates how to use the conversational AI interface
to engage in natural dialogue.
"""

from balorg_ai.models.transformer import create_balorg_model
from balorg_ai.inference.conversational import ConversationalAI


def main():
    """Main function demonstrating conversational AI."""
    print("=" * 50)
    print("Balorg AI - Conversational Interface Demo")
    print("=" * 50)
    
    # Create model
    print("\n1. Creating model...")
    model = create_balorg_model(
        vocab_size=50257,
        hidden_size=384,
        num_layers=6,
        num_heads=6,
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create conversational AI
    print("\n2. Initializing conversational AI...")
    ai = ConversationalAI(model, max_history=5)
    
    # Set system prompt
    system_prompt = (
        "You are Balorg AI, a helpful and knowledgeable assistant. "
        "You provide clear, accurate, and friendly responses."
    )
    ai.set_system_prompt(system_prompt)
    
    print("\n" + "=" * 50)
    print("Conversational AI Ready!")
    print("Type 'quit' to exit, 'reset' to clear history")
    print("=" * 50)
    
    # Interactive loop
    while True:
        print("\n" + "-" * 50)
        user_input = input("You: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye!")
            break
        
        if user_input.lower() == 'reset':
            ai.reset()
            print("Conversation history cleared.")
            continue
        
        if not user_input:
            continue
        
        # Generate response
        print("\nBalorg AI: ", end="", flush=True)
        response = ai.chat(
            user_input,
            max_length=100,
            temperature=0.9,
        )
        print(response)
    
    # Show conversation history
    print("\n" + "=" * 50)
    print("Conversation History:")
    print("=" * 50)
    
    history = ai.get_history()
    for i, turn in enumerate(history):
        role = turn['role'].capitalize()
        content = turn['content']
        print(f"\n{i+1}. {role}: {content}")


def demo_mode():
    """Run a demo conversation without user interaction."""
    print("=" * 50)
    print("Balorg AI - Conversational Demo Mode")
    print("=" * 50)
    
    # Create model and AI
    model = create_balorg_model(hidden_size=384, num_layers=6, num_heads=6)
    ai = ConversationalAI(model)
    
    # Demo conversation
    demo_inputs = [
        "Hello! What can you help me with?",
        "Tell me about artificial intelligence.",
        "What are some applications of AI?",
        "Thank you for the information!",
    ]
    
    print("\n" + "=" * 50)
    print("Demo Conversation:")
    print("=" * 50)
    
    for user_input in demo_inputs:
        print(f"\nUser: {user_input}")
        response = ai.chat(user_input, max_length=80, temperature=0.8)
        print(f"Balorg AI: {response}")
    
    print("\n" + "=" * 50)
    print("Demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        demo_mode()
    else:
        main()
