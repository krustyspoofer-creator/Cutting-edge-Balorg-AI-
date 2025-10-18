"""Example: Question answering with Balorg AI.

This example demonstrates the question answering capabilities
including context-based QA, explanations, and fact extraction.
"""

from balorg_ai.models.transformer import create_balorg_model
from balorg_ai.inference.question_answering import QuestionAnswerer


def main():
    """Main function demonstrating question answering."""
    print("=" * 50)
    print("Balorg AI - Question Answering Demo")
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
    
    # Create question answerer
    print("\n2. Initializing question answerer...")
    qa = QuestionAnswerer(model)
    
    print("\n" + "=" * 50)
    print("Question Answering Examples")
    print("=" * 50)
    
    # Example 1: Simple question answering
    print("\n" + "-" * 50)
    print("Example 1: Basic Question Answering")
    print("-" * 50)
    
    question = "What is machine learning?"
    answer = qa.answer_question(question, max_length=100, temperature=0.7)
    
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Example 2: Context-based question answering
    print("\n" + "-" * 50)
    print("Example 2: Context-Based QA")
    print("-" * 50)
    
    context = (
        "Balorg AI is a cutting-edge connectionist AI system that leverages "
        "transformer architectures and deep learning. It can perform various "
        "tasks including conversation, content generation, and question answering. "
        "The system is designed for high performance and scalability."
    )
    question = "What tasks can Balorg AI perform?"
    answer = qa.answer_question(question, context=context, max_length=100)
    
    print(f"\nContext: {context}")
    print(f"\nQuestion: {question}")
    print(f"Answer: {answer}")
    
    # Example 3: Multiple choice question
    print("\n" + "-" * 50)
    print("Example 3: Multiple Choice Question")
    print("-" * 50)
    
    question = "What is the main architecture used in modern language models?"
    choices = [
        "Convolutional Neural Networks",
        "Recurrent Neural Networks",
        "Transformer Architecture",
        "Decision Trees"
    ]
    
    selected_answer, choice_idx, confidence = qa.answer_multiple_choice(
        question, choices
    )
    
    print(f"\nQuestion: {question}")
    print("\nChoices:")
    for i, choice in enumerate(choices):
        marker = "✓" if i == choice_idx else " "
        print(f"  {marker} {chr(65+i)}. {choice}")
    print(f"\nSelected: {selected_answer}")
    print(f"Confidence: {confidence:.2f}")
    
    # Example 4: Answer with explanation
    print("\n" + "-" * 50)
    print("Example 4: Answer with Explanation")
    print("-" * 50)
    
    question = "Why is deep learning important for AI?"
    result = qa.answer_with_explanation(question, max_length=200, temperature=0.7)
    
    print(f"\nQuestion: {question}")
    print(f"\nAnswer: {result['answer']}")
    if result['explanation']:
        print(f"\nExplanation: {result['explanation']}")
    
    # Example 5: Batch question answering
    print("\n" + "-" * 50)
    print("Example 5: Batch Question Answering")
    print("-" * 50)
    
    questions = [
        "What is natural language processing?",
        "How do neural networks learn?",
        "What is the purpose of attention mechanisms?",
    ]
    
    answers = qa.batch_answer(questions, max_length=80, temperature=0.7)
    
    print("\nBatch QA Results:")
    for q, a in zip(questions, answers):
        print(f"\nQ: {q}")
        print(f"A: {a}")
    
    # Example 6: Fact extraction
    print("\n" + "-" * 50)
    print("Example 6: Fact Extraction")
    print("-" * 50)
    
    text = (
        "Artificial intelligence has made remarkable progress in recent years. "
        "Deep learning models can now perform complex tasks such as image recognition, "
        "natural language understanding, and game playing at superhuman levels. "
        "Transformer models, introduced in 2017, revolutionized NLP. "
        "These models use self-attention mechanisms to process sequences efficiently."
    )
    
    facts = qa.extract_facts(text, max_facts=5, temperature=0.7)
    
    print(f"\nText: {text}")
    print("\nExtracted Facts:")
    for i, fact in enumerate(facts, 1):
        print(f"  {i}. {fact}")
    
    # Example 7: Claim verification
    print("\n" + "-" * 50)
    print("Example 7: Claim Verification")
    print("-" * 50)
    
    claim = "Transformer models were introduced in 2017."
    evidence = (
        "The transformer architecture was first described in the paper "
        "'Attention is All You Need' by Vaswani et al., published in 2017."
    )
    
    verification = qa.verify_claim(claim, evidence, temperature=0.5)
    
    print(f"\nClaim: {claim}")
    print(f"\nEvidence: {evidence}")
    print(f"\nVerdict: {verification['verdict']}")
    print(f"Reasoning: {verification['reasoning']}")
    
    print("\n" + "=" * 50)
    print("Question answering demo completed!")
    print("=" * 50)


if __name__ == "__main__":
    main()
