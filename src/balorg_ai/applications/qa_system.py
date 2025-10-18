"""
Question Answering application for Balorg AI.

Answers complex questions and provides informative responses.
"""

import torch
from typing import Optional, List, Dict, Any
from ..models.transformer import BalorgTransformer
from ..models.config import ModelConfig


class QuestionAnsweringSystem:
    """
    Question answering system using Balorg Transformer.
    
    Provides intelligent answers to questions based on context or
    general knowledge.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize question answering system.
        
        Args:
            model: Balorg Transformer model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def answer_question(
        self,
        question: str,
        context: Optional[str] = None,
        max_length: int = 150,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Answer a question.
        
        Args:
            question: Question to answer
            context: Optional context to base answer on
            max_length: Maximum answer length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated answer
        """
        # Build prompt
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate answer
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        # Decode answer
        full_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract answer (text after "Answer:")
        if "Answer:" in full_text:
            answer = full_text.split("Answer:")[-1].strip()
        else:
            answer = full_text[len(prompt):].strip()
        
        return answer
    
    def answer_multiple_questions(
        self,
        questions: List[str],
        context: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        Answer multiple questions.
        
        Args:
            questions: List of questions
            context: Optional context for all questions
            **kwargs: Additional arguments for answer_question
            
        Returns:
            List of answers
        """
        answers = []
        for question in questions:
            answer = self.answer_question(question, context, **kwargs)
            answers.append(answer)
        return answers
    
    def extractive_qa(
        self,
        question: str,
        context: str,
        max_length: int = 50,
        temperature: float = 0.5,
    ) -> str:
        """
        Perform extractive question answering.
        
        Extracts the answer directly from the provided context.
        
        Args:
            question: Question to answer
            context: Context containing the answer
            max_length: Maximum answer length
            temperature: Sampling temperature
            
        Returns:
            Extracted answer
        """
        return self.answer_question(
            question,
            context=context,
            max_length=max_length,
            temperature=temperature,
            top_k=30,
            top_p=0.85,
        )
    
    def open_domain_qa(
        self,
        question: str,
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Perform open-domain question answering.
        
        Answers questions without requiring specific context, relying on
        the model's learned knowledge.
        
        Args:
            question: Question to answer
            max_length: Maximum answer length
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        return self.answer_question(
            question,
            context=None,
            max_length=max_length,
            temperature=temperature,
        )
    
    def multi_hop_qa(
        self,
        question: str,
        contexts: List[str],
        max_length: int = 200,
        temperature: float = 0.7,
    ) -> str:
        """
        Perform multi-hop question answering.
        
        Answers questions that require reasoning over multiple contexts.
        
        Args:
            question: Question to answer
            contexts: List of context passages
            max_length: Maximum answer length
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        # Combine contexts
        combined_context = "\n\n".join([f"Passage {i+1}: {ctx}" for i, ctx in enumerate(contexts)])
        
        return self.answer_question(
            question,
            context=combined_context,
            max_length=max_length,
            temperature=temperature,
        )
    
    def interactive_qa(self):
        """
        Start an interactive Q&A session.
        
        Continues until user types 'exit', 'quit', or 'bye'.
        """
        print("Balorg AI Question Answering System")
        print("Type 'exit', 'quit', or 'bye' to end the session")
        print("-" * 50)
        
        while True:
            question = input("\nQuestion: ").strip()
            
            if question.lower() in ['exit', 'quit', 'bye']:
                print("\nBalorg AI: Session ended. Thank you!")
                break
            
            if not question:
                continue
            
            answer = self.answer_question(question)
            print(f"\nAnswer: {answer}")
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "QuestionAnsweringSystem":
        """
        Load question answering system from pretrained model.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            device: Device to run inference on
            
        Returns:
            QuestionAnsweringSystem instance
        """
        # Load config
        config = ModelConfig.from_json_file(f"{model_path}/config.json")
        
        # Create model
        model = BalorgTransformer(config)
        
        # Load weights
        state_dict = torch.load(f"{model_path}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        
        return cls(model, tokenizer, device=device)
