"""Question answering system for Balorg AI.

This module provides capabilities for answering questions based on
context or general knowledge.
"""

import torch
from typing import Optional, List, Dict, Any, Tuple
from balorg_ai.models.transformer import BalorgTransformer


class QuestionAnswerer:
    """Question answering system for Balorg AI.
    
    This class provides capabilities for answering complex questions
    and providing informative responses.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize question answerer.
        
        Args:
            model: Balorg transformer model
            tokenizer: Tokenizer for encoding/decoding text
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
    ) -> str:
        """Answer a question, optionally with context.
        
        Args:
            question: Question to answer
            context: Optional context for answering
            max_length: Maximum answer length
            temperature: Sampling temperature
            
        Returns:
            Generated answer
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nAnswer:"
        
        return self._generate_answer(prompt, max_length, temperature)
    
    def answer_multiple_choice(
        self,
        question: str,
        choices: List[str],
        context: Optional[str] = None,
    ) -> Tuple[str, int, float]:
        """Answer a multiple choice question.
        
        Args:
            question: Question to answer
            choices: List of answer choices
            context: Optional context
            
        Returns:
            Tuple of (selected answer, choice index, confidence score)
        """
        # Build prompt with choices
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nChoices:\n{choices_str}\n\nCorrect answer:"
        else:
            prompt = f"Question: {question}\n\nChoices:\n{choices_str}\n\nCorrect answer:"
        
        # Generate answer
        answer = self._generate_answer(prompt, max_length=10, temperature=0.3)
        
        # Parse answer to find selected choice
        answer_upper = answer.strip().upper()
        
        # Try to find letter or match text
        for i, choice in enumerate(choices):
            letter = chr(65 + i)
            if letter in answer_upper[:5] or choice.lower() in answer.lower():
                return choice, i, 0.9  # Simplified confidence
        
        # Default to first choice if no match
        return choices[0], 0, 0.5
    
    def answer_with_explanation(
        self,
        question: str,
        context: Optional[str] = None,
        max_length: int = 300,
        temperature: float = 0.7,
    ) -> Dict[str, str]:
        """Answer a question with detailed explanation.
        
        Args:
            question: Question to answer
            context: Optional context
            max_length: Maximum response length
            temperature: Sampling temperature
            
        Returns:
            Dictionary with 'answer' and 'explanation' keys
        """
        if context:
            prompt = f"Context: {context}\n\nQuestion: {question}\n\nProvide a detailed answer with explanation:\n\nAnswer:"
        else:
            prompt = f"Question: {question}\n\nProvide a detailed answer with explanation:\n\nAnswer:"
        
        full_response = self._generate_answer(prompt, max_length, temperature)
        
        # Try to split into answer and explanation
        if "Explanation:" in full_response:
            parts = full_response.split("Explanation:", 1)
            return {
                "answer": parts[0].strip(),
                "explanation": parts[1].strip() if len(parts) > 1 else ""
            }
        else:
            # Return full response as answer
            return {
                "answer": full_response,
                "explanation": ""
            }
    
    def batch_answer(
        self,
        questions: List[str],
        contexts: Optional[List[str]] = None,
        max_length: int = 150,
        temperature: float = 0.7,
    ) -> List[str]:
        """Answer multiple questions in batch.
        
        Args:
            questions: List of questions
            contexts: Optional list of contexts (same length as questions)
            max_length: Maximum answer length
            temperature: Sampling temperature
            
        Returns:
            List of generated answers
        """
        if contexts is None:
            contexts = [None] * len(questions)
        
        answers = []
        for question, context in zip(questions, contexts):
            answer = self.answer_question(question, context, max_length, temperature)
            answers.append(answer)
        
        return answers
    
    def extract_facts(
        self,
        text: str,
        max_facts: int = 5,
        temperature: float = 0.7,
    ) -> List[str]:
        """Extract key facts from text.
        
        Args:
            text: Text to extract facts from
            max_facts: Maximum number of facts to extract
            temperature: Sampling temperature
            
        Returns:
            List of extracted facts
        """
        prompt = f"Extract {max_facts} key facts from the following text:\n\n{text}\n\nKey facts:\n"
        
        response = self._generate_answer(prompt, max_length=300, temperature=temperature)
        
        # Parse facts (assuming numbered or bulleted list)
        facts = []
        for line in response.split('\n'):
            line = line.strip()
            if line and (line[0].isdigit() or line.startswith('-') or line.startswith('•')):
                # Remove numbering or bullets
                fact = line.lstrip('0123456789.-•').strip()
                if fact:
                    facts.append(fact)
        
        return facts[:max_facts]
    
    def verify_claim(
        self,
        claim: str,
        evidence: str,
        temperature: float = 0.5,
    ) -> Dict[str, Any]:
        """Verify a claim against evidence.
        
        Args:
            claim: Claim to verify
            evidence: Evidence text
            temperature: Sampling temperature
            
        Returns:
            Dictionary with 'verdict' (supported/refuted/uncertain) and 'reasoning'
        """
        prompt = f"Evidence: {evidence}\n\nClaim: {claim}\n\nIs the claim supported by the evidence? Provide verdict (supported/refuted/uncertain) and reasoning:\n\nVerdict:"
        
        response = self._generate_answer(prompt, max_length=200, temperature=temperature)
        
        # Parse verdict
        response_lower = response.lower()
        verdict = "uncertain"
        
        if "supported" in response_lower[:50]:
            verdict = "supported"
        elif "refuted" in response_lower[:50] or "not supported" in response_lower[:50]:
            verdict = "refuted"
        
        return {
            "verdict": verdict,
            "reasoning": response
        }
    
    def _generate_answer(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> str:
        """Generate answer from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated answer
        """
        # Tokenize
        if self.tokenizer:
            input_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"]
        else:
            # Simple tokenization
            tokens = prompt.lower().split()
            input_ids = torch.tensor([[hash(token) % 50000 for token in tokens]])
        
        input_ids = input_ids.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        # Decode
        if self.tokenizer:
            answer = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
        else:
            # Simple decoding (placeholder)
            answer = f"Answer to question based on the provided context and knowledge."
        
        return answer.strip()
