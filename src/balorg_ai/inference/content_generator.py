"""Content generation capabilities for Balorg AI.

This module provides tools for generating various types of content
including articles, stories, dialogue, and more.
"""

import torch
from typing import Optional, List, Dict, Any
from balorg_ai.models.transformer import BalorgTransformer


class ContentGenerator:
    """Content generation system for Balorg AI.
    
    This class provides capabilities for generating high-quality content
    such as articles, stories, dialogue, and more.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Optional[Any] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """Initialize content generator.
        
        Args:
            model: Balorg transformer model
            tokenizer: Tokenizer for encoding/decoding text
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_article(
        self,
        topic: str,
        max_length: int = 500,
        temperature: float = 0.8,
        style: str = "informative",
    ) -> str:
        """Generate an article on a given topic.
        
        Args:
            topic: Topic for the article
            max_length: Maximum article length
            temperature: Sampling temperature
            style: Writing style (informative, creative, technical, etc.)
            
        Returns:
            Generated article text
        """
        prompt = f"Write a {style} article about {topic}.\n\nArticle:\n"
        return self._generate_from_prompt(prompt, max_length, temperature)
    
    def generate_story(
        self,
        premise: str,
        max_length: int = 800,
        temperature: float = 1.0,
        genre: str = "fiction",
    ) -> str:
        """Generate a story based on a premise.
        
        Args:
            premise: Story premise or opening
            max_length: Maximum story length
            temperature: Sampling temperature (higher for more creativity)
            genre: Story genre
            
        Returns:
            Generated story text
        """
        prompt = f"Write a {genre} story based on this premise: {premise}\n\nStory:\n"
        return self._generate_from_prompt(prompt, max_length, temperature)
    
    def generate_dialogue(
        self,
        characters: List[str],
        situation: str,
        num_turns: int = 10,
        temperature: float = 0.9,
    ) -> str:
        """Generate dialogue between characters.
        
        Args:
            characters: List of character names
            situation: Description of the situation
            num_turns: Number of dialogue turns
            temperature: Sampling temperature
            
        Returns:
            Generated dialogue text
        """
        characters_str = ", ".join(characters)
        prompt = f"Write a dialogue between {characters_str} in this situation: {situation}\n\nDialogue:\n"
        
        max_length = num_turns * 50  # Approximate length per turn
        return self._generate_from_prompt(prompt, max_length, temperature)
    
    def generate_summary(
        self,
        text: str,
        max_length: int = 150,
        temperature: float = 0.7,
    ) -> str:
        """Generate a summary of given text.
        
        Args:
            text: Text to summarize
            max_length: Maximum summary length
            temperature: Sampling temperature
            
        Returns:
            Generated summary
        """
        prompt = f"Summarize the following text:\n\n{text}\n\nSummary:\n"
        return self._generate_from_prompt(prompt, max_length, temperature)
    
    def generate_code(
        self,
        description: str,
        language: str = "python",
        max_length: int = 300,
        temperature: float = 0.7,
    ) -> str:
        """Generate code based on description.
        
        Args:
            description: Description of what the code should do
            language: Programming language
            max_length: Maximum code length
            temperature: Sampling temperature
            
        Returns:
            Generated code
        """
        prompt = f"Write {language} code that {description}\n\nCode:\n"
        return self._generate_from_prompt(prompt, max_length, temperature)
    
    def continue_text(
        self,
        text: str,
        max_length: int = 200,
        temperature: float = 0.9,
        num_continuations: int = 1,
    ) -> List[str]:
        """Continue a piece of text with multiple possible continuations.
        
        Args:
            text: Text to continue
            max_length: Maximum continuation length
            temperature: Sampling temperature
            num_continuations: Number of different continuations to generate
            
        Returns:
            List of generated continuations
        """
        continuations = []
        
        for _ in range(num_continuations):
            continuation = self._generate_from_prompt(text, max_length, temperature)
            continuations.append(continuation)
        
        return continuations
    
    def _generate_from_prompt(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
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
            generated_text = self.tokenizer.decode(
                output_ids[0][input_ids.shape[1]:],
                skip_special_tokens=True
            )
        else:
            # Simple decoding (placeholder)
            generated_text = f"Generated content based on: {prompt[:50]}..."
        
        return generated_text
    
    def batch_generate(
        self,
        prompts: List[str],
        max_length: int = 200,
        temperature: float = 0.9,
    ) -> List[str]:
        """Generate text for multiple prompts in batch.
        
        Args:
            prompts: List of input prompts
            max_length: Maximum generation length
            temperature: Sampling temperature
            
        Returns:
            List of generated texts
        """
        results = []
        
        for prompt in prompts:
            result = self._generate_from_prompt(prompt, max_length, temperature)
            results.append(result)
        
        return results
