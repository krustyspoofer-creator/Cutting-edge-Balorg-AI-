"""
Content generation application for Balorg AI.

Generates high-quality content including articles, stories, and dialogue.
"""

import torch
from typing import Optional, List, Dict, Any
from ..models.transformer import BalorgTransformer
from ..models.config import ModelConfig


class ContentGenerator:
    """
    Content generator using Balorg Transformer.
    
    Generates various types of content including articles, stories,
    dialogue, and creative writing.
    """
    
    def __init__(
        self,
        model: BalorgTransformer,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialize content generator.
        
        Args:
            model: Balorg Transformer model
            tokenizer: Tokenizer instance
            device: Device to run inference on
        """
        self.model = model.to(device)
        self.model.eval()
        self.tokenizer = tokenizer
        self.device = device
    
    def generate_article(
        self,
        title: str,
        keywords: Optional[List[str]] = None,
        max_length: int = 500,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate an article based on title and keywords.
        
        Args:
            title: Article title
            keywords: Optional list of keywords to include
            max_length: Maximum article length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated article text
        """
        # Build prompt
        prompt = f"Title: {title}\n\n"
        if keywords:
            prompt += f"Keywords: {', '.join(keywords)}\n\n"
        prompt += "Article:\n"
        
        return self._generate_text(prompt, max_length, temperature, top_k, top_p)
    
    def generate_story(
        self,
        genre: str,
        characters: Optional[List[str]] = None,
        setting: Optional[str] = None,
        max_length: int = 800,
        temperature: float = 0.9,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate a creative story.
        
        Args:
            genre: Story genre (e.g., "fantasy", "sci-fi", "mystery")
            characters: Optional list of character names
            setting: Optional story setting
            max_length: Maximum story length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated story text
        """
        # Build prompt
        prompt = f"Genre: {genre}\n"
        if characters:
            prompt += f"Characters: {', '.join(characters)}\n"
        if setting:
            prompt += f"Setting: {setting}\n"
        prompt += "\nStory:\n"
        
        return self._generate_text(prompt, max_length, temperature, top_k, top_p)
    
    def generate_dialogue(
        self,
        context: str,
        num_turns: int = 5,
        max_length: int = 300,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Generate dialogue between characters.
        
        Args:
            context: Context or scenario for the dialogue
            num_turns: Number of dialogue turns
            max_length: Maximum dialogue length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated dialogue text
        """
        prompt = f"Context: {context}\n\nDialogue:\n"
        return self._generate_text(prompt, max_length, temperature, top_k, top_p)
    
    def generate_continuation(
        self,
        text: str,
        max_length: int = 200,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        """
        Continue existing text.
        
        Args:
            text: Existing text to continue
            max_length: Maximum continuation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Text continuation
        """
        return self._generate_text(text, max_length, temperature, top_k, top_p)
    
    def _generate_text(
        self,
        prompt: str,
        max_length: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> str:
        """
        Internal method to generate text.
        
        Args:
            prompt: Input prompt
            max_length: Maximum generation length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            
        Returns:
            Generated text
        """
        # Tokenize prompt
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # Generate text
        with torch.no_grad():
            output_ids = self.model.generate(
                input_ids,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        return generated_text
    
    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        tokenizer: Any,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> "ContentGenerator":
        """
        Load content generator from pretrained model.
        
        Args:
            model_path: Path to model checkpoint
            tokenizer: Tokenizer instance
            device: Device to run inference on
            
        Returns:
            ContentGenerator instance
        """
        # Load config
        config = ModelConfig.from_json_file(f"{model_path}/config.json")
        
        # Create model
        model = BalorgTransformer(config)
        
        # Load weights
        state_dict = torch.load(f"{model_path}/model.pt", map_location=device)
        model.load_state_dict(state_dict)
        
        return cls(model, tokenizer, device=device)
