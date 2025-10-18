#!/usr/bin/env python3
"""
Test suite for Balorg AI system.
"""

import unittest
from unittest.mock import patch
from balorg_ai import BalorgAI


class TestBalorgAI(unittest.TestCase):
    """Test cases for the Balorg AI system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.balorg = BalorgAI()
    
    def test_initialization(self):
        """Test that BalorgAI initializes correctly."""
        self.assertIsNotNone(self.balorg)
        self.assertEqual(self.balorg.trigger_phrase, "My Balorg is AI!")
        self.assertIsNotNone(self.balorg.deep_learning_topics)
        self.assertIsNotNone(self.balorg.optimization_topics)
        self.assertIsNotNone(self.balorg.application_topics)
    
    def test_trigger_detection_exact_match(self):
        """Test trigger phrase detection with exact match."""
        self.assertTrue(self.balorg.detect_trigger("My Balorg is AI!"))
    
    def test_trigger_detection_case_insensitive(self):
        """Test trigger phrase detection is case insensitive."""
        self.assertTrue(self.balorg.detect_trigger("my balorg is ai!"))
        self.assertTrue(self.balorg.detect_trigger("MY BALORG IS AI!"))
        self.assertTrue(self.balorg.detect_trigger("My BaLoRg Is Ai!"))
    
    def test_trigger_detection_in_sentence(self):
        """Test trigger phrase detection within larger text."""
        self.assertTrue(self.balorg.detect_trigger(
            "I think My Balorg is AI! and it's amazing"))
    
    def test_trigger_detection_no_match(self):
        """Test trigger phrase detection returns false for non-matching text."""
        self.assertFalse(self.balorg.detect_trigger("Hello world"))
        self.assertFalse(self.balorg.detect_trigger("Balorg AI"))
        self.assertFalse(self.balorg.detect_trigger("My robot is AI"))
    
    def test_deep_learning_topics_loaded(self):
        """Test that deep learning topics are properly loaded."""
        self.assertIn("transformers", self.balorg.deep_learning_topics)
        self.assertIn("neural_networks", self.balorg.deep_learning_topics)
        self.assertIn("activation_functions", self.balorg.deep_learning_topics)
        
        # Check structure
        transformer_topic = self.balorg.deep_learning_topics["transformers"]
        self.assertIn("title", transformer_topic)
        self.assertIn("content", transformer_topic)
        self.assertTrue(len(transformer_topic["content"]) > 100)
    
    def test_optimization_topics_loaded(self):
        """Test that optimization topics are properly loaded."""
        self.assertIn("quantization", self.balorg.optimization_topics)
        self.assertIn("pruning", self.balorg.optimization_topics)
        self.assertIn("distillation", self.balorg.optimization_topics)
        
        # Check content quality
        quant_topic = self.balorg.optimization_topics["quantization"]
        self.assertIn("INT8", quant_topic["content"])
        self.assertIn("quantization", quant_topic["content"].lower())
    
    def test_application_topics_loaded(self):
        """Test that application topics are properly loaded."""
        self.assertIn("nlp", self.balorg.application_topics)
        self.assertIn("computer_vision", self.balorg.application_topics)
        self.assertIn("reinforcement_learning", self.balorg.application_topics)
        
        # Check NLP content
        nlp_topic = self.balorg.application_topics["nlp"]
        self.assertIn("BERT", nlp_topic["content"])
        self.assertIn("GPT", nlp_topic["content"])
    
    def test_get_random_technical_info(self):
        """Test getting random technical information."""
        response = self.balorg.get_random_technical_info()
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 100)
        self.assertIn("=", response)  # Check for formatting
    
    def test_get_topic_by_keyword_transformer(self):
        """Test retrieving topic by keyword - transformer."""
        response = self.balorg.get_topic_by_keyword("transformer")
        self.assertIsNotNone(response)
        self.assertIn("Transformer", response)
    
    def test_get_topic_by_keyword_quantization(self):
        """Test retrieving topic by keyword - quantization."""
        response = self.balorg.get_topic_by_keyword("quantization")
        self.assertIsNotNone(response)
        self.assertIn("Quantization", response)
    
    def test_get_topic_by_keyword_nlp(self):
        """Test retrieving topic by keyword - NLP."""
        response = self.balorg.get_topic_by_keyword("nlp")
        self.assertIsNotNone(response)
        self.assertIn("NLP", response.upper())
    
    def test_get_topic_by_keyword_not_found(self):
        """Test retrieving topic with non-existent keyword."""
        response = self.balorg.get_topic_by_keyword("nonexistentkeyword123")
        self.assertIsNone(response)
    
    def test_list_available_topics(self):
        """Test listing all available topics."""
        response = self.balorg.list_available_topics()
        self.assertIsNotNone(response)
        self.assertIn("Deep Learning Architectures", response)
        self.assertIn("Model Optimization", response)
        self.assertIn("AI Applications", response)
        self.assertIn("Transformer", response)
        self.assertIn("Quantization", response)
    
    def test_process_input_trigger_phrase(self):
        """Test processing input with trigger phrase."""
        response = self.balorg.process_input("My Balorg is AI!")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 100)
    
    def test_process_input_help(self):
        """Test processing input requesting help."""
        response = self.balorg.process_input("help")
        self.assertIn("Available Technical Discussion Topics", response)
        
        response = self.balorg.process_input("show me topics")
        self.assertIn("Deep Learning Architectures", response)
        
        response = self.balorg.process_input("topics")
        self.assertIn("Available Technical Discussion Topics", response)
    
    def test_process_input_keyword_transformer(self):
        """Test processing input with transformer keyword."""
        response = self.balorg.process_input("Tell me about transformers")
        self.assertIn("Transformer", response)
    
    def test_process_input_keyword_bert(self):
        """Test processing input with BERT keyword."""
        response = self.balorg.process_input("What is BERT?")
        self.assertIsNotNone(response)
    
    def test_process_input_keyword_quantization(self):
        """Test processing input with quantization keyword."""
        response = self.balorg.process_input("Explain quantization")
        self.assertIn("Quantization", response)
    
    def test_process_input_keyword_nlp(self):
        """Test processing input with NLP keyword."""
        response = self.balorg.process_input("I want to learn about NLP")
        self.assertIsNotNone(response)
    
    def test_process_input_empty(self):
        """Test processing empty input."""
        response = self.balorg.process_input("")
        self.assertIn("provide input", response.lower())
        
        response = self.balorg.process_input("   ")
        self.assertIn("provide input", response.lower())
    
    def test_process_input_generic(self):
        """Test processing generic input."""
        response = self.balorg.process_input("Hello there")
        self.assertIn("technical discussions", response.lower())
    
    def test_technical_content_quality_transformers(self):
        """Test that transformer content includes key concepts."""
        transformer_content = self.balorg.deep_learning_topics["transformers"]["content"]
        
        # Check for key technical terms
        self.assertIn("Attention", transformer_content)
        self.assertIn("Query", transformer_content)
        self.assertIn("Key", transformer_content)
        self.assertIn("Value", transformer_content)
        self.assertIn("BERT", transformer_content)
        self.assertIn("GPT", transformer_content)
    
    def test_technical_content_quality_quantization(self):
        """Test that quantization content includes key concepts."""
        quant_content = self.balorg.optimization_topics["quantization"]["content"]
        
        # Check for key technical terms
        self.assertIn("INT8", quant_content)
        self.assertIn("Post-Training", quant_content)
        self.assertIn("Quantization-Aware Training", quant_content)
        self.assertIn("precision", quant_content.lower())
    
    def test_technical_content_quality_rl(self):
        """Test that RL content includes key concepts."""
        rl_content = self.balorg.application_topics["reinforcement_learning"]["content"]
        
        # Check for key technical terms
        self.assertIn("Q-Learning", rl_content)
        self.assertIn("Policy", rl_content)
        self.assertIn("Markov", rl_content)
        self.assertIn("DQN", rl_content)
        self.assertIn("PPO", rl_content)
    
    def test_multiple_keyword_detection(self):
        """Test that system can detect multiple keywords."""
        # Test CNN keyword
        response = self.balorg.process_input("What about CNN?")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 50)
        
        # Test attention keyword
        response = self.balorg.process_input("Explain attention mechanism")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 50)
    
    def test_case_insensitive_keyword_search(self):
        """Test that keyword search is case insensitive."""
        response1 = self.balorg.get_topic_by_keyword("TRANSFORMER")
        response2 = self.balorg.get_topic_by_keyword("transformer")
        response3 = self.balorg.get_topic_by_keyword("TrAnSfOrMeR")
        
        self.assertIsNotNone(response1)
        self.assertIsNotNone(response2)
        self.assertIsNotNone(response3)


class TestBalorgAIIntegration(unittest.TestCase):
    """Integration tests for Balorg AI system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.balorg = BalorgAI()
    
    def test_full_conversation_flow(self):
        """Test a complete conversation flow."""
        # Start with help
        response = self.balorg.process_input("help")
        self.assertIn("Available", response)
        
        # Ask about specific topic
        response = self.balorg.process_input("Tell me about transformers")
        self.assertIn("Transformer", response)
        
        # Use trigger phrase
        response = self.balorg.process_input("My Balorg is AI!")
        self.assertIsNotNone(response)
        self.assertTrue(len(response) > 100)
    
    def test_all_topics_accessible(self):
        """Test that all topics can be accessed."""
        # Deep learning topics
        for topic_key in self.balorg.deep_learning_topics.keys():
            response = self.balorg.get_topic_by_keyword(topic_key)
            self.assertIsNotNone(response, f"Failed to retrieve {topic_key}")
        
        # Optimization topics
        for topic_key in self.balorg.optimization_topics.keys():
            response = self.balorg.get_topic_by_keyword(topic_key)
            self.assertIsNotNone(response, f"Failed to retrieve {topic_key}")
        
        # Application topics
        for topic_key in self.balorg.application_topics.keys():
            response = self.balorg.get_topic_by_keyword(topic_key)
            self.assertIsNotNone(response, f"Failed to retrieve {topic_key}")


if __name__ == "__main__":
    unittest.main()
