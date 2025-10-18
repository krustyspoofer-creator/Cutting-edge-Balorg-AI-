"""Unit tests for inference modules."""

import pytest
import torch
from balorg_ai.models.transformer import create_balorg_model
from balorg_ai.inference.conversational import ConversationalAI
from balorg_ai.inference.content_generator import ContentGenerator
from balorg_ai.inference.question_answering import QuestionAnswerer


class TestConversationalAI:
    """Tests for ConversationalAI."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return create_balorg_model(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
    
    def test_conversational_ai_initialization(self, model):
        """Test initializing conversational AI."""
        ai = ConversationalAI(model)
        assert ai.model is not None
        assert len(ai.history) == 0
    
    def test_chat_adds_to_history(self, model):
        """Test that chat adds to history."""
        ai = ConversationalAI(model)
        ai.chat("Hello", max_length=10)
        
        assert len(ai.history) == 2  # user + assistant
        assert ai.history[0]["role"] == "user"
        assert ai.history[1]["role"] == "assistant"
    
    def test_reset_clears_history(self, model):
        """Test that reset clears history."""
        ai = ConversationalAI(model)
        ai.chat("Hello", max_length=10)
        ai.reset()
        
        assert len(ai.history) == 0
    
    def test_system_prompt(self, model):
        """Test setting system prompt."""
        ai = ConversationalAI(model)
        ai.set_system_prompt("You are a helpful assistant.")
        
        assert len(ai.history) == 1
        assert ai.history[0]["role"] == "system"


class TestContentGenerator:
    """Tests for ContentGenerator."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return create_balorg_model(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
    
    def test_content_generator_initialization(self, model):
        """Test initializing content generator."""
        generator = ContentGenerator(model)
        assert generator.model is not None
    
    def test_generate_article(self, model):
        """Test article generation."""
        generator = ContentGenerator(model)
        article = generator.generate_article(
            topic="AI",
            max_length=20,
            temperature=0.8
        )
        
        assert isinstance(article, str)
        assert len(article) > 0
    
    def test_generate_story(self, model):
        """Test story generation."""
        generator = ContentGenerator(model)
        story = generator.generate_story(
            premise="A robot learns to love",
            max_length=20,
            temperature=1.0
        )
        
        assert isinstance(story, str)
        assert len(story) > 0
    
    def test_batch_generate(self, model):
        """Test batch generation."""
        generator = ContentGenerator(model)
        prompts = ["Topic 1", "Topic 2", "Topic 3"]
        results = generator.batch_generate(prompts, max_length=10)
        
        assert len(results) == len(prompts)
        assert all(isinstance(r, str) for r in results)


class TestQuestionAnswerer:
    """Tests for QuestionAnswerer."""
    
    @pytest.fixture
    def model(self):
        """Create a small test model."""
        return create_balorg_model(
            vocab_size=1000,
            hidden_size=128,
            num_layers=2,
            num_heads=2,
        )
    
    def test_question_answerer_initialization(self, model):
        """Test initializing question answerer."""
        qa = QuestionAnswerer(model)
        assert qa.model is not None
    
    def test_answer_question(self, model):
        """Test basic question answering."""
        qa = QuestionAnswerer(model)
        answer = qa.answer_question(
            question="What is AI?",
            max_length=20
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_answer_question_with_context(self, model):
        """Test question answering with context."""
        qa = QuestionAnswerer(model)
        answer = qa.answer_question(
            question="What is mentioned?",
            context="This is about AI and machine learning.",
            max_length=20
        )
        
        assert isinstance(answer, str)
        assert len(answer) > 0
    
    def test_answer_multiple_choice(self, model):
        """Test multiple choice answering."""
        qa = QuestionAnswerer(model)
        choices = ["Option A", "Option B", "Option C"]
        selected, idx, confidence = qa.answer_multiple_choice(
            question="Pick one",
            choices=choices
        )
        
        assert selected in choices
        assert 0 <= idx < len(choices)
        assert 0 <= confidence <= 1
    
    def test_batch_answer(self, model):
        """Test batch answering."""
        qa = QuestionAnswerer(model)
        questions = ["Q1?", "Q2?", "Q3?"]
        answers = qa.batch_answer(questions, max_length=10)
        
        assert len(answers) == len(questions)
        assert all(isinstance(a, str) for a in answers)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
