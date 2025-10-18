"""Unit tests for data processing utilities."""

import pytest
import torch
from balorg_ai.data.dataset import (
    TextDataset,
    ConversationalDataset,
    create_sample_dataset,
    DataCollator,
)


class TestTextDataset:
    """Tests for TextDataset."""
    
    def test_dataset_creation(self):
        """Test creating a text dataset."""
        texts = ["Hello world", "This is a test", "AI is amazing"]
        dataset = TextDataset(texts, max_length=10)
        
        assert len(dataset) == len(texts)
    
    def test_dataset_getitem(self):
        """Test getting items from dataset."""
        texts = ["Hello world", "This is a test"]
        dataset = TextDataset(texts, max_length=10)
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert isinstance(item["input_ids"], torch.Tensor)
        assert isinstance(item["attention_mask"], torch.Tensor)
    
    def test_dataset_length_constraints(self):
        """Test that sequences respect max_length."""
        texts = ["word " * 100]  # Very long text
        max_length = 20
        dataset = TextDataset(texts, max_length=max_length)
        
        item = dataset[0]
        assert item["input_ids"].shape[0] == max_length


class TestConversationalDataset:
    """Tests for ConversationalDataset."""
    
    def test_conversational_dataset_creation(self):
        """Test creating a conversational dataset."""
        conversations = [
            {"prompt": "Hello", "response": "Hi there!"},
            {"prompt": "How are you?", "response": "I'm doing well!"},
        ]
        dataset = ConversationalDataset(conversations, max_length=20)
        
        assert len(dataset) == len(conversations)
    
    def test_conversational_dataset_getitem(self):
        """Test getting items from conversational dataset."""
        conversations = [
            {"prompt": "Hello", "response": "Hi there!"},
        ]
        dataset = ConversationalDataset(conversations, max_length=20)
        
        item = dataset[0]
        
        assert "input_ids" in item
        assert "attention_mask" in item
        assert "labels" in item


class TestDataCollator:
    """Tests for DataCollator."""
    
    def test_collator_batching(self):
        """Test batching examples."""
        collator = DataCollator(pad_token_id=0)
        
        examples = [
            {
                "input_ids": torch.tensor([1, 2, 3, 0, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 0, 0]),
            },
            {
                "input_ids": torch.tensor([4, 5, 6, 7, 0]),
                "attention_mask": torch.tensor([1, 1, 1, 1, 0]),
            },
        ]
        
        batch = collator(examples)
        
        assert "input_ids" in batch
        assert "attention_mask" in batch
        assert batch["input_ids"].shape[0] == 2  # batch size


class TestUtilityFunctions:
    """Tests for utility functions."""
    
    def test_create_sample_dataset(self):
        """Test creating sample dataset."""
        num_samples = 50
        samples = create_sample_dataset(num_samples=num_samples)
        
        assert len(samples) == num_samples
        assert all(isinstance(s, str) for s in samples)
        assert all(len(s) > 0 for s in samples)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
