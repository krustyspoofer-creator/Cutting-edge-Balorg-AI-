#!/usr/bin/env python3
"""
Tests for Balorg AI functionality
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from balorg_ai import BalorgAI


def test_initialization():
    """Test that BalorgAI initializes correctly."""
    balorg = BalorgAI()
    assert balorg.trigger_phrase == "My Balorg is AI!"
    assert len(balorg.knowledge_base) > 0
    assert hasattr(balorg, 'title_to_key')
    assert len(balorg.title_to_key) == len(balorg.knowledge_base)
    print("✓ Initialization test passed")


def test_trigger_detection():
    """Test trigger phrase detection."""
    balorg = BalorgAI()
    
    # Test exact match
    assert balorg.detect_trigger("My Balorg is AI!")
    
    # Test case insensitivity
    assert balorg.detect_trigger("my balorg is ai!")
    assert balorg.detect_trigger("MY BALORG IS AI!")
    
    # Test with surrounding text
    assert balorg.detect_trigger("Hello! My Balorg is AI! Tell me more.")
    
    # Test non-matching text
    assert not balorg.detect_trigger("Hello world")
    assert not balorg.detect_trigger("My AI is Balorg")
    
    print("✓ Trigger detection test passed")


def test_knowledge_base_structure():
    """Test knowledge base has correct structure."""
    balorg = BalorgAI()
    
    # Verify knowledge base is non-empty and has expected structure
    assert len(balorg.knowledge_base) > 0, "Knowledge base should not be empty"
    
    # Verify all categories have required structure
    for category in balorg.knowledge_base.keys():
        assert "title" in balorg.knowledge_base[category]
        assert "topics" in balorg.knowledge_base[category]
        assert len(balorg.knowledge_base[category]["topics"]) > 0
        
        for topic in balorg.knowledge_base[category]["topics"]:
            assert "name" in topic
            assert "content" in topic
            assert len(topic["content"]) > 0
    
    print("✓ Knowledge base structure test passed")


def test_technical_response():
    """Test technical response generation."""
    balorg = BalorgAI()
    
    # Test random response
    response = balorg.get_technical_response()
    assert "BALORG AI TECHNICAL RESPONSE" in response
    assert "Category:" in response
    assert "Topic:" in response
    assert len(response) > 100
    
    # Test category-specific responses
    for category in balorg.knowledge_base.keys():
        response = balorg.get_technical_response(category)
        assert "BALORG AI TECHNICAL RESPONSE" in response
        assert len(response) > 100
    
    print("✓ Technical response test passed")


def test_category_listing():
    """Test that all categories can be listed."""
    balorg = BalorgAI()
    
    # Should not raise any exceptions
    import io
    from contextlib import redirect_stdout
    
    f = io.StringIO()
    with redirect_stdout(f):
        balorg.list_categories()
    
    output = f.getvalue()
    assert "Deep Learning Architectures" in output
    assert "Model Optimization" in output
    assert "AI Applications" in output
    
    print("✓ Category listing test passed")


def test_invalid_category():
    """Test handling of invalid category."""
    balorg = BalorgAI()
    response = balorg.get_technical_response("invalid_category")
    assert "don't have information" in response.lower()
    print("✓ Invalid category test passed")


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*60)
    print("Running Balorg AI Tests")
    print("="*60 + "\n")
    
    try:
        test_initialization()
        test_trigger_detection()
        test_knowledge_base_structure()
        test_technical_response()
        test_category_listing()
        test_invalid_category()
        
        print("\n" + "="*60)
        print("All tests passed! ✓")
        print("="*60 + "\n")
        return True
    except AssertionError as e:
        print(f"\n✗ Test failed: {e}\n")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
