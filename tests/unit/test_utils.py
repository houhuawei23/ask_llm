"""Unit tests for utility modules."""

import pytest

from ask_llm.utils.token_counter import TokenCounter
from ask_llm.utils.file_handler import FileHandler


class TestTokenCounter:
    """Test TokenCounter."""
    
    def test_count_words(self):
        """Test word counting."""
        assert TokenCounter.count_words("") == 0
        assert TokenCounter.count_words("Hello") == 1
        assert TokenCounter.count_words("Hello world") == 2
        assert TokenCounter.count_words("  Multiple   spaces  ") == 2
    
    def test_count_characters(self):
        """Test character counting."""
        assert TokenCounter.count_characters("") == 0
        assert TokenCounter.count_characters("Hello") == 5
        assert TokenCounter.count_characters("Hello world") == 11
    
    def test_estimate_tokens(self):
        """Test token estimation."""
        text = "Hello world"
        stats = TokenCounter.estimate_tokens(text)
        
        assert "word_count" in stats
        assert "token_count" in stats
        assert "char_count" in stats
        assert stats["word_count"] == 2
        assert stats["char_count"] == 11
    
    def test_get_encoding(self):
        """Test encoding selection."""
        assert TokenCounter._get_encoding(None) == TokenCounter.DEFAULT_ENCODING
        assert TokenCounter._get_encoding("gpt-4") == "cl100k_base"
        assert TokenCounter._get_encoding("deepseek-chat") == "cl100k_base"
    
    def test_format_stats(self):
        """Test stats formatting."""
        text = "Hello world"
        formatted = TokenCounter.format_stats(text)
        
        assert "Words:" in formatted
        assert "Tokens:" in formatted
        assert "Chars:" in formatted


class TestFileHandler:
    """Test FileHandler."""
    
    def test_read_file(self, temp_dir):
        """Test reading file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("Hello world")
        
        content = FileHandler.read(test_file)
        assert content == "Hello world"
    
    def test_read_nonexistent_file(self, temp_dir):
        """Test reading non-existent file raises error."""
        with pytest.raises(FileNotFoundError):
            FileHandler.read(temp_dir / "nonexistent.txt")
    
    def test_write_file(self, temp_dir):
        """Test writing file."""
        test_file = temp_dir / "output.txt"
        
        FileHandler.write(test_file, "Test content")
        
        assert test_file.exists()
        assert test_file.read_text() == "Test content"
    
    def test_write_file_exists(self, temp_dir):
        """Test writing to existing file without force raises error."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("Original")
        
        with pytest.raises(FileExistsError):
            FileHandler.write(test_file, "New content", force=False)
    
    def test_write_file_force(self, temp_dir):
        """Test writing to existing file with force."""
        test_file = temp_dir / "exists.txt"
        test_file.write_text("Original")
        
        FileHandler.write(test_file, "New content", force=True)
        
        assert test_file.read_text() == "New content"
    
    def test_generate_output_path(self, temp_dir):
        """Test output path generation."""
        input_path = temp_dir / "input.txt"
        
        output = FileHandler.generate_output_path(input_path)
        assert output.endswith("input_output.txt")
    
    def test_generate_output_path_custom(self, temp_dir):
        """Test custom output path."""
        input_path = temp_dir / "input.txt"
        custom = temp_dir / "custom.md"
        
        output = FileHandler.generate_output_path(input_path, custom)
        assert output == str(custom)
    
    def test_detect_type(self):
        """Test file type detection."""
        assert FileHandler.detect_type("file.txt") == ".txt"
        assert FileHandler.detect_type("file.MD") == ".md"
        assert FileHandler.detect_type("/path/to/file.py") == ".py"
    
    def test_is_text_file(self):
        """Test text file detection."""
        assert FileHandler.is_text_file("file.txt") is True
        assert FileHandler.is_text_file("file.py") is True
        assert FileHandler.is_text_file("file.md") is True
        assert FileHandler.is_text_file("file.bin") is False
