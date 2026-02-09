# src/utils/test_markdown_email.py
"""
Unit tests for markdown_email.py
"""

import pytest
from src.utils.markdown_email import markdown_to_html


class TestMarkdownToHtml:
    """Tests for the markdown_to_html function."""

    def test_empty_string(self):
        assert markdown_to_html("") == ""

    def test_plain_text_no_formatting(self):
        text = "Hello, this is plain text."
        result = markdown_to_html(text)
        assert result == "Hello, this is plain text."

    def test_bold_single_word(self):
        text = "This is *bold* text."
        result = markdown_to_html(text)
        assert result == "This is <b>bold</b> text."

    def test_bold_multiple_words(self):
        text = "This is *bold and strong* text."
        result = markdown_to_html(text)
        assert result == "This is <b>bold and strong</b> text."

    def test_multiple_bold_in_text(self):
        text = "I completed my *M.Sc.* at *Amirkabir University*."
        result = markdown_to_html(text)
        assert result == "I completed my <b>M.Sc.</b> at <b>Amirkabir University</b>."

    def test_italic_single_word(self):
        text = "This is _italic_ text."
        result = markdown_to_html(text)
        assert result == "This is <i>italic</i> text."

    def test_italic_multiple_words(self):
        text = "This is _italic and emphasized_ text."
        result = markdown_to_html(text)
        assert result == "This is <i>italic and emphasized</i> text."

    def test_bold_and_italic_together(self):
        text = "This has *bold* and _italic_ formatting."
        result = markdown_to_html(text)
        assert result == "This has <b>bold</b> and <i>italic</i> formatting."

    def test_newlines_converted_to_br(self):
        text = "Line one.\nLine two.\nLine three."
        result = markdown_to_html(text)
        assert result == "Line one.<br>\nLine two.<br>\nLine three."

    def test_html_special_characters_escaped(self):
        text = "Use <script> tags & special chars."
        result = markdown_to_html(text)
        assert "&lt;script&gt;" in result
        assert "&amp;" in result

    def test_underscore_in_snake_case_not_converted(self):
        """Underscores within words (like snake_case) should not become italic."""
        text = "The variable_name is used here."
        result = markdown_to_html(text)
        # Should not convert underscores within words
        assert "<i>" not in result

    def test_real_email_sample(self):
        """Test with a sample from the actual email template."""
        text = """Dear Professor Azadfar,

I completed my *M.Sc.* in electrical engineering at *Amirkabir University of Technology*.

*My GPA is 18.88 out of 20* (equivalent to a grade A+).

Best regards,
AmirMasoud"""
        
        result = markdown_to_html(text)
        
        assert "<b>M.Sc.</b>" in result
        assert "<b>Amirkabir University of Technology</b>" in result
        assert "<b>My GPA is 18.88 out of 20</b>" in result
        assert "<br>" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
