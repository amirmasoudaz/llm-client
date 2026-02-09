# src/utils/markdown_email.py
"""
Lightweight markdown-to-HTML converter for email formatting.

Handles the specific Markdown patterns used in email templates:
- Bold: *text* → <b>text</b>
- Italic: _text_ → <i>text</i>
- Newlines: \n → <br>
"""

import html
import re


def markdown_to_html(text: str) -> str:
    """
    Convert Markdown-style formatting to HTML for email bodies.
    
    Supports:
    - *text* → <b>text</b> (bold)
    - _text_ → <i>text</i> (italic)
    - Newlines → <br>
    
    Args:
        text: Plain text with Markdown-style formatting
        
    Returns:
        HTML-formatted string safe for email
    """
    if not text:
        return ""
    
    # First, escape HTML special characters to prevent XSS
    result = html.escape(text)
    
    # Convert bold: *text* → <b>text</b>
    # Match *text* but not **text** (which would be double asterisks)
    # Use negative lookbehind/lookahead to avoid matching ** or escaped \*
    result = re.sub(
        r'(?<!\*)\*([^*\n]+?)\*(?!\*)',
        r'<b>\1</b>',
        result
    )
    
    # Convert italic: _text_ → <i>text</i>
    # Match _text_ but not __text__ or within words (e.g., snake_case)
    # Only match when underscore is at word boundaries
    result = re.sub(
        r'(?<![a-zA-Z0-9])_([^_\n]+?)_(?![a-zA-Z0-9])',
        r'<i>\1</i>',
        result
    )
    
    # Convert newlines to <br> tags
    result = result.replace("\n", "<br>\n")
    
    return result
