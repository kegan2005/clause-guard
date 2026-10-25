"""Tests for safe HTML highlighting."""
from ml.highlight import build_highlighted_html, esc_text
from ml.inference import analyze_text


def test_html_is_escaped_and_single_line(empty_models_dir):
    text = "<script>alert(1)</script>\n\nCustomer shall not engage in any business that competes with Supplier. Fee is $5."
    html = build_highlighted_html(text, analyze_text(text).results)
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
    assert "\n" not in html and "$" not in html
    assert "[HIGH]" in html


def test_visible_text_equals_original(empty_models_dir):
    import re
    from html import unescape

    text = "Customer shall not engage in any business that competes with Supplier. The sky is blue."
    html = build_highlighted_html(text, analyze_text(text).results)
    stripped = re.sub(r"<[^>]+>", "", html.replace("[HIGH]", ""))
    assert unescape(stripped) == text


def test_esc_text():
    assert esc_text("a<b>&\n$") == "a&lt;b&gt;&amp;&#10;&#36;"
