from ddt import check

PAGE = "The EU AI Act entered into force on 1 August 2024 after its publication in the Official Journal."


def test_real_quote_passes_despite_case_spacing_and_curly_quotes():
    page = PAGE.replace("Official Journal", "“Official  Journal”")
    claim = {"quote": 'the EU AI Act  entered into force on 1 August 2024 after its publication in the "Official Journal'}
    assert check.check_claim(claim, page)


def test_made_up_quote_fails():
    assert not check.check_claim({"quote": "the EU AI Act was repealed on 3 March 2026 by the Council"}, PAGE)


def test_quote_under_8_words_fails():
    assert not check.check_claim({"quote": "entered into force on 1 August"}, PAGE)


def test_page_text_falls_back_to_extract():
    results = [{"url": "https://example.org/eu-ai-act", "raw_content": ""}]
    assert "1 August 2024" in check.page_text_for("https://example.org/eu-ai-act", results)
