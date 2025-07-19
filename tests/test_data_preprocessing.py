import pytest
import pandas as pd
from src.data_preprocessing import data_processing, no_of_words

def test_data_processing():
    # Test case for basic text processing
    input_text = "This is a sample review! It includes punctuation, URLs like http://example.com, and stopwords."
    expected_output = "sample review includes punctuation urls like"
    assert data_processing(input_text) == expected_output

    # Test case for empty input
    input_text = ""
    expected_output = ""
    assert data_processing(input_text) == expected_output

    # Test case for input with only stopwords
    input_text = "the and is in"
    expected_output = ""
    assert data_processing(input_text) == expected_output

def test_no_of_words():
    # Test case for counting words
    input_text = "This is a test review."
    expected_count = 5
    assert no_of_words(input_text) == expected_count

    # Test case for empty string
    input_text = ""
    expected_count = 0
    assert no_of_words(input_text) == expected_count

    # Test case for string with multiple spaces
    input_text = "   This   has   extra   spaces   "
    expected_count = 4
    assert no_of_words(input_text) == expected_count

if __name__ == "__main__":
    pytest.main()