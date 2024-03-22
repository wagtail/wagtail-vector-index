import pytest
from wagtail_vector_index.ai_utils.text_splitting.dummy import (
    DummyLengthCalculator,
)
from wagtail_vector_index.ai_utils.text_splitting.naive import (
    NaiveTextSplitterCalculator,
)

LENGTH_CALCULATOR_SAMPLE_TEXTS = [
    """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Morbi ornare magna et urna volutpat, ut fermentum velit tincidunt.
    Aliquam erat volutpat. Nam erat mi, porta eu scelerisque sed, pharetra eget quam.
    Sed aliquet massa purus, vel sagittis libero fermentum nec.
    Donec placerat leo in tortor semper, sit amet venenatis ipsum tincidunt. Fusce at porttitor orci.
    Donec nibh diam, consectetur a sagittis eu, laoreet vitae erat.
    Aliquam bibendum dolor sed ornare aliquet. Aliquam sodales,
    felis nec aliquet condimentum, sem lacus placerat est...""",
    """Lorem ipsum dolor sit amet, consectetur adipiscing elit.
    Morbi ornare magna et urna volutpat, ut fermentum velit tincidunt.
    Aliquam erat volutpat. Nam erat mi, porta eu scelerisque sed, pharetra eget quam.
    Sed aliquet massa purus, vel sagittis libero fermentum nec.
    Donec placerat leo in tortor semper, sit amet venenatis ipsum tincidunt. Fusce at porttitor orci.
    Donec nibh diam, consectetur a sagittis eu, laoreet vitae erat.
    Aliquam bibendum dolor sed ornare aliquet. Aliquam sodales,
    felis nec aliquet condimentum, sem lacus placerat est...

    Test.""",
]

NAIVE_LENGTH_CALCULATOR_TESTS_TABLE = [
    (LENGTH_CALCULATOR_SAMPLE_TEXTS[0], 143),
    (LENGTH_CALCULATOR_SAMPLE_TEXTS[1], 146),
]


@pytest.mark.parametrize("test_input,expected", NAIVE_LENGTH_CALCULATOR_TESTS_TABLE)
def test_naive_text_splitter_length_calculator(test_input, expected):
    length_calculator = NaiveTextSplitterCalculator()
    assert length_calculator.get_splitter_length(test_input) == expected


DUMMY_LENGTH_CALCULATOR_TESTS_TABLE = [
    (val, len(val)) for val in LENGTH_CALCULATOR_SAMPLE_TEXTS
]


@pytest.mark.parametrize("test_input,expected", DUMMY_LENGTH_CALCULATOR_TESTS_TABLE)
def test_dummy_text_splitter_length_calculator(test_input, expected):
    """
    Dummy length calculator just returns the length of text.
    """
    length_calculator = DummyLengthCalculator()
    assert length_calculator.get_splitter_length(test_input) == expected
