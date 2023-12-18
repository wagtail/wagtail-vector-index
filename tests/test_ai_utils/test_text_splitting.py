import pytest
from wagtail_vector_index.ai_utils.backends import get_chat_backend
from wagtail_vector_index.ai_utils.text_splitting.dummy import (
    DummyLengthCalculator,
    DummyTextSplitter,
)
from wagtail_vector_index.ai_utils.text_splitting.langchain import (
    LangchainRecursiveCharacterTextSplitter,
)
from wagtail_vector_index.ai_utils.text_splitting.naive import (
    NaiveTextSplitterCalculator,
)


def test_default_text_splitter():
    chat_backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
            },
        },
        backend_id="default",
    )
    text_splitter = chat_backend.get_text_splitter()
    assert isinstance(text_splitter, LangchainRecursiveCharacterTextSplitter)


def test_default_length_calculator():
    chat_backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
            },
        },
        backend_id="default",
    )
    length_calculator = chat_backend.get_splitter_length_calculator()
    assert isinstance(length_calculator, NaiveTextSplitterCalculator)


def test_custom_text_splitter():
    chat_backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
            },
            "TEXT_SPLITTING": {
                "SPLITTER_CLASS": "wagtail_vector_index.ai_utils.text_splitting.dummy.DummyTextSplitter"
            },
        },
        backend_id="default",
    )

    text_splitter = chat_backend.get_text_splitter()
    assert isinstance(text_splitter, DummyTextSplitter)


def test_custom_length_calculator():
    chat_backend = get_chat_backend(
        backend_dict={
            "CLASS": "wagtail_vector_index.ai_utils.backends.echo.EchoChatBackend",
            "CONFIG": {
                "MODEL_ID": "echo",
                "TOKEN_LIMIT": 1024,
            },
            "TEXT_SPLITTING": {
                "SPLITTER_LENGTH_CALCULATOR_CLASS": "wagtail_vector_index.ai_utils.text_splitting.dummy.DummyLengthCalculator"
            },
        },
        backend_id="default",
    )
    length_calculator = chat_backend.get_splitter_length_calculator()
    assert isinstance(length_calculator, DummyLengthCalculator)


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
