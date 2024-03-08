# tests/test_generate_slugs.py
import sys
sys.path.append('.')
sys.path.append('langdspy')
import os
import dotenv
dotenv.load_dotenv()
import pytest
from unittest.mock import MagicMock
from examples.amazon.generate_slugs import ProductSlugGenerator, slug_similarity, get_llm

@pytest.fixture
def model():
    return ProductSlugGenerator(n_jobs=1, print_prompt=False)

@pytest.fixture
def llm():
    return get_llm()

@pytest.fixture
def dataset():
    return {
        'train': {
            'X': [
                {'h1': 'Product 1', 'title': 'Title 1', 'product_copy': 'Description 1'},
                {'h1': 'Product 2', 'title': 'Title 2', 'product_copy': 'Description 2'}
            ],
            'y': ['product-1', 'product-2']
        },
        'test': {
            'X': [
                {'h1': 'Product 3', 'title': 'Title 3', 'product_copy': 'Description 3'},
                {'h1': 'Product 4', 'title': 'Title 4', 'product_copy': 'Description 4'}
            ],
            'y': ['product-3', 'product-4']
        }
    }

def test_invoke_untrained(model, llm, dataset):
    input_data = dataset['test']['X'][0]
    result = model.invoke(input_data, config={'llm': llm})
    assert isinstance(result, str)
    assert len(result) <= 50

def test_invoke_trained(model, llm, dataset):
    model.fit(dataset['train']['X'], dataset['train']['y'], score_func=slug_similarity, llm=llm, n_examples=1, n_iter=1)
    input_data = dataset['test']['X'][0]
    result = model.invoke(input_data, config={'llm': llm})
    assert isinstance(result, str)
    assert len(result) <= 50

def test_predict_untrained(model, llm, dataset):
    X_test = dataset['test']['X']
    y_test = dataset['test']['y']
    predicted_slugs = model.predict(X_test, llm)
    assert len(predicted_slugs) == len(y_test)
    for slug in predicted_slugs:
        assert isinstance(slug, str)
        assert len(slug) <= 50

def test_predict_trained(model, llm, dataset):
    model.fit(dataset['train']['X'], dataset['train']['y'], score_func=slug_similarity, llm=llm, n_examples=1, n_iter=1)
    X_test = dataset['test']['X']
    y_test = dataset['test']['y']
    predicted_slugs = model.predict(X_test, llm)
    assert len(predicted_slugs) == len(y_test)
    for slug in predicted_slugs:
        assert isinstance(slug, str)
        assert len(slug) <= 50

def test_fit(model, llm, dataset):
    X_train = dataset['train']['X']
    y_train = dataset['train']['y']
    model.fit(X_train, y_train, score_func=slug_similarity, llm=llm, n_examples=1, n_iter=1)
    assert model.trained_state.examples is not None
    assert len(model.trained_state.examples) == 1