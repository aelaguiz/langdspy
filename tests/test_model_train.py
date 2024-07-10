# tests/test_generate_slugs.py
import sys
sys.path.append('.')
sys.path.append('langdspy')
import os
import dotenv
dotenv.load_dotenv()
import pytest
from unittest.mock import MagicMock
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Configure logging for langdspy
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
langdspy_logger = logging.getLogger('langdspy')
langdspy_logger.setLevel(logging.DEBUG)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
langdspy_logger.addHandler(console_handler)

import langdspy

# Configure logging for this module
logger = logging.getLogger(__name__)


class GenerateSlug(langdspy.PromptSignature):
    hint_slug = langdspy.HintField(desc="Generate a URL-friendly slug based on the provided H1, title, and product copy. The slug should be lowercase, use hyphens to separate words, and not exceed 50 characters.")
    
    h1 = langdspy.InputField(name="H1", desc="The H1 heading of the product page")
    title = langdspy.InputField(name="Title", desc="The title of the product page")
    product_copy = langdspy.InputField(name="Product Copy", desc="The product description or copy")
    
    slug = langdspy.OutputField(name="Slug", desc="The generated URL-friendly slug")

class ProductSlugGenerator(langdspy.Model):
    generate_slug = langdspy.PromptRunner(template_class=GenerateSlug, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input_dict, config):
        h1 = input_dict['h1']
        title = input_dict['title']
        product_copy = input_dict['product_copy']
        
        slug_res = self.generate_slug.invoke({'h1': h1, 'title': title, 'product_copy': product_copy}, config=config)
        
        return slug_res.slug


def cosine_similarity_tfidf(true_slugs, predicted_slugs):
    # Convert slugs to lowercase
    true_slugs = [slug.lower() for slug in true_slugs]
    predicted_slugs = [slug.lower() for slug in predicted_slugs]

    # for i in range(len(true_slugs)):
    #     print(f"Actual Slug: {true_slugs[i]} Predicted: {predicted_slugs[i]}")

    vectorizer = TfidfVectorizer()
    true_vectors = vectorizer.fit_transform(true_slugs)
    predicted_vectors = vectorizer.transform(predicted_slugs)
    similarity_scores = cosine_similarity(true_vectors, predicted_vectors)
    return similarity_scores.diagonal()

def slug_similarity(X, true_slugs, predicted_slugs):
    similarity_scores = cosine_similarity_tfidf(true_slugs, predicted_slugs)
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return average_similarity

@pytest.fixture
def model():
    return ProductSlugGenerator(n_jobs=1, print_prompt=False)

@pytest.fixture
def llm():
    return FakeLLM()

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


from langchain.chat_models.base import BaseChatModel

class FakeLLM(BaseChatModel):
    def invoke(self, *args, **kwargs):
        return "INVOKED"

    def _generate(self, *args, **kwargs):
        return None

    def _llm_type(self) -> str:
        return "fake"

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
