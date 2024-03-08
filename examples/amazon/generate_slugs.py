import sys
sys.path.append('.')
sys.path.append('langdspy')

import os
import dotenv
dotenv.load_dotenv()

import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logging.getLogger("httpx").disabled = True
logging.getLogger("openai").disabled = True
logging.getLogger("httpcore.connection").disabled = True
logging.getLogger("httpcore.http11").disabled = True
logging.getLogger("openai._base_client").disabled = True
logging.getLogger("paramiko.transport").disabled = True
logging.getLogger("anthropic._base_client").disabled = True
# logging.getLogger("langdspy").disabled = True

import langdspy
import httpx
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_anthropic import ChatAnthropic
from sklearn.metrics import accuracy_score
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def get_llm():
    FAST_OPENAI_MODEL = os.getenv("FAST_OPENAI_MODEL")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
    OPENAI_TEMPERATURE = os.getenv("OPENAI_TEMPERATURE")
    FAST_MODEL_PROVIDER = os.getenv("FAST_MODEL_PROVIDER")
    FAST_ANTHROPIC_MODEL = os.getenv("FAST_ANTHROPIC_MODEL")

    if FAST_MODEL_PROVIDER.lower() == "anthropic":
        _fast_llm = ChatAnthropic(model_name=FAST_ANTHROPIC_MODEL, temperature=OPENAI_TEMPERATURE, anthropic_api_key=ANTHROPIC_API_KEY)
    else:
        _fast_llm = ChatOpenAI(model_name=FAST_OPENAI_MODEL, temperature=OPENAI_TEMPERATURE, timeout=httpx.Timeout(15.0, read=60.0, write=10.0, connect=3.0), max_retries=2)

    return _fast_llm


class GenerateSlug(langdspy.PromptSignature):
    hint_slug = langdspy.HintField(desc="Generate a URL-friendly slug based on the provided H1, title, and product copy. The slug should be lowercase, use hyphens to separate words, and not exceed 50 characters.")
    
    h1 = langdspy.InputField(name="H1", desc="The H1 heading of the product page")
    title = langdspy.InputField(name="Title", desc="The title of the product page")
    product_copy = langdspy.InputField(name="Product Copy", desc="The product description or copy")
    
    slug = langdspy.OutputField(name="Slug", desc="The generated URL-friendly slug")

class ProductSlugGenerator(langdspy.Model):
    generate_slug = langdspy.PromptRunner(template_class=GenerateSlug, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input, config):
        h1 = input['h1']
        title = input['title']
        product_copy = input['product_copy']
        
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

def slug_similarity(true_slugs, predicted_slugs):
    similarity_scores = cosine_similarity_tfidf(true_slugs, predicted_slugs)
    average_similarity = sum(similarity_scores) / len(similarity_scores)
    return average_similarity

def evaluate_model(model, X, y):
    predicted_slugs = model.predict(X, llm)
    accuracy = slug_similarity(y, predicted_slugs)
    return accuracy

llm = get_llm()

if __name__ == "__main__":
    output_path = sys.argv[1]
    dataset_file= "data/amazon_products_split.json"
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)
    
    X_train = dataset['train']['X']
    y_train = dataset['train']['y']
    X_test = dataset['test']['X']
    y_test = dataset['test']['y']
    
    model = ProductSlugGenerator(n_jobs=4, print_prompt=True)

    before_test_accuracy = None
    if os.path.exists(output_path):
        model.load(output_path)
    else:
        input("Hit enter to evaluate the untrained model...")
        before_test_accuracy = evaluate_model(model, X_test, y_test)
        print(f"Before Training Accuracy: {before_test_accuracy}")
        
        input("Hit enter to train the model...")
        model.fit(X_train, y_train, score_func=slug_similarity, llm=llm, n_examples=2, n_iter=100)
        
    input("Hit enter to evaluate the trained model...")
    # Evaluate the model on the test set
    test_accuracy = evaluate_model(model, X_test, y_test)
    print(f"Before Training Accuracy: {before_test_accuracy}")
    print(f"After Training Accuracy: {test_accuracy}")

    model.save(output_path)