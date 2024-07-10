# langdspy: Langchain implementation of stanford DSPy

This is intended to be a langchain native implementation of the innovative principles pioneered by the wonderful [DSPy](https://github.com/stanfordnlp/dspy)

I believe that the principles they have pioneered should be built on top of more mature primitives with a larger ecosystem & community to support it.

This is intended to be usable in production enterprise environments.

Designed to be interface compatible with both langchain and scikit-learn.

## Example in action

Automatically tunning n-shot prompts based on provided samples. See examples/amazon/generate_slugs.py

```python
if __name__ == "__main__":
    output_path = sys.argv[1]
    dataset_file= "data/amazon_products_split.json"
    with open(dataset_file, 'r') as file:
        dataset = json.load(file)
    
    X_train = dataset['train']['X']
    y_train = dataset['train']['y']
    X_test = dataset['test']['X']
    y_test = dataset['test']['y']
    
    model = ProductSlugGenerator(n_jobs=4)

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
```

## Quick start

```bash
git clone https://github.com/aelaguiz/langdspy
cd langdspy
pip install poetry
poetry install
poetry run test
```

### Roadmap

* Unit tests
* Auto-Tuning with RAG steps
* New prmompt strategies (e.g. Chain Of Thought)
* Tighter integration with Langchain templates (can we use their few-shot prompting templates for example?)

## Author

Amir Elaguizy is the original author but sincerely hopes to be the smallest piece of this going forward.

Use it as you wish.