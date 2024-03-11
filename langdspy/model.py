from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
import time
import dill as pickle
import random
from joblib import Parallel, delayed
from tqdm import tqdm
import itertools
import re
from langchain_core.runnables import RunnableSerializable
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator, Extra, PrivateAttr
from langchain_core.pydantic_v1 import validator
from typing import Any, Dict, List, Type, Optional, Callable
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import uuid
from sklearn.base import BaseEstimator, ClassifierMixin
import threading

from .prompt_runners import PromptRunner



from langchain_core.runnables.utils import (
    Input,
    Output
)
from langchain_core.runnables.config import (
    RunnableConfig
)

import logging

logger = logging.getLogger("langdspy")

class TrainedModelState(BaseModel):
    examples: Optional[List[Any]] = None

    def __init__(self, **kwargs):
        super().__init__(**kwargs)  # Initialize BaseModel with kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)  # Dynamically assign attributes

class Model(RunnableSerializable, BaseEstimator, ClassifierMixin):
    prompt_runners = []
    kwargs = []
    trained_state: TrainedModelState = TrainedModelState()
    n_jobs: int = 1
    
    def __init__(self, n_jobs=1, **kwargs):
        super().__init__()
        self.n_jobs = n_jobs
        self.kwargs = {**kwargs, 'trained_state': self.trained_state}
        for field_name, field in self.__fields__.items():
            if issubclass(field.type_, PromptRunner):
                self.prompt_runners.append((field_name, field.default))

                field.default.set_model_kwargs(self.kwargs)
                # Necessary since pydantic creates a new version of the object
                setattr(self, field_name, field.default)

    
    def save(self, filepath):
        with open(filepath, 'wb') as file:
            pickle.dump(self.trained_state, file)
    
    def load(self, filepath):
        with open(filepath, 'rb') as file:
            self.trained_state = pickle.load(file)
    

    def predict(self, X, llm):
        y = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(self.invoke)(item, {**self.kwargs, 'trained_state': self.trained_state, 'llm': llm})
            for item in tqdm(X, desc="Predicting", total=len(X))
        )
        return y

    def fit(self, X, y, score_func, llm, n_examples=3, example_ratio=0.7, n_iter=None):
        # Split the data into example selection set and scoring set
        example_size = int(len(X) * example_ratio)
        example_indices = random.sample(range(len(X)), example_size)
        scoring_indices = [i for i in range(len(X)) if i not in example_indices]
        
        example_X = [X[i] for i in example_indices]
        example_y = [y[i] for i in example_indices]
        scoring_X = [X[i] for i in scoring_indices]
        scoring_y = [y[i] for i in scoring_indices]
        
        best_score = 0
        best_subset = []
        
        logger.debug(f"Total number of examples: {n_examples} Example size: {example_size} n_examples: {n_examples} example_X size: {len(example_X)} Scoring size: {len(scoring_X)}")
        
        def evaluate_subset(subset):
            subset_X, subset_y = zip(*subset)
            self.trained_state.examples = subset
            
            # Predict on the scoring set
            predicted_slugs = Parallel(n_jobs=self.n_jobs)(
                delayed(self.invoke)(item, config={
                    **self.kwargs,
                    'trained_state': self.trained_state,
                    'llm': llm
                })
                for item in scoring_X
            )
            score = score_func(scoring_y, predicted_slugs)
            logger.debug(f"Training subset scored {score}")
            return score, subset
        
        # Generate all possible subsets
        all_subsets = list(itertools.combinations(zip(example_X, example_y), n_examples))
        
        # Randomize the order of subsets
        random.shuffle(all_subsets)
        
        # Limit the number of iterations if n_iter is specified
        if n_iter is not None:
            all_subsets = all_subsets[:n_iter]
        
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            delayed(evaluate_subset)(subset)
            for subset in tqdm(all_subsets, desc="Evaluating subsets", total=len(all_subsets))
        )
        
        best_score, best_subset = max(results, key=lambda x: x[0])
        logger.debug(f"Best score: {best_score} with subset: {best_subset}")
        
        self.trained_state.examples = best_subset
        return self