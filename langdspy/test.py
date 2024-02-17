from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
from langchain.prompts import FewShotPromptTemplate
from langchain_core.runnables import RunnableSerializable
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from langchain_core.pydantic_v1 import validator
from typing import Any, Dict, List, Type, Optional
from abc import ABC, abstractmethod
from langchain_core.runnables.utils import (
    Input,
    Output
)
from langchain_core.runnables.config import (
    RunnableConfig
)

class FieldDescriptor:
    def __init__(self, desc: str):
        self.desc = desc

    def __set_name__(self, owner, name):
        self.private_name = f"_{name}"

    def __get__(self, obj, objtype=None):
        return getattr(obj, self.private_name, None)

    def __set__(self, obj, value):
        setattr(obj, self.private_name, value)

class InputField(FieldDescriptor):
    pass

class OutputField(FieldDescriptor):
    pass

class langdspyPromptTemplate(BasePromptTemplate, BaseModel):
    # Assuming input_variables and output_variables are defined as class attributes
    input_variables: List[str] = []
    output_variables: List[str] = []

    def __init__(self, **kwargs):
        print("Init")
        print(dir(self.__class__))
        print("Calling super")
        # Temporarily bypassing the setting of input and output variables
        super().__init__(**kwargs)
        print("Done super")
        print(self.__class__.__fields__)

        inputs = {}
        outputs = {}

        for name, attribute in self.__class__.__fields__.items():
            print(f"field: {name}, value: {attribute}")

            if attribute.type_ == InputField:
                inputs[name] = attribute
            elif attribute.type_ == OutputField:
                outputs[name] = attribute

        self.input_variables = list(inputs.keys())
        self.output_variables = list(outputs.keys())

        print(f"Inputs: {self.input_variables}")
        print(f"Outputs: {self.output_variables}")


    def format_prompt(self, **kwargs: Any) -> str:
        prompt = ""
        for input_name in self.input_variables:
            # print(f"input_name: {input_name}")
            prompt += f"{input_name}: {kwargs.get(input_name, 'N/A')}\n"
        return prompt

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)

        
class langdspyPredict(RunnableSerializable):
    template: langdspyPromptTemplate = None
    
    @validator("template")
    def check_template(
        cls, value: langdspyPromptTemplate
    ) -> langdspyPromptTemplate:
        print(f"Checking template: {value}")
        return value
    
    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        pass


class GenerateAbsoluteHandStrength(langdspyPromptTemplate):
    player_hole_cards = InputField(desc="Description for player hole cards")
    board_cards = InputField(desc="Description for board cards")
    street = InputField(desc="Description for street")
    strength = OutputField(desc="Description for absolute hand strength")


class HoleCardsCommentary(langdspyPredict):
    absolute_hand_strength = langdspyPredict(template=GenerateAbsoluteHandStrength())
    # def __init__(self):
    #     self.absolute_hand_strength = langdspyPredict(template="hi")


# FewShotPromptTemplate()
model = HoleCardsCommentary()
print(model.absolute_hand_strength)
# print(model.format())