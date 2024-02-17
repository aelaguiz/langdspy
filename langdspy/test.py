from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from typing import Any, Dict, List, Type
from abc import ABC, abstractmethod

# class FieldDescriptor:
#     def __init__(self, desc: str):
#         self.desc = desc

#     def __set_name__(self, owner, name):
#         self.private_name = f"_{name}"

#     def __get__(self, obj, objtype=None):
#         return getattr(obj, self.private_name, None)

#     def __set__(self, obj, value):
#         setattr(obj, self.private_name, value)

# class InputField(FieldDescriptor):
#     pass

# class OutputField(FieldDescriptor):
#     pass

# class langdspySignature(BasePromptTemplate, BaseModel):
#     def __init_subclass__(cls, **kwargs):
#         super().__init_subclass__(**kwargs)
#         cls._collect_input_output_fields()

#     @classmethod
#     def _collect_input_output_fields(cls):
#         cls._inputs = {}
#         cls._outputs = {}
#         for name, attribute in cls.__dict__.items():
#             if isinstance(attribute, InputField):
#                 cls._inputs[name] = attribute.desc
#             elif isinstance(attribute, OutputField):
#                 cls._outputs[name] = attribute.desc

#     def format_prompt(self, **kwargs: Any) -> str:
#         prompt = ""
#         for input_name in self.input_variables:
#             prompt += f"{input_name}: {kwargs.get(input_name, 'N/A')}\n"
#         return prompt

#     @classmethod
#     def get_input_schema(cls) -> Type[BaseModel]:
#         fields = {name: (str, Field(..., description=desc)) for name, desc in cls._inputs.items()}
#         return create_model(f'{cls.__name__}InputSchema', **fields)

#     def format(self, **kwargs: Any) -> str:
#         return self.format_prompt(**kwargs)


# class GenerateAbsoluteHandStrength(langdspySignature):

#     input_variables: List[str]
#     """A list of the names of the variables the prompt template expects."""

#     player_hole_cards = InputField(desc="Description for player hole cards")
#     board_cards = InputField(desc="Description for board cards")
#     street = InputField(desc="Description for street")
#     strength = OutputField(desc="Description for absolute hand strength")


class GenerateAbsoluteHandStrength(PromptTemplate):
    """String prompt that exposes the format method, returning a prompt."""
    pass

    # @classmethod
    # def get_lc_namespace(cls) -> List[str]:
    #     """Get the namespace of the langchain object."""
    #     return ["langchain", "prompts", "base"]

    # def format_prompt(self, **kwargs: Any):
    #     return ""

    # def pretty_repr(self, html: bool = False) -> str:
    #     return ""

    # def pretty_print(self) -> None:
    #     print("HI")

# FewShotPromptTemplate()
# model = GenerateAbsoluteHandStrength()
# print(model.__dict__)

from langchain_core.load import Serializable
from langchain_core.runnables import RunnableSerializable

class Parent(BasePromptTemplate):
    parent_field: str

    def invoke():
        pass

    def format():
        pass

    def format_prompt():
        pass

# Child class inheriting from Parent
class Child(Parent):
    child_field: int

# Creating an instance of Child with both parent and child fields
child_instance = Child(parent_field='parent value', child_field=123, input_variables=['input1', 'input2'])

print(child_instance)
