from langchain.prompts import BasePromptTemplate  # Assuming this is the correct import path
from langchain.prompts import FewShotPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, create_model, root_validator
from typing import Any, Dict, List, Type

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

class langdspySignature(BasePromptTemplate, BaseModel):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._collect_input_output_fields()

    @classmethod
    def _collect_input_output_fields(cls):
        cls._inputs = {}
        cls._outputs = {}
        for name, attribute in cls.__dict__.items():
            if isinstance(attribute, InputField):
                cls._inputs[name] = attribute.desc
            elif isinstance(attribute, OutputField):
                cls._outputs[name] = attribute.desc

    def format_prompt(self, **kwargs: Any) -> str:
        prompt = ""
        for input_name in self.input_variables:
            prompt += f"{input_name}: {kwargs.get(input_name, 'N/A')}\n"
        return prompt

    @classmethod
    def get_input_schema(cls) -> Type[BaseModel]:
        fields = {name: (str, Field(..., description=desc)) for name, desc in cls._inputs.items()}
        return create_model(f'{cls.__name__}InputSchema', **fields)

    def format(self, **kwargs: Any) -> str:
        return self.format_prompt(**kwargs)


class GenerateAbsoluteHandStrength(langdspySignature):
    player_hole_cards = InputField(desc="Description for player hole cards")
    board_cards = InputField(desc="Description for board cards")
    street = InputField(desc="Description for street")
    strength = OutputField(desc="Description for absolute hand strength")


# FewShotPromptTemplate()
model = GenerateAbsoluteHandStrength(input_variables=["player_hole_cards", "board_cards", "street"], output_variables=["strength"])
print(model.__dict__)