import langdspy
import dotenv
import os
import logging
import logging.config

dotenv.load_dotenv()

config_path = os.getenv('LOGGING_CONF_PATH')

# Use the configuration file appropriate to the environment
logging.config.fileConfig(config_path)
logging.getLogger("httpx").setLevel(logging.DEBUG)
logging.getLogger("openai").setLevel(logging.DEBUG)
logging.getLogger("httpcore.connection").setLevel(logging.DEBUG)
logging.getLogger("httpcore.http11").setLevel(logging.DEBUG)
logging.getLogger("openai._base_client").setLevel(logging.DEBUG)

from typing import Any, Dict, List, Type, Optional
from langchain_core.runnables.utils import (
    Input,
    Output
)
from langchain_core.runnables.config import (
    RunnableConfig
)

from langchain_openai import ChatOpenAI

llm = ChatOpenAI(api_key=os.getenv("OPENAI_API_KEY"), model=os.getenv("FAST_OPENAI_MODEL"))

class GenerateAbsoluteHandStrength(langdspy.PromptSignature):
    player_hole_cards = langdspy.InputField(name="Player Hole Cards", desc="Description for player hole cards")
    board_cards = langdspy.InputField(name="Board Cards", desc="Description for board cards")
    street = langdspy.InputField(name="Street", desc="Description for street")
    strength = langdspy.OutputField(name="Hand Strength", desc="Description for absolute hand strength")


class HoleCardsCommentary(langdspy.Model):
    absolute_hand_strength = langdspy.PromptRunner(template_class=GenerateAbsoluteHandStrength, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        print(f"Running HoleCardsCommentary with input: {input}")
        res = self.absolute_hand_strength.invoke(input, config)
        return res


model = HoleCardsCommentary()
res = model.invoke({
    'player_hole_cards': 'Ah Kh',
    'board_cards': '2h 3h 4h',
    'street': 'flop'
}, config={'llm': llm})

print(f"Result: {res}")