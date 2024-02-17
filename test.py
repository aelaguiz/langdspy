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

logger = logging.getLogger(__name__)

class GenerateAbsoluteHandStrength(langdspy.PromptSignature):
    player_hole_cards = langdspy.InputField(name="Player Hole Cards", desc="The hole cards of the player.")
    board_cards = langdspy.InputField(name="Board Cards", desc="The cards on the board.")
    street = langdspy.InputField(name="Street", desc="The current street of the hand (one of: 'preflop', 'flop', 'turn', 'river')")
    strength = langdspy.OutputField(name="Hand Strength", desc="The absolute strength of the player's hand based on the hole cards and board cards (Premium Hands (Top 3%), Strong Hands (Top 5%), Good Hands (Top 10%), Playable Hands, Marginal Hands, Lower-Tier Pairs, Offsuit Connectors and Gappers, Suited Gappers, Speculative Hands, Pair, Two Pair, Top Pair, etc)")


class HoleCardsCommentary(langdspy.Model):
    absolute_hand_strength = langdspy.PromptRunner(template_class=GenerateAbsoluteHandStrength, prompt_strategy=langdspy.DefaultPromptStrategy)

    def invoke(self, input: Input, config: Optional[RunnableConfig] = None) -> Output:
        print(f"Running HoleCardsCommentary with input: {input}")
        res = self.absolute_hand_strength.invoke(input, config)
        return res


model = HoleCardsCommentary()
pred = model.invoke({
    'player_hole_cards': 'Ah Kh',
    'board_cards': '2h 3h 4h',
    'street': 'flop'
}, config={'llm': llm})

logger.info(f"Hand strength: {pred.strength}")