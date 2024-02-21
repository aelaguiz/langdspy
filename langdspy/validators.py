import json
import logging


logger = logging.getLogger(__name__)


def is_json_list(input, output_val) -> bool:
    try:
        try:
            choices = json.loads(output_val)
            if not isinstance(choices, list):
                logger.error(f"Field must be a JSON list, not {output_val}")
                return False
        except json.JSONDecodeError:
            logger.error(f"Field must be a JSON list, not {output_val}")
            return False
        return True
    except Exception as e:
        logger.error(f"Field must be a JSON list, not {output_val}")
        return False