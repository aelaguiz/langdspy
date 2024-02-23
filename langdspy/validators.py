import json
import logging


logger = logging.getLogger(__name__)


def is_json_list(input, output_val, kwargs) -> bool:
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

def is_one_of(input, output_val, kwargs) -> bool:
    if not kwargs.get('choices'):
        raise ValueError("is_one_of validator requires 'choices' keyword argument")

    try:
        if not kwargs.get('case_sensitive', False):
            choices = [c.lower() for c in kwargs['choices']]
            output_val = output_val.lower()

        logger.debug(f"Checking if {output_val} is one of {choices}")
        for choice in choices:
            if output_val.startswith(choice):
                logger.debug(f"Matched {output_val} to {choice}")
                return True

        return False
    except Exception as e:
        logger.error(f"Field must be one of {choices}, not {output_val}")
        return False