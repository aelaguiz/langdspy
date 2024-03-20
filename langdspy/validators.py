import json
import logging


logger = logging.getLogger("langdspy")


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

    none_ok = False
    if kwargs.get('none_ok', False):
        none_ok = True

    if none_ok and output_val.lower().startswith("none"):
        logger.debug(f"None is okay and got none")
        return True

    try:
        if not kwargs.get('case_sensitive', False):
            choices = [c.lower().replace("_", " ") for c in kwargs['choices']]
            output_val = output_val.lower().replace("_", " ")

        # logger.debug(f"Checking if {output_val} is one of {choices}")
        for choice in choices:
            if output_val.startswith(choice):
                # logger.debug(f"Matched {output_val} to {choice}")
                return True

        return False
    except Exception as e:
        logger.error(f"Field must be one of {kwargs.get('choices')}, not {output_val}")
        import traceback
        traceback.print_exc()
        return False

def is_subset_of(input, output_val, kwargs) -> bool:
    if not kwargs.get('choices'):
        raise ValueError("is_subset_of validator requires 'choices' keyword argument")
    
    none_ok = kwargs.get('none_ok', False)
    if none_ok and output_val.lower().strip() == "none":
        return True
    
    try:
        values = [v.strip() for v in output_val.split(",")]
        if not kwargs.get('case_sensitive', False):
            choices = [c.lower().replace("_", " ") for c in kwargs['choices']]
            values = [v.lower().replace("_", " ") for v in values]
        for value in values:
            if value not in choices:
                return False
        return True
    except Exception as e:
        logger.error(f"Field must be a comma-separated list of one or more of {kwargs.get('choices')}, not {output_val}")
        return False