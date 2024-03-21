def normalize_enum_value(val: str) -> str:
    return val.replace(" ", "_").replace("-", "_").upper()
