from langchain.tools import tool


@tool
def count_letters(text: str, character: str) -> int:
    """Counts how many times a specific letter appears in a given text."""
    return text.lower().count(character.lower())



@tool
def compare_numbers(num1: float, num2: float) -> str:
    """Compares two numbers and returns which is greater or if they are equal."""
    if num1 > num2:
        return f"{num1} is greater than {num2}."
    elif num2 > num1:
        return f"{num2} is greater than {num1}."
    else:
        return f"Both numbers are equal ({num1})."