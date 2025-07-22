from langchain.tools import tool
from pydantic import BaseModel, Field

# --- Tool for counting letters ---

# To ensure the agent knows exactly what inputs are needed for functions
# with multiple arguments, we can define a Pydantic model.
class CountLettersInput(BaseModel):
    """Input schema for the count_letters tool."""
    text: str = Field(description="The text to search within.")
    letter: str = Field(description="The letter to count.")

@tool(args_schema=CountLettersInput)
def count_letters(text: str, letter: str) -> int:
    """Counts how many times a specific letter appears in a given text."""
    return text.lower().count(letter.lower())


# --- Tool for comparing numbers ---

class CompareNumbersInput(BaseModel):
    """Input schema for the compare_numbers tool."""
    num1: float = Field(description="The first number for comparison.")
    num2: float = Field(description="The second number for comparison.")

@tool(args_schema=CompareNumbersInput)
def compare_numbers(num1: float, num2: float) -> str:
    """Compares two numbers and returns which is greater or if they are equal."""
    if num1 > num2:
        return f"{num1} is greater than {num2}."
    elif num2 > num1:
        return f"{num2} is greater than {num1}."
    else:
        return f"Both numbers are equal ({num1})."