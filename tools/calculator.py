from langchain_core.tools import tool

@tool
def add(a: int, b: int) -> int:
    """
    Adds two integers together.
    
    Args:
        a (int): The first integer.
        b (int): The second integer.
        
    Returns:
        int: The sum of the two integers.
    """
    return a + b

@tool 
def subtract(a: int, b: int) -> int:
    """
    Subtracts the second integer from the first.
    
    Args:
        a (int): The first integer.
        b (int): The second integer.
        
    Returns:
        int: The result of a - b.
    """
    return a - b

@tool 
def multiply(a: int, b: int) -> int:
    """
    Multiplies two integers together.
    
    Args:
        a (int): The first integer.
        b (int): The second integer.
        
    Returns:
        int: The product of the two integers.
    """
    return a * b

@tool 
def divide(a: int, b: int) -> float:
    """
    Divides the first integer by the second.
    
    Args:
        a (int): The numerator.
        b (int): The denominator.
        
    Returns:
        float: The result of a / b.
        
    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b

if __name__ == "__main__":
    print(add.invoke(input={"a": 5, "b": 3}))        # 8
    print(subtract.invoke(input={"a": 5, "b": 3}))   # 2
    print(multiply.invoke(input={"a": 5, "b": 3}))   # 15
    print(divide.invoke(input={"a": 5, "b": 3}))     # 1.6666666666666667
