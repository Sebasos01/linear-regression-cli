# Module functions.py

print(f"The variable __name__ of the functions.py module has taken the name of {__name__}")

from typing import List, Tuple, Dict, Optional

def partial_derivative(intercept: Tuple[float, bool], slope: Tuple[float, bool], y: float, x: float) -> Optional[float]:
    """
    Calculate the partial derivative of the loss function with respect to intercept or slope.

    Args:
        intercept (Tuple[float, bool]): A tuple where the first element is the intercept value and the second is a boolean indicating differentiation.
        slope (Tuple[float, bool]): A tuple where the first element is the slope value and the second is a boolean indicating differentiation.
        y (float): The actual y value.
        x (float): The x value.

    Returns:
        Optional[float]: The partial derivative with respect to intercept or slope, or None if neither.
    """
    residual = y - intercept[0] - (slope[0] * x)
    if intercept[1]:
        derivative = -2 * residual
    elif slope[1]:
        derivative = -2 * x * residual
    else:
        derivative = None
    return derivative

def lineal_function(intercept: float, slope: float, x: float) -> float:
    """
    Compute the linear function value for given intercept, slope, and x.

    Args:
        intercept (float): The intercept of the line.
        slope (float): The slope of the line.
        x (float): The x value.

    Returns:
        float: The predicted y value.
    """
    return intercept + slope * x

def loss_function(data: List[Tuple[float, float]], parameters: Dict[str, float]) -> float:
    """
    Calculate the sum of squared errors loss.

    Args:
        data (List[Tuple[float, float]]): A list of (x, y) tuples.
        parameters (Dict[str, float]): A dictionary containing 'intercept' and 'slope'.

    Returns:
        float: The total loss.
    """
    intercept = parameters["intercept"]
    slope = parameters["slope"]
    return sum((y - lineal_function(intercept, slope, x)) ** 2 for x, y in data)

def gradients(data: List[Tuple[float, float]], intercept: float, slope: float) -> Dict[str, float]:
    """
    Compute the gradients of the loss function with respect to intercept and slope.

    Args:
        data (List[Tuple[float, float]]): A list of (x, y) tuples.
        intercept (float): The current intercept value.
        slope (float): The current slope value.

    Returns:
        Dict[str, float]: A dictionary with keys 'dl/di' and 'dl/ds' representing gradients.
    """
    gradient = {"dl/di": 0.0, "dl/ds": 0.0}
    for x, y in data:
        dl_di = partial_derivative((intercept, True), (slope, False), y, x)
        dl_ds = partial_derivative((intercept, False), (slope, True), y, x)
        if dl_di is not None:
            gradient["dl/di"] += dl_di
        if dl_ds is not None:
            gradient["dl/ds"] += dl_ds
    return gradient

def adjust_parameters(data: List[Tuple[float, float]], parameters: Dict[str, float]) -> None:
    """
    Adjust the parameters using gradient descent.

    Args:
        data (List[Tuple[float, float]]): A list of (x, y) tuples.
        parameters (Dict[str, float]): A dictionary containing 'intercept' and 'slope' to be updated.

    Returns:
        None
    """
    steps = 0
    learning_rate = 0.001
    max_steps = 100_000 

    for _ in range(max_steps):
        gradient = gradients(data, parameters["intercept"], parameters["slope"])
        intercept_nudge = gradient["dl/di"] * learning_rate
        slope_nudge = gradient["dl/ds"] * learning_rate
        parameters["intercept"] -= intercept_nudge
        parameters["slope"] -= slope_nudge
        steps += 1

    print(f"The number of steps was {steps}")
