# Module console.py

print(f"The variable __name__ of the console.py module has taken the name of {__name__}")

import functions
from typing import Dict, Any


def fill_data(data: Dict[float, float]) -> None:
    """
    Prompt the user to input X and Y coordinates and add them to the data dictionary.

    Args:
        data (Dict[float, float]): A dictionary to store X as keys and Y as values.

    Returns:
        None
    """
    try:
        x_input = float(input("X -> "))
        y_input = float(input("Y -> "))
        print("\n")
        data[x_input] = y_input
    except ValueError:
        print("Invalid input. Only numbers are allowed.\n")


def perform_regression(data: Dict[float, float]) -> None:
    """
    Perform linear regression on the provided data and display initial and final parameters.

    Args:
        data (Dict[float, float]): A dictionary containing X and Y coordinates.

    Returns:
        None
    """
    data_items = list(data.items())
    parameters = {"intercept": 0.0, "slope": 1.0}
    
    initial_info = (
        f"\nInitial intercept: {parameters['intercept']}\n"
        f"Initial slope: {parameters['slope']}\n"
        f"Initial error: {functions.loss_function(data_items, parameters)}\n"
    )
    print(initial_info)
    
    functions.adjust_parameters(data_items, parameters)
    
    final_info = (
        f"\nFinal intercept: {round(parameters['intercept'], 6)}\n"
        f"Final slope: {round(parameters['slope'], 6)}\n"
        f"Final error: {functions.loss_function(data_items, parameters)}\n"
    )
    print(final_info)


def run() -> None:
    """
    Run the linear regression console application, allowing users to input data and perform regression.

    Returns:
        None
    """
    data: Dict[float, float] = {}
    print("\nWelcome to the Linear Regression Program")
    
    menu = """\n1. Enter coordinate
2. Finish
>>> """
    
    while True:
        option = input(menu)
        if option == "1":
            fill_data(data)
        elif option == "2":
            break
        else:
            print("Please enter a valid option.\n")
    
    if not data:
        print("No coordinates were added.")
    else:
        perform_regression(data)


if __name__ == "__main__":
    run()
