# Linear Regression Console Application

This repository contains a simple linear regression program implemented in Python. The program uses basic gradient descent to fit a line to a set of user-provided data points. The user can input coordinates and perform linear regression through a simple command-line interface.

## Features

- Allows users to input coordinates (X, Y) for linear regression.
- Performs linear regression using gradient descent to find the optimal intercept and slope.
- Displays both the initial and final intercept, slope, and error values.
- Handles user input validation, ensuring that only numeric values are accepted for X and Y coordinates.

## Usage

To run the linear regression program, simply execute the following command in the project directory:

```bash
python console.py
```

## Code Explanation

### `functions.py`

The `functions.py` module contains functions related to the core linear regression logic:

- **`partial_derivative`**: Calculates the partial derivative of the loss function with respect to intercept or slope.
- **`lineal_function`**: Calculates the predicted value of Y given X, intercept, and slope.
- **`loss_function`**: Computes the total loss (sum of squared errors) for a given dataset and parameters.
- **`gradients`**: Computes the gradients (derivatives) of the loss function with respect to the intercept and slope.
- **`adjust_parameters`**: Adjusts the intercept and slope using gradient descent to minimize the loss function.

### `console.py`

The `console.py` module provides a simple command-line interface for the user to input coordinates and run the linear regression:

- **`fill_data`**: Prompts the user for X and Y values, which are added to a dictionary.
- **`perform_regression`**: Performs linear regression on the input data and displays the initial and final intercept, slope, and error.
- **`run`**: The main driver function that displays the menu, accepts user input, and calls other functions.
