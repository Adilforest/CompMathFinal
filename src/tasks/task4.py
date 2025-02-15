import numpy as np
import matplotlib.pyplot as plt


def solve_task(matrix: np.ndarray, tol: float = 1e-10, max_iter: int = 1000, axes=None):
    """
    Implements the power method to find the eigenvalue of matrix A
    with the largest magnitude.

    Parameters:
      - A: square matrix (numpy.ndarray)
      - tol: tolerance (for convergence based on the change in eigenvalue)
      - max_iter: maximum number of iterations

    Returns:
      - lambda_approx: approximate largest eigenvalue
      - x: corresponding eigenvector (normalized)
      - eigenvalue_history: list of eigenvalue approximations per iteration
      - iter_numbers: list of iteration numbers (for plotting)
    """
    n = matrix.shape[0]
    # Initial approximation: an arbitrary vector (here, a vector of ones)
    x = np.ones(n)
    x = x / np.linalg.norm(x)  # normalize the vector
    eigenvalue_history = []
    iter_numbers = []
    relative_errors = []

    for i in range(max_iter):
        # Compute the product A * x
        y = matrix.dot(x)
        # Approximate the eigenvalue using the Rayleigh quotient:
        lambda_approx = np.dot(x, y)
        eigenvalue_history.append(lambda_approx)
        iter_numbers.append(i)

        # Calculate relative error (if not the first iteration)
        if i > 0:
            relative_error = abs(
                (eigenvalue_history[-1] - eigenvalue_history[-2])
                / eigenvalue_history[-2]
            )
            relative_errors.append(relative_error)
        else:
            relative_errors.append(None)
        # Normalize the vector y to obtain the next eigenvector approximation
        x_new = y / np.linalg.norm(y)

        # Check for convergence: if the difference between consecutive approximations is less than tol, exit
        if i > 0 and abs(eigenvalue_history[-1] - eigenvalue_history[-2]) < tol:
            break

        x = x_new

    if axes is None:
        fig, axes = plt.subplots()
    axes.plot(
        iter_numbers[1:], relative_errors[1:], marker="o", linestyle="-", color="b"
    )
    axes.set_xlabel("Iteration")
    axes.set_ylabel("Relative Error")
    axes.set_title("Convergence of Power Method (Relative Error)")
    plt.grid(True)

    return {
        "lambda_approx": lambda_approx,
        "eigenvector": x,
        "eigenvalue_history": eigenvalue_history,
        "iter_numbers": iter_numbers,
    }


def main():

    # If you want to use a test example, uncomment the following lines:
    # A = np.array([[6, 2, 3],
    #               [2, 6, 4],
    #               [3, 4, 6]])

    lambda_approx, eigenvector, eigenvalue_history, iter_numbers = power_method(None)


if __name__ == "__main__":
    main()
