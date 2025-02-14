import numpy as np
import matplotlib.pyplot as plt

def power_method(A, tol=1e-10, max_iter=1000):
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
    n = A.shape[0]
    # Initial approximation: an arbitrary vector (here, a vector of ones)
    x = np.ones(n)
    x = x / np.linalg.norm(x)  # normalize the vector
    eigenvalue_history = []
    iter_numbers = []

    for i in range(max_iter):
        # Compute the product A * x
        y = A.dot(x)
        # Approximate the eigenvalue using the Rayleigh quotient:
        lambda_approx = np.dot(x, y)
        eigenvalue_history.append(lambda_approx)
        iter_numbers.append(i)

        # Normalize the vector y to obtain the next eigenvector approximation
        x_new = y / np.linalg.norm(y)

        # Check for convergence: if the difference between consecutive approximations is less than tol, exit
        if i > 0 and abs(eigenvalue_history[-1] - eigenvalue_history[-2]) < tol:
            break

        x = x_new

    return lambda_approx, x, eigenvalue_history, iter_numbers

def read_matrix():
    """
    Reads a 3x3 matrix from the user input.

    Returns:
      - A: 3x3 numpy.ndarray
    """
    print("Enter the elements of matrix A (3x3):")
    a11 = float(input("a11: "))
    a12 = float(input("a12: "))
    a13 = float(input("a13: "))
    a21 = float(input("a21: "))
    a22 = float(input("a22: "))
    a23 = float(input("a23: "))
    a31 = float(input("a31: "))
    a32 = float(input("a32: "))
    a33 = float(input("a33: "))

    A = np.array([[a11, a12, a13],
                  [a21, a22, a23],
                  [a31, a32, a33]])
    return A

def print_results(lambda_approx, eigenvector, eigenvalue_history, iter_numbers):
    """
    Prints the results of the power method.
    """
    print("\nResults of the Power Method:")
    print("Approximate largest eigenvalue:", lambda_approx)
    print("Corresponding eigenvector (normalized):")
    print(eigenvector)
    print("Number of iterations:", len(iter_numbers))

    print("\nConvergence Table (Iteration, Eigenvalue Approximation):")
    print("{:<10} {:<20}".format("Iteration", "Eigenvalue"))
    for it, val in zip(iter_numbers, eigenvalue_history):
        print("{:<10} {:<20.12f}".format(it, val))

def plot_convergence(iter_numbers, eigenvalue_history):
    """
    Plots the convergence of the eigenvalue approximations.
    """
    plt.figure(figsize=(8, 6))
    plt.plot(iter_numbers, eigenvalue_history, marker='o', linestyle='-', color='blue')
    plt.xlabel("Iteration Number")
    plt.ylabel("Eigenvalue Approximation")
    plt.title("Convergence of the Power Method")
    plt.grid(True)
    plt.show()

def main():
    A = read_matrix()

    # If you want to use a test example, uncomment the following lines:
    # A = np.array([[6, 2, 3],
    #               [2, 6, 4],
    #               [3, 4, 6]])

    lambda_approx, eigenvector, eigenvalue_history, iter_numbers = power_method(A)
    print_results(lambda_approx, eigenvector, eigenvalue_history, iter_numbers)
    plot_convergence(iter_numbers, eigenvalue_history)

if __name__ == "__main__":
    main()
