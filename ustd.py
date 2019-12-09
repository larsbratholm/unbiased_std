import numpy as np
import math

def get_scaling_factor(n):
    k = n // 2
    if n % 2 == 0:
        scaling_factor = np.sqrt(2 / (np.pi * (2 * k - 1))) \
                * (2 ** (2 * k - 2) * math.factorial(k-1)**2) / math.factorial(2 * k - 2)
    else:
        scaling_factor = np.sqrt(np.pi / k) * math.factorial(2 * k - 1) / \
                (2 ** (2 * k - 1) * math.factorial(k - 1)**2)
    return scaling_factor

def ustd(x):
    """
    See https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation#Bias_correction
    """
    # Reshape to handle both 1d and 2d arrays
    if x.ndim == 1:
        n = x.size
        unbiased_variance = np.var(x, ddof=1)
    else:
        n = x.shape[1]
        unbiased_variance = np.var(x, ddof=1, axis=1)

    unbiased_std = np.sqrt(unbiased_variance) / get_scaling_factor(n)

    return unbiased_std

if __name__ == "__main__":
    assert abs(get_scaling_factor(2) - 0.7978845608) < 1e-6
    assert abs(get_scaling_factor(3) - 0.8862269255) < 1e-6
    assert abs(get_scaling_factor(4) - 0.9213177319) < 1e-6
    # test that this actually works
    x = np.random.normal(size=(100000,4))
    sigma = ustd(x)
    assert abs(np.mean(sigma) - 1) < 0.005

