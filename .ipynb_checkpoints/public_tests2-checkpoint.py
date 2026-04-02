import numpy as np


def estimate_gaussian_test(target):
    print("Running estimate_gaussian_test...")

    X = np.array([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0]
    ])

    mu, var = target(X)

    expected_mu = np.array([3.0, 4.0])
    expected_var = np.array([
        8.0 / 3.0,
        8.0 / 3.0
    ])

    assert isinstance(mu, np.ndarray), "mu should be a numpy.ndarray"
    assert isinstance(var, np.ndarray), "var should be a numpy.ndarray"
    assert mu.shape == (2,), f"Wrong mu shape: got {mu.shape}, expected (2,)"
    assert var.shape == (2,), f"Wrong var shape: got {var.shape}, expected (2,)"
    assert np.allclose(mu, expected_mu), f"Wrong mu: got {mu}, expected {expected_mu}"
    assert np.allclose(var, expected_var), f"Wrong var: got {var}, expected {expected_var}"

    X = np.array([
        [2.0],
        [4.0],
        [6.0],
        [8.0]
    ])

    mu, var = target(X)
    expected_mu = np.array([5.0])
    expected_var = np.array([5.0])

    assert mu.shape == (1,), f"Wrong single-feature mu shape: got {mu.shape}"
    assert var.shape == (1,), f"Wrong single-feature var shape: got {var.shape}"
    assert np.allclose(mu, expected_mu), f"Wrong single-feature mu: got {mu}, expected {expected_mu}"
    assert np.allclose(var, expected_var), f"Wrong single-feature var: got {var}, expected {expected_var}"

    print("All tests passed!")


def select_threshold_test(target):
    print("Running select_threshold_test...")

    y_val = np.array([0, 0, 1, 1, 0, 1])
    p_val = np.array([0.9, 0.8, 0.1, 0.05, 0.7, 0.02])

    epsilon, F1 = target(y_val, p_val)

    assert isinstance(epsilon, (float, np.floating, int)), f"Wrong epsilon type: got {type(epsilon)}"
    assert isinstance(F1, (float, np.floating, int)), f"Wrong F1 type: got {type(F1)}"
    assert 0 <= F1 <= 1, f"F1 should be in [0, 1], got {F1}"
    assert F1 > 0.8, f"F1 too low: got {F1}, expected > 0.8"

    predictions = (p_val < epsilon)
    tp = np.sum((predictions == 1) & (y_val == 1))
    assert tp >= 1, "The returned epsilon did not detect any true anomaly"

    print("All tests passed!")
