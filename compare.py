import numpy as np
from math import factorial


def main():
    x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 2, 3, 4])

    z = np.convolve(x, y, mode="valid")
    print(z)

    z = np.convolve(x, y, mode="full")
    print(z)

    x = np.arange(0, 20, 1) ** 2

    smoothed = savitzky_golay(x, 4, 3, 0, 1)
    print(smoothed)


def savitzky_golay(y, window_size, order, deriv=0, rate=1):

    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    # print(half_window)
    b = np.mat(
        [[k**i for i in order_range] for k in range(-half_window, half_window + 1)]
    )
    # print(b)
    m = np.linalg.pinv(b).A[deriv] * rate**deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1 : half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1 : -1][::-1] - y[-1])

    y = np.concatenate((firstvals, y, lastvals))

    return np.convolve(m[::-1], y, mode="valid")


if __name__ == "__main__":
    main()
