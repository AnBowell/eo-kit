from re import M
import EOkit
from EOkit import gaussian_processes
from EOkit.smoothers import whittaker
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter


def main():

    data = pd.read_csv("data/proc_test_vci3m.csv")

    days = data["dates"].to_numpy(dtype=np.float64)
    vci = (
        data["VCI3M"].to_numpy(dtype=np.float64)
        + np.random.standard_normal(days.size) * 2
    )

    rust_start = perf_counter()

    # print(days,vci)
    rust_smoothed_data = gaussian_processes.single_gp(
        days, vci, forecast_spacing=7, forecast_amount=10, length_scale=40
    )

    rust_end = perf_counter()
    print("Rust GP done. This took {}s".format(rust_end - rust_start))

    ten_seven_day_forecasts_x = np.append(
        days, np.arange(max(days), max(days) + (7 * 10), 7)
    )

    plots_smoothed_unsmoothed(
        days, vci, ten_seven_day_forecasts_x, rust_smoothed_data, "Single GP"
    )

    rust_start = perf_counter()

    days_list = [days] * 300
    vci_list_input = [vci] * 300

    # print(days,vci)
    rust_smoothed_data = gaussian_processes.multiple_gps(
        days_list,
        vci_list_input,
        forecast_spacing=7,
        forecast_amount=10,
        length_scale=50,
        amplitude=0.5,
        noise=0.01,
        n_threads=-1,
    )

    rust_end = perf_counter()
    print("Rust GP done. This took {}s".format(rust_end - rust_start))

    plots_smoothed_unsmoothed(
        days, vci, ten_seven_day_forecasts_x, rust_smoothed_data[0], "Multi GP"
    )

    rust_start = perf_counter()

    weights = np.full(vci.size, 1.0, dtype=np.float64)
    rust_smoothed_data = whittaker.single_whittaker(vci, weights, 5, 3)

    rust_end = perf_counter()
    print("Rust whittaker done. This took {}s".format(rust_end - rust_start))

    plots_smoothed_unsmoothed(days, vci, days, rust_smoothed_data, "Single Whittaker")

    vci_inputs_whitt = [vci] * 1000
    weights_input_whitt = [weights] * 1000

    rust_start = perf_counter()

    rust_smoothed_data = whittaker.multiple_whittakers(
        vci_inputs_whitt, weights_input_whitt, 5, 3, n_threads=-1
    )

    rust_end = perf_counter()
    print("Rust whittaker done. This took {}s".format(rust_end - rust_start))

    plots_smoothed_unsmoothed(days, vci, days, rust_smoothed_data[0], "Multi Whittaker")

    plt.show()


def plots_smoothed_unsmoothed(x, y, x_smoothed, y_smoothed, title):

    fig, ax1 = plt.subplots(figsize=(8, 5))

    plt.title("{}".format(title))
    plt.plot(x, y, label="Raw data")
    plt.plot(x_smoothed, y_smoothed, label="Smoothed data")


if __name__ == "__main__":
    main()
