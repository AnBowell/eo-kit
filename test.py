import EOkit
from EOkit import gaussian_processes
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter


data = pd.read_csv("data/proc_test_vci3m.csv")


days = data["dates"].to_numpy(dtype=np.float64)
vci = data["VCI3M"].to_numpy(dtype=np.float64)


rust_start = perf_counter()

# print(days,vci)
rust_smoothed_data = gaussian_processes.rust_run_single_gp(
    days, vci, forecast_spacing=7, forecast_amount=10, length_scale=40
)

rust_end = perf_counter()
print("Rust GP done. This took {}s".format(rust_end - rust_start))



plt.plot(days,vci)
plt.plot(np.append(days,np.arange(max(days),max(days)+(7*10),7)), rust_smoothed_data)
plt.show()




rust_start = perf_counter()


days_list = [days]*10
vci_no_means = [vci_arr - np.mean(vci_arr) for vci_arr in [vci]*10]


# print(days,vci)
rust_smoothed_data = gaussian_processes.run_multiple_gps(
    days_list, vci_no_means, forecast_spacing=7, forecast_amount=10,             length_scale=50,
            amplitude=0.5,
            noise=0.01,
)

rust_end = perf_counter()
print("Rust GP done. This took {}s".format(rust_end - rust_start))



plt.plot(days,vci)
plt.plot(np.append(days,np.arange(max(days),max(days)+(7*10),7)), rust_smoothed_data[0]+np.mean(vci))
plt.show()
