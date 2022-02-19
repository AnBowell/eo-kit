import EOkit
from EOkit import gp
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from time import perf_counter


data = pd.read_csv("data/proc_test_vci3m.csv")


days = data["dates"].to_numpy(dtype=np.float64)
vci = data["VCI3M"].to_numpy(dtype=np.float64)


rust_start = perf_counter()

# print(days,vci)
rust_smoothed_data = gp.rust_run_single_gp(
    days, vci, forecast_spacing=0, forecast_amount=0, length_scale=40
)

rust_end = perf_counter()
print("Rust GP done. This took {}s".format(rust_end - rust_start))

plt.plot(days,vci)
plt.plot(days, rust_smoothed_data)
plt.show()
