import EOkit
from EOkit import gp
import numpy as np
import matplotlib.pyplot as plt

from time import perf_counter


rust_start = perf_counter()


days = np.arange(1,500,1)

print(len(days))

vci = 50 + np.cos(25*days) + np.random.rand(days.size) * 20

# print(days,vci)
rust_smoothed_data = gp.rust_run_single_gp(
    days, vci, forecast_spacing=0, forecast_amount=0, length_scale=60
)

plt.plot(days,vci)
# plt.plot(np.append(days,np.arange(np.max(days), np.max(days) + (7*10),7)), rust_smoothed_data)
plt.show()


rust_end = perf_counter()
print("Rust GP done. This took {}s".format(rust_end - rust_start))