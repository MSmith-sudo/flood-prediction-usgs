import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_parquet("data/site.parquet")
df["stage"].plot()
plt.title("Stage (hourly)")
plt.xlabel("Time")
plt.ylabel("Stage")
plt.show()