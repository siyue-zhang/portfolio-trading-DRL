
import os
import pandas as pd
from matplotlib import pyplot as plt

files = os.listdir("./log")
print(files)
all = pd.DataFrame()
for i in range(4):
    df = pd.read_csv(f"./log/{files[i]}")
    all[f"{i+2011}"] = df
all.plot()
plt.show()