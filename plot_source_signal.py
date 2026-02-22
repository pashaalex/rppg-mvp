import pandas as pd
import matplotlib.pyplot as plt

# load signal
df = pd.read_csv("signals.csv")
signal = df["face_g_mean"]

# plot
plt.figure()
plt.plot(signal)
plt.xlabel("Frame index")
plt.ylabel("Green channel intensity")
plt.title("Raw green channel signal")
plt.tight_layout()
plt.show()
