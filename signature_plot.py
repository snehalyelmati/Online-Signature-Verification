import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("./signature_data/U1S2.TXT", delimiter=' ', names=['X', 'Y', 'TS', 'T', 'AZ', 'AL', 'P'],
                             header=None,
                             skiprows=1)

plt.plot(df['X'], df['Y'])
plt.show()