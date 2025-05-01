import pandas as pd
from matplotlib import pyplot as plt

df = pd.read_csv('tdacc-run.csv')
print(df)

df.plot(x="NUM_THREADS", y="TIME")

plt.savefig("threads.png")

df2 = pd.read_csv('tdacc-run-mpi.csv')
print(df2)

df2.plot(x="MPI/OMP", y="TIME")

plt.savefig("mpiomp.png")
