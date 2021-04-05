import matplotlib.pyplot as plt
import numpy as np


# file_path = 'aws_loss_dump.txt'
file_path = 'lap_loss_dump.txt'

loss_dump = open(file_path, "r")
loss_dump = loss_dump.readlines()

loss_dump = [float(line.split()[-1]) for line in loss_dump]
indices = np.arange(len(loss_dump))

plt.plot(indices, loss_dump)
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

