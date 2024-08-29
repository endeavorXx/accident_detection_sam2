import matplotlib.pyplot as plt
import numpy as np

# Data from the table
models = ['N', 'S', 'M', 'L', 'X']
latency = [8.2389, 8.7219, 14.1924, 20.4150, 28.2899]
throughput = [121.3755, 114.6537, 70.4603, 48.9837, 35.3483]

# Plotting the data
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for Average Latency per frame (ms)
ax1.set_xlabel('Model')
ax1.set_ylabel('Avg. Latency per frame (ms)', color='tab:blue')
ax1.bar(models, latency, color='tab:blue', alpha=0.6, label='Latency (ms)')
ax1.tick_params(axis='y', labelcolor='tab:blue')

# Create a second y-axis to plot the Throughput (fps)
ax2 = ax1.twinx()
ax2.set_ylabel('Throughput (fps)', color='tab:orange')
ax2.bar(models, throughput, color='tab:orange', alpha=0.6, label='Throughput (fps)')
ax2.tick_params(axis='y', labelcolor='tab:orange')

# Adding title and legend
fig.suptitle('YOLOV8 Model Performance')
fig.legend(loc='upper left', bbox_to_anchor=(0.1,0.9))

# Showing the plot
plt.savefig('fig1.png')
