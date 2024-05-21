import plotext as plt
import random
import time

# Initialize empty lists for storing data
x_data = []
y_data = []

# Set the number of data points to display at once
window_size = 20

# Create a loop to update the plot in real-time
for _ in range(100):
    # Generate new data point
    new_x = len(x_data) + 1
    new_y = random.uniform(0, 10)
    
    # Append new data to lists
    x_data.append(new_x)
    y_data.append(new_y)
    
    # Maintain the window size
    if len(x_data) > window_size:
        x_data.pop(0)
        y_data.pop(0)
    
    # Clear the plot
    plt.clear_figure()
    
    # Plot the data
    plt.plot(x_data, y_data)
    
    # Display the plot
    plt.show()
    
    # Pause for a short duration to simulate real-time update
    time.sleep(0.5)
