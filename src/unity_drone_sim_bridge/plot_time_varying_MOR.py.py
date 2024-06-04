import matplotlib.pyplot as plt
import csv

# Function to read data from CSV file
def read_data(filename):
    num_trees = []
    setup_times = []
    t_wall_total_times = []
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip the header row
        for row in reader:
            num_trees.append(int(row[0]))
            setup_times.append(float(row[1]))
            t_wall_total_times.append(float(row[2]))
    return num_trees, setup_times, t_wall_total_times

# Filenames and labels
filenames = [
    'setup_time_gp_cond.csv',
    'setup_time_gp_no_cond.csv',
    'setup_time_gp_cond_1_time_g_load.csv',
    'setup_time_mlp_cond.csv',
    'setup_time_mlp_no_cond.csv',
    'setup_time_mlp_cond_1_time_g_load.csv'
]

labels = [
    'GP with Condition',
    'GP without Condition',
    'GP With Fixed Condition',
    'MLP with Condition',
    'MLP without Condition',
    'MLP With Fixed Condition'
]

# Initialize the plots
plt.figure(figsize=(12, 6))

# Plot for setup times
plt.subplot(1, 2, 1)
for i, filename in enumerate(filenames):
    num_trees, setup_times, _ = read_data(filename)
    plt.plot(num_trees, setup_times, marker='o', label=labels[i])
plt.xlabel('Number of Trees Considered')
plt.ylabel('Setup Time (s)')
plt.title('Setup Times')
plt.legend()
plt.grid(True)

# Plot for t_wall_total times
plt.subplot(1, 2, 2)
for i, filename in enumerate(filenames):
    num_trees, _, t_wall_total_times = read_data(filename)
    plt.plot(num_trees, t_wall_total_times, marker='o', label=labels[i])
plt.xlabel('Number of Trees Considered')
plt.ylabel('t_wall_total (s)')
plt.title('t_wall_total Times')
plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plots as PNG files
plt.savefig('setup_times_vs_num_trees_comparison_full.png')

# Show the plots (optional)
plt.show()
