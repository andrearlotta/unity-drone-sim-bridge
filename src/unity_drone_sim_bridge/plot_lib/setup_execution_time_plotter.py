import matplotlib.pyplot as plt
import csv
import glob
import re

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

# Function to generate labels based on filenames
def generate_label(filename):
    label = re.sub(r'setup_time_mlp', 'MLP', filename)
    label = re.sub(r'variable_dim', '', label)
    label = re.sub(r'.csv', '', label)
    label = label.replace('_', ' ').title()
    
    # Identify if 'rt' or 'refactor' is in the filename and add to label
    if 'refactor' in label.lower():
        label += ' (Refactor)'
    if 'weight' in label.lower():
        label = label.replace('Weight', 'fov weighting fun').title()
    if 'obstacle' in label.lower():
        label = label.replace('Obstacle', '& Obstacle constr.').title()
    return label

# Find all relevant files
file_patterns = [
    'setup_time_mlp*.csv'
]
import os
filenames = []
for pattern in file_patterns:
    for filename in sorted(glob.glob(pattern)):
        filenames.append(filename)
print(filenames)
# Initialize the plots
plt.figure(figsize=(12, 6))


# Plot for setup times
plt.subplot(1, 2, 1)
for i, filename in enumerate(filenames):
    num_trees, setup_times, _ = read_data(filename)
    label = generate_label(os.path.basename(filename))
    # Determine linestyle based on label content
    linestyle = '--' if 'rt' in label.lower() else 'solid'
    plt.plot(num_trees, setup_times, marker='o' if 'refactor' in label.lower() else 'x', linestyle=linestyle, label=label)
    plt.xticks(num_trees)
plt.xlabel('Number of Trees Considered')
plt.ylabel('Setup Time (s)')
plt.title('Setup Times')
plt.yscale('log') 
plt.legend()
plt.grid(True)


# Plot for setup times
plt.subplot(1, 2, 2)
# Plotting
for filename in filenames:
    num_trees, _, t_wall_total_times = read_data(filename)
 
    
    # Determine linestyle based on label content
    linestyle = '--' if 'rt' in label.lower() else 'solid'
    
    plt.plot(num_trees, t_wall_total_times, marker='o' if 'refactor' in label.lower() else 'x', linestyle=linestyle, label=label)
    plt.xticks(num_trees)

plt.xlabel('Number of Trees Considered')
plt.ylabel('t_wall_total (s)')
plt.title('t_wall_total Times')

plt.legend()
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Save the plot as a PNG file
plt.savefig('setup_times_vs_num_trees_comparison.png')

# Show the plot (optional)
plt.show()
