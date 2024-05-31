
from do_mpc.data import save_results, load_results
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import matplotlib.patches as mpatches  # Import for patches

class MPCPlotter:
    def __init__(self, mpc):
        plt.style.use('ggplot')
        self.mpc = mpc
        self.fig = plt.figure(figsize=(12, 6))
        self.outer = gridspec.GridSpec(1, 2, width_ratios=[2, 1], wspace=0.2)
    
        # Subplot for 2D plot (left side)
        self.ax_2d = plt.Subplot(self.fig, self.outer[0])
        self.fig.add_subplot(self.ax_2d)

        # Subplots for y, lambda, H (right side)
        self.inner = gridspec.GridSpecFromSubplotSpec(4, 1, subplot_spec=self.outer[1], wspace=0.1, hspace=0.4)

        self.ax_y = plt.Subplot(self.fig, self.inner[0])
        self.ax_lambda = plt.Subplot(self.fig, self.inner[1])
        self.ax_H = plt.Subplot(self.fig, self.inner[2])
        self.ax_costf = plt.Subplot(self.fig, self.inner[3])


        self.fig.add_subplot(self.ax_y)
        self.fig.add_subplot(self.ax_lambda)
        self.fig.add_subplot(self.ax_H)
        self.fig.add_subplot(self.ax_costf)

        self.colors = None
        self.pred_colors = None
        self.tree_dim = None

    def plot_saved_data(self):
        for i in range(len(self.mpc.data['_x', 'Xrobot'])):
            self.plot(l=i)
    
    def plot(self, l=-1, online= True):
        self.ax_2d.clear()  # Clear the 2D plot
        self.ax_y.clear()   # Clear the y plot
        self.ax_lambda.clear()  # Clear the lambda plot
        self.ax_H.clear()   # Clear the H plot
        self.ax_costf.clear()   # Clear the H plot

        # Plot 2D on the left
        self.plot_robot_position(l=l)

        # Plot y-lambda-H on the right
        self.plot_prediction_table(l=l)

        # Set labels for the right-side plots
        self.ax_y.set_xlabel('Time Steps')
        self.ax_lambda.set_xlabel('Time Steps')
        self.ax_H.set_xlabel('Time Steps')
        self.ax_costf.set_xlabel('Time Steps')

        plt.tight_layout()  # Ensure subplots are properly spaced
        if online: plt.pause(0.01)  # Pause to update the plot in the loop

    def plot_robot_position(self, l = -1):
        x_pred, y_pred, yaw_pred = self.mpc.data.prediction(('_x', 'Xrobot'),l-1)
        x = self.mpc.data['_x', 'Xrobot'][:l, 0]
        y = self.mpc.data['_x', 'Xrobot'][:l, 1]
        yaw = self.mpc.data['_x', 'Xrobot'][:l, 2]

        self.ax_2d.plot(x, y, marker='o', linestyle='-', color='b', label='Robot Path')
        self.ax_2d.plot(x_pred, y_pred, marker='o', linestyle='-', color='r', label='Predicted Path')

        arr_len = 1.0 
        for i in range(len(x)):
            arrow = mpatches.Arrow(x[i], y[i], arr_len * np.cos(yaw[i]), arr_len * np.sin(yaw[i]), color='b',linewidth=0.1, width=.1 )
            self.ax_2d.add_patch(arrow)

        for i in range(len(x_pred)):
            arrow = mpatches.Arrow(x_pred[i,0], y_pred[i,0], arr_len * np.cos(yaw_pred[i,0]), arr_len * np.sin(yaw_pred[i,0]), color='r', linewidth=0.1, width=.1)
            self.ax_2d.add_patch(arrow)

        self.ax_2d.set_title('Robot Position')
        self.ax_2d.legend()

    def plot_prediction_table(self,l=-1):
                
        # Generate the colors array automatically
        if self.mpc.data['_x', 'y'].shape[1] != self.tree_dim :
            self.colors = plt.cm.get_cmap('tab10', self.mpc.data['_x', 'y'].shape[1])  # 'tab10' is one of the colormaps in matplotlib
            self.pred_colors = plt.cm.get_cmap('viridis', self.mpc.data['_x', 'y'].shape[1])

        for i in  range(self.mpc.data['_x', 'y'].shape[1]):
            data_y = self.mpc.data['_x', 'y'][:l, i]
            prediction_data_y = self.mpc.data.prediction(('_x', 'y'),l-1)[i,:]
            self.ax_y.plot(range(len(data_y)), data_y, label=f'y{i}', linestyle='-', color=self.colors(i))
            self.ax_y.plot(np.array(range(len(data_y), len(data_y) + len(prediction_data_y)))-1, prediction_data_y, color=self.pred_colors(i), label=f'y_pred{i}', linestyle='--')
            self.ax_y.set_title(f'Prediction of y')
        
        self.ax_y.legend()

        for i in range(self.mpc.data['_x', 'lambda'].shape[1]):
            data_lambda = self.mpc.data['_x', 'lambda'][:l, i]
            prediction_data_lambda = self.mpc.data.prediction(('_x', 'lambda'),l-1)[i,:]
            self.ax_lambda.plot(range(len(data_lambda)), data_lambda, label=f'lambda{i}', linestyle='-', color=self.colors(i))
            self.ax_lambda.plot(np.array(range(len(data_lambda), len(data_lambda) + len(prediction_data_lambda)))-1, prediction_data_lambda,color=self.pred_colors(i), label=f'lambda_pred{i}', linestyle='--')
            self.ax_lambda.set_title('Prediction of lambda')
        self.ax_lambda.legend()

        for i in range(self.mpc.data['_aux', 'H'].shape[1]):
            prediction_data_H = self.mpc.data.prediction(('_aux', 'H'),l-1)[i,:]
            data_H = self.mpc.data['_aux', 'H'][:l, i]
            self.ax_H.plot(range(len(data_H)), data_H, label='H', linestyle='-', color='y')
            self.ax_H.plot(np.array(range(len(data_H), len(data_H) + len(prediction_data_H)))-1, prediction_data_H, label='H_pred', linestyle='--', color='b')
            self.ax_H.set_title('Prediction of H')
        self.ax_H.legend()

        for i in range(self.mpc.data['_aux', 'cost_function'].shape[1]):
            prediction_data_H = self.mpc.data.prediction(('_aux', 'cost_function'),l-1)[i,:]
            data_H = self.mpc.data['_aux', 'cost_function'][:l, i]
            self.ax_costf.plot(range(len(data_H)), data_H, label=r'$\sum_1^{Ntree}\lambda_k*log(\lambda_k)-\sum_1^{Ntree}\lambda_{k-1}*log(\lambda_{k-1})$', linestyle='-', color='y')
            self.ax_costf.plot(np.array(range(len(data_H), len(data_H) + len(prediction_data_H)))-1, prediction_data_H, label='Pred', linestyle='--', color='b')
            self.ax_costf.set_title('Prediction of cost_function')
        self.ax_costf.legend()

        
        # Adjust title positions
        self.ax_y.title.set_y(1.05)
        self.ax_lambda.title.set_y(1.05)
        self.ax_H.title.set_y(1.05)
        self.ax_costf.title.set_y(1.05)

    def save_plot_as_gif(self, filename='plot.gif', interval=100, frames=None):
        if frames is None:
            frames = len(self.mpc.data['_x', 'Xrobot'])

        # Define a function to update the plot for each frame
        def update(frame):
            self.plot_robot_position(frame)
            self.plot_prediction_table(frame)

        # Create animation using FuncAnimation
        ani = FuncAnimation(self.fig, update, frames=frames, interval=interval)

        # Save the animation as a GIF
        ani.save(filename, writer='pillow')
        
    def save_plot_as_mp4(self, filename='plot.mp4', interval=100, frames=None):
        if frames is None:
            frames = len(self.mpc.data['_x', 'Xrobot'])

        # Define a function to update the plot for each frame
        def update(frame):
            self.plot_robot_position(frame)
            self.plot_prediction_table(frame)

        # Create animation using FuncAnimation
        ani = FuncAnimation(self.fig, update, frames=frames, interval=interval)

        # Save the animation as an MP4
        ani.save(filename, writer='ffmpeg')

import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import numpy as np

class MPCGUI:
    def __init__(self, master, mpc):

        self.master = master
        self.mpc_plotter = mpc
        self.current_frame = 0
        
        self.fig = self.mpc_plotter.fig
        self.canvas = FigureCanvasTkAgg(self.fig, master=master)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        self.plot_frame()

        self.frame_slider = ttk.Scale(master, from_=0, to=len(self.mpc_plotter.mpc.data['_x', 'Xrobot']) - 1, orient=tk.HORIZONTAL, command=self.on_slider_move)
        self.frame_slider.pack(side=tk.BOTTOM, fill=tk.X)
        self.frame_slider.set(0)

    def on_slider_move(self, event):
        self.current_frame = int(self.frame_slider.get())
        self.plot_frame()

    def plot_frame(self):
        self.mpc_plotter.plot(l=self.current_frame, online=False)
        self.canvas.draw()


class mpc_data:
    def __init__(self, data):
        self.data= data['mpc']



import argparse
import glob
import os
import tkinter as tk
from datetime import datetime

def get_most_recent_file(directory, pattern):
    """Return the most recent file matching the given pattern in the directory."""
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError("No files found in the directory.")
    most_recent_file = max(files, key=os.path.getmtime)
    return most_recent_file

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MPC Data Viewer")
    parser.add_argument('--file_number', type=int, help='The number of the result file to load (e.g., 060 for 060_results.pkl)')
    args = parser.parse_args()
    
    results_directory = 'results'
    if args.file_number is not None:
        file_name = f'{args.file_number:03d}_results.pkl'
        file_path = os.path.join(results_directory, file_name)
    else:
        file_path = get_most_recent_file(results_directory, '*_results.pkl')

    data = load_results(file_path)
    plotter = MPCPlotter(mpc_data(data))

    if True: 
        plotter.save_plot_as_gif(filename='test.gif')
    else:
        root = tk.Tk()
        root.title("MPC Data Viewer")
        mpc_gui = MPCGUI(root, plotter)
        root.mainloop()
