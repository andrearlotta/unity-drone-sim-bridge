"""
    Metrics plotter for the MPC performance metrics (time, cost, etc.)
"""
import matplotlib.pyplot as plt
import numpy as np
import glob
import os
import pickle

def load_data(directory):
    files =  sorted(glob.glob(os.path.join(directory, 'synthetic*.pkl')))
    print(files)
    data_list = {}
    for file in files:
        with open(file, 'rb') as f:
            data = pickle.load(f)
            data_list[file] = data['mpc']
    return data_list

def extract_metrics(data_list, metric):
    values = {}
    for namefile, data in data_list.items():
        if hasattr(data, metric):
            print(metric)
            print(np.array(getattr(data, metric)).shape)
            print(namefile)
            values[namefile] = getattr(data, metric)
    return values

def plot_metrics(data_list, metrics, method='A', save_path='plot.png'):
    if method == 'A':
        for metric in metrics:
            metric_values = extract_metrics(data_list, metric)
            if not metric_values:
                continue
            plt.figure(figsize=(10, 6))
            for idx, data_values in metric_values.items():
                plt.plot(range(len(data_values)), data_values, label=f'{metric} for test {idx}')
            plt.xlabel('Iteration Count [#]')
            plt.ylabel(f'{metric} [s]')
            #plt.yscale('log') 
            plt.title(f'{metric} over Iterations for Different Tests')
            plt.legend()
            plt.savefig(f'{metric}_method_A.png')
            plt.clf()

    elif method == 'B':
        for metric in metrics:
            metric_dict= extract_metrics(data_list, metric) 
            metric_values = list(metric_dict.values()) 
            if not metric_values:
                continue
            
            metric_values = np.array(metric_values)[:,:,0]
            print(metric_values.shape)
            mean_values = np.mean(metric_values, axis=1)
            std_values = np.std(metric_values, axis=1)

            plt.figure(figsize=(10, 6))
            plt.plot(metric_dict.keys(), mean_values, label=f'Mean {metric}')
            plt.fill_between(metric_dict.keys(), mean_values - std_values, mean_values + std_values, alpha=0.2, label=f'Standard Deviation {metric}')
            plt.xlabel('Iteration Count')
            plt.ylabel(metric)
            plt.yscale('log') 
            plt.xticks(rotation=45)
            plt.title(f'Mean and Standard Deviation of {metric} over Iterations')
            plt.legend()
            plt.tight_layout()  # Adjust layout to prevent clipping of labels
            plt.savefig(f'{metric}_method_B.png')
            plt.clf()

if __name__ == '__main__':
    directory = 'results'
    data_list = load_data(directory)

    metrics = [ 'success','t_proc_callback_fun', 't_proc_nlp_f', 't_proc_nlp_g', 't_proc_nlp_grad',
                    't_proc_nlp_grad_f', 't_proc_nlp_hess_l', 't_proc_nlp_jac_g', 't_wall_total',
                    't_wall_callback_fun', 't_wall_nlp_f', 't_wall_nlp_g', 't_wall_nlp_grad', 't_wall_nlp_grad_f',
                    't_wall_nlp_hess_l', 't_wall_nlp_jac_g']

    plot_metrics(data_list, metrics, method='A')
    plot_metrics(data_list, metrics, method='B')
