import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import casadi as ca
from unity_drone_sim_bridge.surrogate_lib.nn_tools import random_input_test_std
from unity_drone_sim_bridge.surrogate_lib.nn_models import SurrogateNetworkFixedOutput
from sklearn.model_selection import train_test_split
import l4casadi as l4c
import plotly.graph_objects as go
import plotly.io as pio

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def LoadNN(hidden_size, hidden_layer, use_yolo=True):
    """
    Carica una rete neurale già allenata (SurrogateNetworkFixedOutput)
    """
    # Supponiamo che questo sia il nome dell'esperimento e il percorso del checkpoint
    EXPERIMENT_NAME = "cartesian_SurrogateNetworkFixedOutput_hs64_hl2_lr0_001_e50_bs1_ts0_2_synFalse"
    CHECKPOINT_PATH = find_best_model_with_highest_epoch(f"./checkpoints/{EXPERIMENT_NAME}")
    
    model = SurrogateNetworkFixedOutput(use_yolo, hidden_size, hidden_layer)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    print(f"Loading model from {CHECKPOINT_PATH}")
    print(f"Number of parameters: {count_parameters(model)}")

    return model


def find_best_model_with_highest_epoch(folder_path):
    import re
    pattern = re.compile(r'best_model_epoch_(\d+)\.ckpt')
    return max(
        (os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.match(f)),
        key=lambda f: int(pattern.search(f).group(1)),
        default=None
    )

def random_input_test_std(l4c_model, model, gpu):
    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    x_sym = ca.MX.sym('x', 3, 1)
    y_sym = l4c_model(x_sym)
    f = ca.Function('y', [x_sym], [y_sym])
    
    for _ in range(1000):
        i_ = np.array([[np.random.uniform(-10, 10),np.random.uniform(-10, 10), 0.0]])
        i_[0,-1] = np.arctan2(i_[0,0], i_[0,1])
        p_ = f(i_.reshape((3, 1)))
        print(p_)
        p_ = p_[0, 0]
        t_out = model(torch.tensor(i_, dtype=torch.float32).to("cuda" if gpu else "cpu")).detach().cpu().numpy().flatten()
        
        random_inputs.append(i_[0])
        torch_prediction.append(t_out)
        prediction.append(p_)

    # Plot using Plotly and display the plots immediately
    fig1 = create_3d_plot(np.array(random_inputs), [], np.array(prediction).flatten(), 'test_l4casasdi')
    fig2 = create_3d_plot(np.array(random_inputs), [], np.array(torch_prediction).flatten(), 'test_l4casasdi_torch')

    # Show the plots instead of saving to files
    fig1.show()
    fig2.show()

def create_3d_plot(X_combined, Y_combined, y_pred, exp_name):
    fig = go.Figure()
    if len(Y_combined):
        fig.add_scatter3d(
            x=X_combined[:, 0],
            y=X_combined[:, 1],
            z=Y_combined,
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name="True"
        )
    fig.add_scatter3d(
        x=X_combined[:, 0],
        y=X_combined[:, 1],
        z=y_pred,
        mode='markers',
        marker=dict(size=5, opacity=0.8) if len(Y_combined) else dict(size=5, opacity=0.8, color=y_pred, colorscale='Viridis'),
        name="Predicted"
    )
    fig.update_layout(
        title=f'3D Scatter Plot of Values - {exp_name}',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Conf Value'
        ),
        legend_title="Dataset"
    )
    return fig

if __name__ == '__main__':
    # Configurazione del test
    hidden_size = 64
    hidden_layer = 2
    use_yolo = True
    gpu = False  # Cambia a True se vuoi usare la GPU

    # Carica il modello neurale
    model = LoadNN(hidden_size, hidden_layer, use_yolo)
    l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=True, device="cuda" if gpu else "cpu")
    # Puoi caricare il modello L4CasADi qui, se necessario
    # Esegui random_input_test_std
    random_input_test_std(l4c_model, model, gpu=gpu)
