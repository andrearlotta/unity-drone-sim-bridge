import os
import numpy as np
import torch
import casadi as ca
from unity_drone_sim_bridge.surrogate_lib.nn_tools import random_input_test_std, LoadNN
from unity_drone_sim_bridge.surrogate_lib.nn_models import SurrogateNetworkFixedOutput
import l4casadi as l4c
import plotly.graph_objects as go

def random_input_test_std(l4c_model, model, gpu, n_input, is_polar):
    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    x_sym = ca.MX.sym('x', n_input, 1)
    y_sym = l4c_model(x_sym)
    f = ca.Function('y', [x_sym], [y_sym])
    
    x_vals = np.linspace(0, 15, 250) if is_polar else np.linspace(-10,10,250)
    y_vals = np.linspace(-np.pi, +np.pi, 250) if is_polar else np.linspace(-10,10,250)
    X, Y = np.meshgrid(x_vals, y_vals)

    import torch
    Z = np.zeros_like(X)
    Z_t = np.zeros_like(X)
    # Compute the function for each (x, y) pair
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            x, y = X[i, j], Y[i, j]

            Z[i, j] = f(np.array([x,y]).reshape((n_input, 1)))
            Z_t[i, j] = model(torch.tensor([[x, y]], dtype=torch.float32)).item()  # Convert to scalar

    # Transform X and Y based on the angle (polar coordinates transformation)
    X_transformed = X * np.cos(Y) if is_polar else X
    Y_transformed = X * np.sin(Y) if is_polar else Y
    # Plot using Plotly and display the plots immediately
    fig1 = create_3d_plot(X_transformed, Y_transformed, Z, 'test_l4casasdi', is_polar=is_polar)
    fig2 = create_3d_plot(X_transformed, Y_transformed, Z_t, 'test_torch', is_polar=is_polar)

    # Show the plots instead of saving to files
    fig1.show()
    fig2.show()

def create_3d_plot(X_combined, Y_combined, y_pred, exp_name, is_polar):
    fig = go.Figure()
    
    fig.add_surface(
        x=X_combined,
        y=Y_combined,
        z=y_pred,
        #mode='markers',
        #marker=dict(size=5, opacity=0.8) if len(Y_combined) else dict(size=5, opacity=0.8, color=y_pred, colorscale='Viridis'),
        name="Predicted"
    )
    fig.update_traces(contours_z=dict(show=True, usecolormap=True,
                                    highlightcolor="limegreen", project_z=True))

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
    hidden_size = 16
    hidden_layer = 2
    n_input = 3
    gpu = False  # Cambia a True se vuoi usare la GPU
    is_polar = False

    # Carica il modello neurale
    model = LoadNN(hidden_size, hidden_layer, n_input)
    l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=True, device="cuda" if gpu else "cpu")
    # Esegui random_input_test_std
    random_input_test_std(l4c_model, model, gpu=gpu, n_input=n_input, is_polar=is_polar)
