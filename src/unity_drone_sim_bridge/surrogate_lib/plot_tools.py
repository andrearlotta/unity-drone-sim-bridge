import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import torch
import casadi as ca


def create_3d_plot(X_combined, Y_combined, y_pred, exp_name, is_polar= False):
    fig = go.Figure()
    if len(Y_combined):
        fig.add_scatter3d(
            x=X_combined[:, 0] * np.cos(X_combined[:, 1]) if is_polar else X_combined[:, 0],
            y=X_combined[:, 0] * np.sin(X_combined[:, 1]) if is_polar else X_combined[:, 1],
            z=Y_combined,
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name="True"
        )
    fig.add_scatter3d(
        x=X_combined[:, 0] * np.cos(X_combined[:, 1]) if is_polar else X_combined[:, 0],
        y=X_combined[:, 0] * np.sin(X_combined[:, 1]) if is_polar else X_combined[:, 1],
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


def random_input_test_rt(casadi_quad_approx_func, l4c_model_order2, gpu, model):

    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    for _ in range(1000):
        i_ = np.array([[np.random.uniform(0, 10), np.random.uniform(0, 2*np.pi), 0]])
        p_ = casadi_quad_approx_func(i_.reshape((3, 1)), l4c_model_order2.get_params(i_))
        print(p_)
        p_ = p_[0, 0]
        t_out = model(torch.tensor(i_, dtype=torch.float32).to("cuda" if gpu else "cpu")).detach().cpu().numpy().flatten()
        random_inputs.append(i_[0])
        torch_prediction.append(t_out)
        prediction.append(p_)

    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(prediction).flatten(), 'test_l4casasdi',  is_polar=(exp['dataset_type'] == 'is_polar')), "test_l4casasdi_random_3d_plot.html")
    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(torch_prediction).flatten(), 'test_l4casasdi',  is_polar=(exp['dataset_type'] == 'is_polar')), "test_l4casasdi_torch_random_3d_plot.html")


def random_input_test_std(l4c_model, model, gpu):
    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    x_sym = ca.MX.sym('x', 3, 1)
    y_sym = l4c_model(x_sym)
    f = ca.Function('y', [x_sym], [y_sym])
    for _ in range(1000):
        i_ = np.array([[np.random.uniform(0, 10), np.random.uniform(0, 2*np.pi), 0]])
        p_ = f(i_.reshape((3, 1)))
        print(p_)
        p_ = p_[0, 0]
        t_out = model(torch.tensor(i_, dtype=torch.float32).to("cuda" if gpu else "cpu")).detach().cpu().numpy().flatten()
        random_inputs.append(i_[0])
        torch_prediction.append(t_out)
        prediction.append(p_)

    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(prediction).flatten(), 'test_l4casasdi',  is_polar=(exp['dataset_type'] == 'is_polar')),       "test_l4casasdi_random_3d_plot.html")
    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(torch_prediction).flatten(), 'test_l4casasdi',  is_polar=(exp['dataset_type'] == 'is_polar')), "test_l4casasdi_torch_random_3d_plot.html")

