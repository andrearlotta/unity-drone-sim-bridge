.
├── build
│   └── bdist.linux-x86_64
├── cfg
│   └── camera.yaml
├── CMakeLists.txt
├── include
│   └── unity-drone-sim-bridge
├── launch
│   └── data_association.launch
├── package.xml
├── README.md
├── scripts
│   ├── data_association_node.py
│   └── main.py
├── setup.py
├── src
│   └── unity_drone_sim_bridge
│       ├── baselines
│       │   └── drone_controller.py
│       ├── examples
│       │   ├── reduced_order_system
│       │   │   ├── main.py
│       │   │   ├── __pycache__
│       │   │   │   ├── main.cpython-39.pyc
│       │   │   │   ├── template_model.cpython-39.pyc
│       │   │   │   ├── template_mpc.cpython-39.pyc
│       │   │   │   └── template_simulator.cpython-39.pyc
│       │   │   ├── template_model.py
│       │   │   ├── template_mpc.py
│       │   │   └── template_simulator.py
│       │   ├── sitl
│       │   │   ├── fixed_order_system
│       │   │   │   ├── main_.py
│       │   │   │   ├── __pycache__
│       │   │   │   │   ├── template_model.cpython-39.pyc
│       │   │   │   │   ├── template_mpc.cpython-39.pyc
│       │   │   │   │   └── template_simulator.cpython-39.pyc
│       │   │   │   ├── template_model.py
│       │   │   │   ├── template_mpc.py
│       │   │   │   └── template_simulator.py
│       │   │   ├── mpc_dimension_adjustment
│       │   │   │   ├── main.py
│       │   │   │   ├── __pycache__
│       │   │   │   │   ├── template_model.cpython-39.pyc
│       │   │   │   │   ├── template_mpc.cpython-39.pyc
│       │   │   │   │   └── template_simulator.cpython-39.pyc
│       │   │   │   ├── template_model.py
│       │   │   │   ├── template_mpc.py
│       │   │   │   └── template_simulator.py
│       │   │   ├── __pycache__
│       │   │   │   └── setup_fake_field.cpython-39.pyc
│       │   │   └── setup_fake_field.py
│       │   ├── test_surrogate.py
│       │   ├── yolo_order_system
│       │   │   ├── main.py
│       │   │   ├── __pycache__
│       │   │   │   ├── main.cpython-39.pyc
│       │   │   │   ├── template_model.cpython-39.pyc
│       │   │   │   ├── template_mpc.cpython-39.pyc
│       │   │   │   └── template_simulator.cpython-39.pyc
│       │   │   ├── template_model.py
│       │   │   ├── template_mpc.py
│       │   │   └── template_simulator.py
│       │   ├── yolo_reduced_order_system
│       │   │   ├── main.py
│       │   │   ├── __pycache__
│       │   │   │   ├── main.cpython-39.pyc
│       │   │   │   ├── template_model.cpython-39.pyc
│       │   │   │   ├── template_mpc.cpython-39.pyc
│       │   │   │   └── template_simulator.cpython-39.pyc
│       │   │   ├── template_model.py
│       │   │   ├── template_mpc.py
│       │   │   └── template_simulator.py
│       │   └── yolo_system_1tree
│       │       ├── acados_example.py
│       │       ├── main.py
│       │       ├── model.py
│       │       ├── __pycache__
│       │       │   ├── main.cpython-39.pyc
│       │       │   ├── model.cpython-39.pyc
│       │       │   ├── template_model.cpython-39.pyc
│       │       │   ├── template_mpc.cpython-39.pyc
│       │       │   └── template_simulator.cpython-39.pyc
│       │       ├── template_model.py
│       │       ├── template_mpc.py
│       │       └── template_simulator.py
│       ├── generic_tools.py
│       ├── gp_mpc_lib
│       │   ├── gp_class.py
│       │   ├── gp_functions.py
│       │   ├── __init__.py
│       │   ├── model_class.py
│       │   ├── mpc_class.py
│       │   ├── optimize.py
│       │   └── __pycache__
│       │       ├── gp_class.cpython-39.pyc
│       │       ├── gp_functions.cpython-39.pyc
│       │       ├── __init__.cpython-39.pyc
│       │       ├── mpc_class.cpython-39.pyc
│       │       └── optimize.cpython-39.pyc
│       ├── __init__.py
│       ├── plot_lib
│       │   ├── mpc_metrics_plotter.py
│       │   ├── mpc_plotter.py
│       │   ├── __pycache__
│       │   │   └── MpcPlotter.cpython-39.pyc
│       │   └── setup_execution_time_plotter.py
│       ├── __pycache__
│       │   ├── BridgeClass.cpython-38.pyc
│       │   ├── BridgeClass.cpython-39.pyc
│       │   ├── DroneClass.cpython-38.pyc
│       │   ├── DroneClass.cpython-39.pyc
│       │   ├── DroneClassMultiTree.cpython-39.pyc
│       │   ├── DroneClassMultiTreeMor.cpython-39.pyc
│       │   ├── generic_tools.cpython-39.pyc
│       │   ├── MainNode_ReducedOrder.cpython-39.pyc
│       │   ├── MpcClass.cpython-38.pyc
│       │   ├── MpcClass.cpython-39.pyc
│       │   ├── MpcPlotter.cpython-39.pyc
│       │   ├── run_simulation.cpython-39.pyc
│       │   ├── sensors.cpython-38.pyc
│       │   ├── sensors.cpython-39.pyc
│       │   ├── StateClass.cpython-38.pyc
│       │   ├── StateClass.cpython-39.pyc
│       │   ├── template_model.cpython-39.pyc
│       │   ├── template_mpc.cpython-39.pyc
│       │   └── template_simulator.cpython-39.pyc
│       ├── qi_lib
│       │   ├── __pycache__
│       │   │   └── qi_tools.cpython-39.pyc
│       │   └── qi_tools.py
│       ├── ros_com_lib
│       │   ├── bridge_class.py
│       │   ├── client_image_pos.py
│       │   ├── client_node.py
│       │   ├── publisher_node.py
│       │   ├── __pycache__
│       │   │   ├── bridge_class.cpython-39.pyc
│       │   │   ├── ClientNode.cpython-38.pyc
│       │   │   ├── client_node.cpython-39.pyc
│       │   │   ├── ClientNode.cpython-39.pyc
│       │   │   ├── PublisherNode.cpython-38.pyc
│       │   │   ├── publisher_node.cpython-39.pyc
│       │   │   ├── PublisherNode.cpython-39.pyc
│       │   │   ├── sensors.cpython-39.pyc
│       │   │   ├── SubscriberNode.cpython-38.pyc
│       │   │   ├── subscriber_node.cpython-39.pyc
│       │   │   └── SubscriberNode.cpython-39.pyc
│       │   ├── sensors.py
│       │   └── subscriber_node.py
│       └── surrogate_lib
│           ├── gp_tools.py
│           ├── __init__.py
│           ├── load_database.py
│           ├── nn_models.py
│           ├── nn_tools.py
│           ├── __pycache__
│           │   ├── generic_tools.cpython-39.pyc
│           │   ├── g_func_tools.cpython-39.pyc
│           │   ├── gp_nn_tools.cpython-39.pyc
│           │   ├── gp_tools.cpython-39.pyc
│           │   ├── __init__.cpython-38.pyc
│           │   ├── __init__.cpython-39.pyc
│           │   ├── load_database.cpython-39.pyc
│           │   ├── NeuralClass.cpython-38.pyc
│           │   ├── NeuralClass.cpython-39.pyc
│           │   ├── nn_models.cpython-39.pyc
│           │   ├── nn_tools.cpython-38.pyc
│           │   ├── nn_tools.cpython-39.pyc
│           │   └── surrogate_func_tools.cpython-39.pyc
│           ├── surrogate_func_tools.py
│           └── test_rl.py
├── srv
│   └── GetTreesPoses.srv
└── tree.md

37 directories, 138 files
