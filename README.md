# ROS Library for Neural-MPC Interaction with Unity Simulation 🤖🌐

This repository hosts a ROS library designed to facilitate interaction between a Neural-MPC system and a Unity Simulation environment.

### Folder Structure 📁 :
```
.
├── CMakeLists.txt
├── Main.py
├── README.md
├── cfg
│   └── camera.yaml
├── include
│   └── unity-drone-sim-bridge
├── package.xml
├── scripts
│   ├── DroneClass.py
│   └── main.py
├── setup.py
└── src
    └── unity_drone_sim_bridge
        ├── BridgeClass.py
        ├── StateClass.py
        ├── __init__.py
        ├── nn_lib
        │   ├── __init__.py
        │   ├── NeuralClass.py
        │   └── tools.py
        ├── ros_com_lib
        │   ├── __init__.py
        │   ├── ClientNode.py
        │   ├── PublisherNode.py
        │   └── SubscriberNode.py
        └── sensors.py
```
- **cfg/**: Work in progress; this directory will contain configuration files for injecting object class populations.
- **scripts/**: Work in progress; ready-to-go scripts will be stored here.
- **script/unity_drone_sim_bridge/**: This main folder collects all essential tools:
    - `BridgeClass.py`: Class responsible for setting up all ROS communication components and managing I/O communication between the control system and the simulation.
    - `StateClass.py`: Class responsible for managing the system's model and populating the [`do_mpc.model.Model()`](https://www.do-mpc.com/en/latest/api/do_mpc.model.Model.html#do_mpc.model.Model) class.
    - `sensors.py`: *Work in progress*; intended to be used as a *config* file where ROS components to instantiate are declared.
    - **/nn_lib/**: A library dedicated to setting up and executing Neural Network (NN) and Genetic Programming (GP) components.
    - **/ros_com_lib/**: A library declaring ROS communication component classes
