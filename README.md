This repository provides the data and source code used to reproduce the computational experiments reported in the manuscript. The model evaluates cooperative pricing strategies between public transport (PT) and ride-hailing (RH) under a stochastic user equilibrium framework.

1. Repository Structure
```text
.
├── data/
│   ├── OD_demand.csv
│   ├── road_network.gpkg
│   ├── car_ksp_ps_results.csv
│   └── bus_od2od_shortest.csv
├── code/
│   ├── main.py
│   └── model1.py
│   
└── README.md
```

2. Requirements

The code was implemented in Python. The main required packages are:
numpy
pandas
geopandas
scikit-learn
pygad

The required packages can be installed using:
pip install numpy pandas geopandas scikit-learn pygad

3. Input Data

The model requires the following input files:

1. OD_demand.csv: OD demand matrix.
2. road_network.gpkg: road network attribute data, including link ID, free-flow travel time, capacity, and link length.
3. car_ksp_ps_results.csv: candidate car/RH paths and path-size factors.
4. bus_od2od_shortest.csv: public transport path information for each OD pair.


4. Running the Model

Before running the model, update the data_dir variable in main.py to the local directory containing the input data.
Then run:
python main.py
The program first evaluates the baseline scenario and then applies a two-stage genetic algorithm to optimize the pricing strategies for public transport and ride-hailing.
