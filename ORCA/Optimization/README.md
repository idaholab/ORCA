# Optimization Objects

The Optimization objects are intended to be easily swappable. Each Optimization object 
performs the dispatch optimization using a different algorithm or method. Additional 
input parameters may be included to represent the necessary variables for performing the 
dispatch optimization for each individual Optimization object. To make these 
objects swappable, each object must contain the following:

## `return_next_dispatch(rewards, x_init)` method

This method takes in a rewards/price dictionary and a list of the initial state values, 
performs the dispatch optimization over the time horizon, and returns the optimal next 
dispatch values as a dictionary.

### Inputs

#### `rewards`

This input is a dictionary. The keys are the names of the reward/price information and 
the values are the reward/price data as a numpy.ndarray or list.

#### `x_init`

This input is a numpy.ndarray or list of the initial state values.

### Output

#### `result`

The output is a dictionary. The keys are "states", "control", and "measurements". The 
values are lists of the state, control, and measurement values in order as they were 
specified when instantiating the object. These values represent the optimal values of 
the next dispatch at a time step dt forward from the initial states.