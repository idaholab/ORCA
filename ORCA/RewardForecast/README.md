# RewardForecast Objects

The RewardForecast objects are intended to be easily swappable. Each RewardForecast 
object returns reward/price information as a numpy.ndarray or list for each point in the 
optimization time horizon. Additional input parameters may be included to represent the 
necessary variables to perform the reward/price forecast via the specific algorithm. To 
make these objects swappable, each object must contain the following:

## `gen_reward()` method

This method requires no inputs and returns the reward/price data as a numpy.ndarray or 
list.

### Output

#### `reward`

The output is a numpy.ndarray or list with the number of entries equal to the number of 
time points in the optimization time horizon.

## `i` attribute

This attribute is a counter that keeps track of how many times reward/price information 
has been requested. It must be incremented every time `gen_reward()` is called. For 
some methods of producing reward/price information this counter is very important (such 
as for `StaticHistoricalForecast`).