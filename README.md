# ORCA

Optimization of Real-time Capacity Allocation

This Python package performs dispatch optimization for real-time economic optimization.

## Installation

Clone the repository, navigate to the directory containing setup.py and execute:
```
pip install -e .
```

## Use

ORCA can be used via individual components or with the `CollectedNextDispatch` object.

The `Optimization` and `RewardForecast` objects are intended to have a simple, cohesive
interface that allows plug and play use. Each `Optimization` object is required to have
a `return_next_dispatch` method that performs the optimization and returns the next time
step's optimal dispatch. Each `RewardForecast` object is required to have a `gen_reward`
method that returns n samples of the reward/price data required in the time horizon for
optimization. New optimization or reward forecast algorithms may be implemented and when
placed in the appropriate directory, the `CollectedNextDispatch` object can find and use
them or they may be used independently for optimization workflows.

The `CollectedNextDispatch` object instantiates the required `Optimization` and
`RewardForecast` objects specified in a YAML file. Note that only one `Optimization`
object is required, but many `RewardForecast` objects may be specified to represent
reward/price information of multiple components. An example of a YAML specification file
is given in `notebooks/CollectedNextDispatchExample.yaml`.

## Examples

Examples of how to use the various objects within ORCA are given in the `notebooks`
directory.

## Unit Tests

Each `Optimization` and `RewardForecast` object should have associated unit tests. These
can be found in the `tests` directory. They are written using the `unittest` framework
and can be run using a command like `python -m unittest discover -v` from the directory
adjacent to `tests`.

## Code Formatting
ORCA uses [black](https://black.readthedocs.io/en/stable/) for code formatting.

## Citing ORCA
ORCA is included in the U.S. Department of Energy [CODE database](https://www.osti.gov/doecode/biblio/112006), which includes citation guidelines for several citation styles.
DOI:10.11578/dc.20230815.1
