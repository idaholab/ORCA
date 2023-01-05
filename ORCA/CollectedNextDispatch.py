import importlib
import os
import yaml
from yaml import Loader
import pandas as pd


def instantiate_optimization_or_reward_object(specs, object_type):
    """
    Instantiates and returns an Optimization or RewardForecast object.

    Parameters
    ----------
    specs : dict
        dictionary of specifications for the object to be instantiated
    object_type : str
        'Optimization' or 'RewardForecast'

    Returns
    -------
    obj : ORCA.Optimization or ORCA.RewardForecast
        Optimization or RewardForecast instantiated object

    """

    if object_type == specs["type"]:
        # requested basic version of object
        module_name = "ORCA.Basic." + object_type
    elif object_type == "Optimization":
        module_name = "ORCA.Optimization." + specs["type"]
    else:
        # object_type is "RewardForecast"
        module_name = "ORCA.RewardForecast." + specs["type"]

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise ImportError(f"Requested {object_type} {specs['type']} not found!")

    obj = getattr(module, specs["type"])(**specs)

    return obj


class CollectedNextDispatch(object):
    """
    ORCA Workflow Object

    This object parses the spec file and instantiates relevant Optimization
    and RewardForecast objects. Methods are provided to get the next step's
    optimal dispatch and stores the history.

    Parameters
    ----------
    spec_path : str
        path to YAML spec file

    Attributes
    ----------
    specs : dict
        dictionary of specifications
    optimization : ORCA.Optimization
        optimization object
    reward : dict
        dictionary of all ORCA.RewardForecast objects
    initial_states : pandas.DataFrame
        DataFrame of initial state values at each time step
    optimal_results : pandas.DataFrame
        DataFrame of results of optimization including states, controls, measurements, and rewards

    Methods
    -------
    return_optimal_next_dispatch(time, x_init)
        returns optimal next dispatch states, controls, and measurements
    reset_objects()
        resets counter in RewardForecast objects, sets initial_states and optimal_results to None

    """

    def __init__(self, spec_path):
        # ensure spec_path is a valid file path
        assert os.path.isfile(
            spec_path
        ), f"{spec_path} is not a valid path to a YAML spec file."
        # parse spec file
        try:
            with open(spec_path, "r") as f:
                self.specs = yaml.load(f, Loader=Loader)
        except yaml.YAMLError as exc:
            raise ValueError(f"{spec_path} spec file could not be parsed: ", exc)
        # ensure required keys are in specs
        req_keys = ["t_window", "dt", "optimization", "reward"]
        for key in req_keys:
            assert key in self.specs, f"{key} missing from spec file."
        # make sure optimization and reward are dicts
        assert isinstance(
            self.specs["optimization"], dict
        ), "optimization key in spec file must be dictionary."
        assert isinstance(
            self.specs["reward"], dict
        ), "reward key in spec file must be dictionary."
        # add t_window and dt to optimization dictionary and all reward dictionaries
        add_keys = ["t_window", "dt"]
        for key in add_keys:
            self.specs["optimization"][key] = self.specs[key]
            for rkey in self.specs["reward"]:
                self.specs["reward"][rkey][key] = self.specs[key]

        # instantiate optimization object
        self.optimization = instantiate_optimization_or_reward_object(
            self.specs["optimization"], "Optimization"
        )

        # instantiate all reward forecast objects
        self.reward = {}
        for key in self.specs["reward"]:
            self.reward[key] = instantiate_optimization_or_reward_object(
                self.specs["reward"][key], "RewardForecast"
            )

        # set up placeholders for initial_states and optimal_results
        self.initial_states = None
        self.optimal_results = None

    def return_optimal_next_dispatch(self, time, x_init):
        """
        Returns and stores optimal next dispatch by running gen_reward for each RewardForecast
        and return_next_dispatch from the Optimization object.

        Results are stored in initial_states and optimal_results as pandas DataFrames.

        Parameters
        ----------
        time : pandas.Timestamp or datetime
            initial time for optimization (time when x_init takes place)
        x_init : numpy.ndarray or list
            initial state values in order of specs['states']['order']

        Returns
        -------
        result : dict
            dictionary with states, control, and measurement values in order of spec['states']['order'], etc.

        """

        # store the initial states
        initial_dict = {"Time": time}
        for i in range(len(self.specs["optimization"]["states"]["order"])):
            initial_dict[self.specs["optimization"]["states"]["order"][i]] = [x_init[i]]
        initial = pd.DataFrame(initial_dict)
        if self.initial_states is None:
            self.initial_states = initial
        else:
            self.initial_states = pd.concat(
                [self.initial_states, initial], ignore_index=True
            )

        # generate reward/price forecasts
        rewards = {key: self.reward[key].gen_reward() for key in self.reward}

        # get the optimal next dispatch
        result = self.optimization.return_next_dispatch(rewards, x_init)

        # store result in optimal_results
        current_time = time + pd.Timedelta(minutes=self.specs["dt"])
        optimal_dict = {"Time": [current_time]}
        # store states
        for i in range(len(self.specs["optimization"]["states"]["order"])):
            optimal_dict[self.specs["optimization"]["states"]["order"][i]] = [
                result["states"][i]
            ]
        # store control
        for i in range(len(self.specs["optimization"]["control"]["order"])):
            optimal_dict[self.specs["optimization"]["control"]["order"][i]] = [
                result["control"][i]
            ]
        # store measurements if specified
        if self.optimization.measurements is not None:
            for i in range(len(self.specs["optimization"]["measurements"]["order"])):
                optimal_dict[self.specs["optimization"]["measurements"]["order"][i]] = [
                    result["measurements"][i]
                ]
        # store rewards
        for key in rewards:
            optimal_dict[key] = [rewards[key][1]]
        optimal = pd.DataFrame(optimal_dict)
        if self.optimal_results is None:
            self.optimal_results = optimal
        else:
            self.optimal_results = pd.concat(
                [self.optimal_results, optimal], ignore_index=True
            )

        return result

    def reset_objects(self):
        """
        Returns the counter in RewardForecast objects to 0 and sets initial_states
        and optimal_results to None

        """

        # set counter in RewardForecast objects to 0
        for key in self.reward:
            self.reward[key].i = 0

        # set initial_states and optimal_results to None
        self.initial_states, self.optimal_results = None, None
