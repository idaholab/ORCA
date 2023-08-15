class Optimization(object):
    """
    MPC dispatch optimization.

    This is the basic class. The only required method is the return_next_dispatch method.

    The return_next_dispatch method for this class returns the initial values for states,
    zeros for controls, and nothing for measurements.

    Parameters
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    states : dict
        dictionary of information about state variables
    control : dict
        dictionary of information about control variables
    measurements : dict or None, optional
        dictionary of information about measurement variables

    Attributes
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    n : int
        number of steps to take in time horizon
    states : dict
        dictionary of information about state variables
    control : dict
        dictionary of information about control variables
    measurements : dict or None, optional
        dictionary of information about measurement variables

    Methods
    -------
    check_states_control_measurements_dicts(name, test_dict)
        checks that states, control, and measurement dictionaries are properly inputted
    return_next_dispatch(rewards, x_init)
        returns state, control, and measurement information at optimal dispatch

    """

    def __init__(
        self,
        t_window=60.0 * 12.0,
        dt=5.0,
        states={},
        control={},
        measurements=None,
        objective={},
        **specs,
    ):
        # get time window, time step, and number of steps to take
        assert isinstance(t_window, float), "t_window must be float."
        assert isinstance(dt, float), "dt must be float."
        self.t_window = t_window
        self.dt = dt
        self.n = int(self.t_window / self.dt)

        # ensure states input dictionary has everything needed
        assert isinstance(states, dict), "states must be dictionary."
        self.check_states_control_measurements_dicts("states", states)
        self.states = states

        # ensure control input dictionary has everything needed
        assert isinstance(control, dict), "control must be dictionary."
        self.check_states_control_measurements_dicts("control", control)
        self.control = control

        # ensure optional measurements dictionary has everything needed
        if measurements is not None:
            assert isinstance(
                measurements, dict
            ), "measurements must be dictionary or None."
            self.check_states_control_measurements_dicts("measurements", measurements)
        self.measurements = measurements

        # ensure objective dictionary has everything needed
        assert isinstance(objective, dict), "objective must be dictionary."
        assert "sense" in objective, "objective must contain 'sense' key"
        assert objective["sense"] in [
            "maximize",
            "minimize",
        ], "'sense' must be either 'maximize' or 'minimize'"
        objective_keys = list(objective.keys())
        objective_keys.remove("sense")
        for key in objective_keys:
            # take care of state information
            assert (
                "state_multiplier" in objective[key]
            ), f"'state_multiplier' list must be in {key} for objective dictionary."
            assert isinstance(
                objective[key]["state_multiplier"], list
            ), f"'state_multiplier' in {key} for objective dictionary must be list."
            assert len(objective[key]["state_multiplier"]) == len(
                self.states["order"]
            ), f"number of states in {key} for objective dictionary must be same as in states dictionary."
            # take care of control information
            assert (
                "control_multiplier" in objective[key]
            ), f"'control_multiplier' list must be in {key} for objective dictionary."
            assert isinstance(
                objective[key]["control_multiplier"], list
            ), f"'control_multiplier' in {key} for objective dictionary must be list."
            assert len(objective[key]["control_multiplier"]) == len(
                self.control["order"]
            ), f"number of control variables in {key} for objective dictionary must be same as in control dictionary."
            # take care of measurement information (optional)
            if "measurement_multiplier" in objective[key]:
                assert isinstance(
                    objective[key]["measurement_multiplier"], list
                ), f"'measurement_multiplier' in {key} for objective dictionary must be list."
                assert isinstance(
                    self.measurements, dict
                ), f"to use 'measurement_multiplier' in {key} for objective dictionary, measurement dictionary must be defined."
                assert len(objective[key]["measurement_multiplier"]) == len(
                    self.measurements["order"]
                ), f"number of measurement variables in {key} for objective dictionary must be same as in measurement dictionary."
        self.objective = objective

    def check_states_control_measurements_dicts(self, name, test_dict):
        """
        Checks that all required keys are in dictionary, all values are lists, and
        all lists have the same length

        Parameters
        ----------
        name : str
            name of the dictionary
        test_dict : dict
            dictionary to test

        """
        # required keys
        req_keys = ["order", "lb", "ub"]
        lens = []
        for key in req_keys:
            assert key in test_dict, f"{key} missing from {name} dictionary."
            assert isinstance(test_dict[key], list), f"{key} in {name} must be list."
            lens.append(len(test_dict[key]))
        assert all(lens), f"all lists in {name} must have same length."

    def return_next_dispatch(self, rewards, x_init):
        """
        Solves the Pyomo ConcreteModel and returns state, control, and measurement values of next step

        Parameters
        ----------
        rewards : dict
            dictionary keys are names of reward/price, values are numpy.ndarray or list of n reward/price samples
        x_init : numpy.ndarray or list
            initial state values in order given by states['order']

        Returns
        -------
        result : dict
            dictionary with states, control, and measurements values in lists

        """

        # return values of states, control, measurements
        result = {"states": [], "control": [], "measurements": []}
        # states
        for i in range(len(x_init)):
            result["states"].append(x_init[i])
        # control
        for i in range(len(self.control["order"])):
            result["control"].append(0.0)
        # measurements
        if self.measurements is not None:
            for i in range(len(self.measurements["order"])):
                result["measurements"].append(0.0)

        return result
