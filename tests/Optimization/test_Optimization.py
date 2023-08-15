import unittest
import pkgutil
import importlib
import os
from ORCA.Basic.Optimization import Optimization
import ORCA.Optimization
from ..data.SamplePKLFile import generate_matrices_pkl_from_csv
from ORCA.RewardForecast.StaticHistoricalForecast import StaticHistoricalForecast


class TestOptimization(unittest.TestCase):
    """
    Global Optimization tests.

    This tests instantiation and overloaded methods. Existence and type of required
    attributes are checked as a part of instantiation. Type and shape of outputs are
    checked for overloaded methods. No check on correct values are made, these type of
    checks should be implemented in separate tests.

    """

    def setUp(self):
        # gather all Optimization objects
        self.all_optimization = [Optimization]
        # get all in Optimization directory
        for module_info in pkgutil.walk_packages(
            ORCA.Optimization.__path__, ORCA.Optimization.__name__ + "."
        ):
            module_string = module_info.name
            specific_class = module_string.split(".")[-1]
            module = importlib.import_module(module_string)
            self.all_optimization.append(getattr(module, specific_class))

        # example optimization problem comes from storage_data.csv
        # states: qNPP (50.0), SOC (0.0, 20.0)
        # control: qC (0.0, 20.0), qD (0.0, 20.0)
        # measurement: SOC2 (0.0, 20.0)
        # price: LMP
        # StaticHistoricalForecast will be used to generate reward data

        # generate .pkl file for A, B, C matrices for LTIStateSpaceMPCPyomoOptimization
        generate_matrices_pkl_from_csv()

        # specs that will work for instantiation
        self.specs = {
            "t_window": 720.0,
            "dt": 5.0,
            "states": {"order": ["qNPP", "SOC"], "lb": [0.0, 0.0], "ub": [50.0, 20.0]},
            "control": {"order": ["qC", "qD"], "lb": [0.0, 0.0], "ub": [20.0, 20.0]},
            "measurements": {"order": ["SOC2"], "lb": [0.0], "ub": [20.0]},
            "objective": {
                "sense": "maximize",
                "price": {
                    "state_multiplier": [1.0, 0.0],
                    "control_multiplier": [-1.0, 1.0],
                    "measurement_multiplier": [0.0],
                },
            },
            # specs specific to LTIStateSpaceMPCPyomoOptimization
            "solver": "glpk",
            "matrices": os.path.join(
                os.path.dirname(__file__), "..", "data", "ABC.pkl"
            ),
        }

    def tearDown(self):
        os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "ABC.pkl"))

    def test_instantiation(self):
        """
        Tests that Optimization objects instantiate and contain required attributes.
        """

        required_attributes = [
            "t_window",
            "dt",
            "n",
            "states",
            "control",
            "measurements",
            "objective",
        ]
        required_type = [float, float, int, dict, dict, dict, dict]

        for mod in self.all_optimization:
            obj = mod(**self.specs)

            # check that required attributes exist and are correct type
            for att, typ in zip(required_attributes, required_type):
                self.assertTrue(
                    hasattr(obj, att),
                    f"Required attribute {att} missing for {obj.__class__}",
                )
                self.assertTrue(
                    isinstance(getattr(obj, att), typ), f"{att} should be {typ}"
                )

            # check that n is calculated correctly
            self.assertEqual(
                obj.n,
                int(obj.t_window / obj.dt),
                f"{obj.__class__} calculates number of steps in time horizon, n, incorrectly.",
            )

    def test_t_window_input(self):
        """
        Tests that Optimization input 't_window' checks are performed correctly.
        """

        # t_window must be float, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check["t_window"] = "taco"
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when t_window is not float.",
            )

    def test_dt_input(self):
        """
        Tests that Optimization input 'dt' checks are performed correctly.
        """

        # dt must be float, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check["dt"] = "taco"
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when dt is not float.",
            )

    def dictionary_checks(self, key):
        """
        Tests states, control, and measurement dictionaries.

        This function effectively tests check_states_control_measurements_dicts.

        Parameters
        ----------
        key : str
            name of dictionary to check

        """

        # key must be dict, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check[key] = []
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {key} is not dict.",
            )

        # dictionary must have keys: 'order', 'lb', 'ub', if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check[key] = {"order": [], "lb": []}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {key} is missing ub key.",
            )
        spec_check = self.specs.copy()
        spec_check[key] = {"order": [], "ub": []}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {key} is missing lb key.",
            )
        spec_check = self.specs.copy()
        spec_check[key] = {"lb": [], "ub": []}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {key} is missing order key.",
            )

        # states values must be lists, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check[key] = {"order": "taco", "lb": [], "ub": []}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when order is not list.",
            )
        spec_check = self.specs.copy()
        spec_check[key] = {"order": [], "lb": "taco", "ub": []}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when lb is not list.",
            )
        spec_check = self.specs.copy()
        spec_check[key] = {"order": [], "lb": [], "ub": "taco"}
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when ub is not list.",
            )

        # states values must all be same length lists
        spec_check = self.specs.copy()
        spec_check[key]["order"] = []
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {key} lists are not all same length.",
            )

    def test_states_input(self):
        """
        Tests that Optimization input 'states' checks are performed correctly.
        """

        self.dictionary_checks("states")

    def test_control_input(self):
        """
        Tests that Optimization input 'control' checks are performed correctly.
        """

        self.dictionary_checks("control")

    def test_measurements_input(self):
        """
        Tests that Optimization input 'measurements' checks are performed correctly.
        """

        self.dictionary_checks("measurements")

    def test_objective_input(self):
        """
        Tests that Optimization input 'objective' checks are performed correctly.
        """

        # objective must be dictionary, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check["objective"] = []
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when objective is not dict.",
            )

        # objective must have a sense key, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check["objective"].pop("sense")
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when objective dictionary is missing sense key.",
            )
        # sense must be minimize or maximize, if not, raise AssertionError
        spec_check = self.specs.copy()
        spec_check["objective"]["sense"] = "zero"
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when sense is not minimize or maximize.",
            )

        # for each reward dictionary, check that multipliers are correct
        self.multiplier_checks("state")
        self.multiplier_checks("control")
        self.multiplier_checks("measurement")

    def multiplier_checks(self, name):
        """
        Performs checks for name_multiplier in price dictionary.

        Parameters
        ----------
        name : str
            name of multiplier to check

        """
        # for each reward dictionary, check that multipliers are correct
        spec_check = self.specs.copy()
        spec_check["objective"]["price"].pop(f"{name}_multiplier")
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {name}_multiplier is missing.",
            )
        spec_check = self.specs.copy()
        spec_check["objective"]["price"][f"{name}_multiplier"] = "taco"
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {name}_multiplier is not a list.",
            )
        spec_check = self.specs.copy()
        spec_check["objective"]["price"][f"{name}_multiplier"] = [0.0] * 100
        for mod in self.all_optimization:
            self.assertRaises(
                AssertionError,
                mod,
                **spec_check,
                msg=f"{mod} should have AssertionError when {name}_multiplier list larger than {name}.",
            )

    def test_return_next_dispatch(self):
        """
        Tests Optimization return_next_dispatch method.
        """

        # use a StaticHistoricalForecast
        forecast_spec = {
            "t_window": self.specs["t_window"],
            "dt": self.specs["dt"],
            "history": os.path.join(
                os.path.dirname(__file__), "..", "data", "storage_data.csv"
            ),
            "name": "LMP",
        }
        forecast = StaticHistoricalForecast(**forecast_spec)
        for mod in self.all_optimization:
            obj = mod(**self.specs)
            rewards = {"price": forecast.gen_reward()}
            x_init = [50.0, 0.0]
            next_dispatch = obj.return_next_dispatch(rewards, x_init)

            # make sure next_dispatch is a dictionary
            self.assertTrue(
                isinstance(next_dispatch, dict),
                f"{obj.__class__} return_next_dispatch did not return dict.",
            )

            for key in ["states", "control", "measurements"]:
                # make sure states, control, measurements are keys in next_dispatch
                self.assertTrue(
                    key in next_dispatch,
                    f"{obj.__class__} return_next_dispatch did not return key: {key}.",
                )
                # make sure values are list
                self.assertTrue(
                    isinstance(next_dispatch[key], list),
                    f"{obj.__class__} return_next_dispatch did not return list for {key}.",
                )
                # make sure each list has correct number of entries
                self.assertEqual(
                    len(next_dispatch[key]),
                    len(self.specs[key]["order"]),
                    f"{obj.__class__} return_next_dispatch returned wrong number of entries for {key}.",
                )
