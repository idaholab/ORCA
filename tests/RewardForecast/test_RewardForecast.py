import unittest
import pkgutil
import importlib
import os
import numpy as np
from ORCA.Basic.RewardForecast import RewardForecast
import ORCA.RewardForecast


class TestRewardForecast(unittest.TestCase):
    """
    Global RewardForecast tests.

    This tests instantiation and overloaded methods. Existance and type of required
    attributes are checked as a part of instantiation. Type and shape of outputs are
    checked for overloaded methods. No check on correct values are made, these type of
    checks should be implemented in separate tests.

    """

    def setUp(self):
        # specs to initialize
        self.specs = {
            "t_window": 720.0,
            "dt": 5.0,
            # SinusoidalForecast specific inputs
            "amplitude": 10.0,
            "phase": np.pi / 4.0,
            "frequency": 2.0 * np.pi / 144.0,
            "offset": 10.0,
            # StaticHistoricalForecast specific inputs
            "history": os.path.join(
                os.path.dirname(__file__), "..", "data", "storage_data.csv"
            ),
            "name": "LMP",
        }

        # specs for failure
        self.specs_failure1 = self.specs.copy()
        self.specs_failure1["t_window"] = 10
        self.specs_failure2 = self.specs.copy()
        self.specs_failure2["dt"] = 1

        self.all_reward_forecasts = [RewardForecast]
        # get all belonging to the RewardForecast directory
        for module_info in pkgutil.walk_packages(
            ORCA.RewardForecast.__path__, ORCA.RewardForecast.__name__ + "."
        ):
            module_string = module_info.name
            specific_class = module_string.split(".")[-1]
            module = importlib.import_module(module_string)
            self.all_reward_forecasts.append(getattr(module, specific_class))

    def test_instantiation(self):
        """
        Tests that the RewardForecast object is instantiated and contains required attributes.
        """

        required_attributes = ["t_window", "dt", "n", "i"]
        required_type = [float, float, int, int]

        for mod in self.all_reward_forecasts:
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
                f"{obj.__class__} calculates number of steps in time horizon, n, incorrectly",
            )
            # check that counter initialized to 0
            self.assertEqual(
                obj.i, 0, f"{obj.__class__} does not initialize counter i to 0"
            )

            # check that incorrect type inputs throw AssertionErrors
            self.assertRaises(
                AssertionError,
                mod,
                **self.specs_failure1,
                msg=f"{obj.__class__} should have AssertionError when 't_window' is int",
            )
            self.assertRaises(
                AssertionError,
                mod,
                **self.specs_failure2,
                msg=f"{obj.__class__} should have AssertionError when 'dt' is int",
            )

    def test_gen_reward(self):
        """
        Tests that gen_reward() returns the correct type and amount of samples.
        """

        for mod in self.all_reward_forecasts:
            obj = mod(**self.specs)

            for i in range(2):
                self.assertEqual(
                    i,
                    obj.i,
                    f"{obj.__class__} gen_reward does not increment counter i correctly",
                )
                reward = obj.gen_reward()
                self.assertTrue(
                    isinstance(reward, np.ndarray),
                    f"{obj.__class__} gen_reward returns something other than numpy.ndarray",
                )
                self.assertEqual(
                    i + 1,
                    obj.i,
                    f"{obj.__class__} gen_reward does not increment counter i correctly",
                )


if __name__ == "__main__":
    unittest.main()
