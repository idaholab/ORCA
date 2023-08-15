import unittest
import os
import pandas as pd
import numpy as np
from ORCA.RewardForecast.StaticHistoricalForecast import StaticHistoricalForecast


class TestStaticHistoricalForecast(unittest.TestCase):
    """
    StaticHistoricalForecast tests.

    This tests instantiation and all other methods belonging to StaticHistoricalForecast.
    """

    def setUp(self):
        # specs that cause AssertionError for history not str
        self.specs_fail1 = {"history": 1}
        # specs that cause AssertionError for history path
        self.specs_fail2 = {"history": "taco/burrito.csv"}
        # specs that cause AssertionError for name not str
        self.specs_fail3 = {
            "history": os.path.join(
                os.path.dirname(__file__), "..", "storage_data.csv"
            ),
            "name": ["LMP"],
        }
        # specs that cause AssertionError for name not in history columns
        self.specs_fail4 = {
            "history": os.path.join(
                os.path.dirname(__file__), "..", "storage_data.csv"
            ),
            "name": "taco",
        }
        # specs that instantiate correctly and can be used for gen_reward
        self.specs_good = {
            "history": os.path.join(
                os.path.dirname(__file__), "..", "data", "storage_data.csv"
            ),
            "name": "LMP",
        }

    def test_instantiation(self):
        """
        Tests that StaticHistoricalForecast instantiates correctly with required attributes.
        """

        required_attributes = ["history", "name"]
        required_type = [pd.DataFrame, str]

        obj = StaticHistoricalForecast(**self.specs_good)
        # check that required attributes exist and are the correct type
        for att, typ in zip(required_attributes, required_type):
            self.assertTrue(
                hasattr(obj, att),
                f"Required attribute {att} missing for StaticHistoricalForecast.",
            )
            self.assertTrue(
                isinstance(getattr(obj, att), typ), f"{att} should be {typ}."
            )

    def test_history_AssertionError(self):
        """
        Tests that AssertionError thrown when history specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            StaticHistoricalForecast,
            **self.specs_fail1,
            msg="StaticHistoricalForecast should have AssertionError when history is not str.",
        )
        self.assertRaises(
            AssertionError,
            StaticHistoricalForecast,
            **self.specs_fail2,
            msg="StaticHistoricalForecast should have AssertionError when history file does not exist.",
        )

    def test_name_AssertionError(self):
        """
        Tests that AssertionError thrown when name specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            StaticHistoricalForecast,
            **self.specs_fail3,
            msg="StaticHistoricalForecast should have AssertionError when name is not str.",
        )
        self.assertRaises(
            AssertionError,
            StaticHistoricalForecast,
            **self.specs_fail4,
            msg="StaticHistoricalForecast should have AssertionError when name is not in history columns.",
        )

    def test_StaticHistoricalForecast_gen_reward(self):
        """
        Tests that StaticHistoricalForecast gen_reward functions correctly.
        """

        # history dataframe to check against
        history = pd.read_csv(self.specs_good["history"])

        obj = StaticHistoricalForecast(**self.specs_good)
        for i in range(2):
            # check that gen_reward returns correct type and length
            reward = obj.gen_reward()
            self.assertTrue(
                isinstance(reward, np.ndarray),
                "StaticHistoricalForecast gen_reward must return numpy.ndarray.",
            )
            self.assertEqual(
                len(reward),
                obj.n,
                "StaticHistoricalForecast gen_reward must return numpy.ndarray of length n.",
            )
            # check that values are correct
            end = i + obj.n
            correct_reward = history[self.specs_good["name"]].values[i:end]
            self.assertTrue(
                all(correct_reward == reward),
                "StaticHistoricalForecast gen_reward calculated incorrectly.",
            )
            # check that counter i is incremented
            self.assertEqual(
                i + 1,
                obj.i,
                "StaticHistoricalForecast does not increment counter i correctly.",
            )

        # check that a ValueError is raised if samples are requested beyond historical data
        obj.i = 10000
        self.assertRaises(ValueError, obj.gen_reward)
