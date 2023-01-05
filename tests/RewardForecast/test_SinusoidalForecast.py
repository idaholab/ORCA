import unittest
import numpy as np
from ORCA.RewardForecast.SinusoidalForecast import SinusoidalForecast


class TestSinusoidalForecast(unittest.TestCase):
    """
    SinusoidalForecast tests.

    This tests instantiation and all other methods belonging to SinusoidalForecast.

    """

    def setUp(self):
        # specs that should cause AssertionError for amplitude
        self.specs_fail1 = {"amplitude": 10}
        # specs that should cause AssertionError for phase
        self.specs_fail2 = {"phase": [np.pi / 4.0]}
        # specs that should cause AssertionError for frequency
        self.specs_fail3 = {"frequency": np.array([2.0 * np.pi / 144.0])}
        # specs that should cause AssertionError for offset
        self.specs_fail4 = {"offset": None}
        # specs for gen_reward test
        self.specs_gen_reward = {
            "amplitude": 5.0,
            "phase": np.pi / 3.0,
            "frequency": 2.0 * np.pi / 144.0,
            "offset": 20.0,
        }

    def test_instantiation(self):
        """
        Tests that SinusoidalForecast instantiates correctly with required attributes.
        """

        required_attributes = ["amplitude", "phase", "frequency", "offset"]
        required_type = [float, float, float, float]

        # instantiate default object
        obj = SinusoidalForecast()
        # check that required attributes exist and are correct type
        for att, typ in zip(required_attributes, required_type):
            self.assertTrue(
                hasattr(obj, att),
                f"Required attribute {att} missing for SinusoidalForecast",
            )
            self.assertTrue(
                isinstance(getattr(obj, att), typ), f"{att} should be {typ}"
            )

    def test_amplitude_AssertionError(self):
        """
        Tests that an AssertionError is thrown when amplitude is specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            SinusoidalForecast,
            **self.specs_fail1,
            msg="SinusoidalForecast should have AssertionError when 'amplitude' is not float.",
        )

    def test_phase_AssertionError(self):
        """
        Tests that an AssertionError is thrown when phase is specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            SinusoidalForecast,
            **self.specs_fail2,
            msg="SinusoidalForecast should have AssertionError when 'phase' is not float.",
        )

    def test_frequency_AssertionError(self):
        """
        Tests that an AssertionError is thrown when frequency is specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            SinusoidalForecast,
            **self.specs_fail3,
            msg="SinusoidalForecast should have AssertionError when 'frequency' is not float.",
        )

    def test_offset_AssertionError(self):
        """
        Tests that an AssertionError is thrown when offset is specified incorrectly.
        """

        self.assertRaises(
            AssertionError,
            SinusoidalForecast,
            **self.specs_fail4,
            msg="SinusoidalForecast should have AssertionError when 'offset' is not float.",
        )

    def test_SinusoidalForecast_gen_reward(self):
        """
        Tests that SinusoidalForecast gen_reward functions correctly.
        """

        obj = SinusoidalForecast(**self.specs_gen_reward)
        for i in range(2):
            reward = obj.gen_reward()
            # check that gen_reward returns the correct type and length
            self.assertTrue(
                isinstance(reward, np.ndarray),
                "SinusoidalForecast gen_reward must return numpy.ndarray.",
            )
            self.assertEqual(
                len(reward),
                obj.n,
                "SinusoidalForecast gen_reward must return numpy.ndarray of length n.",
            )
            # check that the values are correct
            x = np.arange(i, i + obj.n)
            correct_reward = self.specs_gen_reward["offset"] + self.specs_gen_reward[
                "amplitude"
            ] * np.sin(
                self.specs_gen_reward["frequency"] * x + self.specs_gen_reward["phase"]
            )
            self.assertTrue(
                all(correct_reward == reward),
                "SinusoidalForecast gen_reward calculated incorrectly.",
            )
            # check that counter i is incremented
            self.assertEqual(
                i + 1,
                obj.i,
                "SinusoidalForecast does not increment counter i correctly.",
            )
