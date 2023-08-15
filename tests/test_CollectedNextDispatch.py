import unittest
import os
import yaml
from yaml import Loader
import pandas as pd
from ORCA.Optimization.LTIStateSpaceMPCPyomoOptimization import (
    LTIStateSpaceMPCPyomoOptimization as LS,
)
from ORCA.CollectedNextDispatch import CollectedNextDispatch
from .data.SamplePKLFile import generate_matrices_pkl_from_csv


class TestORCA(unittest.TestCase):
    """
    ORCA tests.

    This tests instantiation and all other methods belonging to ORCA.

    """

    def setUp(self):
        YAML_template_spec = os.path.join(
            os.path.dirname(__file__), "data", "test_template.yaml"
        )
        # ensure ABC.pkl file is in YAMLspec
        generate_matrices_pkl_from_csv()
        with open(YAML_template_spec, "r") as f:
            tmp_spec = yaml.load(f, Loader=Loader)
        matrices_path = os.path.join(os.path.dirname(__file__), "data", "ABC.pkl")
        tmp_spec["optimization"]["matrices"] = matrices_path
        self.YAMLspec = os.path.join(os.path.dirname(__file__), "data", "test.yaml")
        with open(self.YAMLspec, "w") as f:
            yaml.dump(tmp_spec, f)

    def tearDown(self):
        os.remove(os.path.join(os.path.dirname(__file__), "data", "test.yaml"))
        os.remove(os.path.join(os.path.dirname(__file__), "data", "ABC.pkl"))

    def test_instantiation(self):
        """
        Tests that ORCA instantiates correctly with attributes.

        """

        required_attributes = [
            "specs",
            "optimization",
            "reward",
            "initial_states",
            "optimal_results",
        ]
        required_type = [
            dict,
            LS,
            dict,
            type(None),
            type(None),
        ]

        obj = CollectedNextDispatch(self.YAMLspec)
        for att, typ in zip(required_attributes, required_type):
            self.assertTrue(
                hasattr(obj, att), f"Required attribute {att} missing for ORCA."
            )
            self.assertTrue(
                isinstance(getattr(obj, att), typ), f"{att} should be {typ}"
            )

    def test_return_optimal_next_dispatch(self):
        """
        Test functionality of return_optimal_next_dispatch.

        """

        obj = CollectedNextDispatch(self.YAMLspec)
        time = pd.to_datetime("2022-05-31 00:05:00")
        x_init = [50.0, 0.0]
        next_dispatch = obj.return_optimal_next_dispatch(time, x_init)

        history_path = os.path.join(
            os.path.dirname(__file__), "data", "storage_data.csv"
        )
        vals = pd.read_csv(history_path)
        needed_keys = ["states", "control", "measurements"]
        for key in needed_keys:
            self.assertTrue(
                key in next_dispatch,
                f"{key} missing in next_dispatch for ORCA.",
            )
            self.assertTrue(
                isinstance(next_dispatch[key], list),
                f"{key} in next_dispatch for ORCA value should be list.",
            )
            self.assertEqual(
                len(next_dispatch[key]),
                len(obj.specs["optimization"][key]["order"]),
                f"{key} in next_dispatch for ORCA returns wrong number of entries.",
            )
            # check that values are close to what they should be
            for i in range(len(obj.specs["optimization"][key]["order"])):
                self.assertAlmostEqual(
                    next_dispatch[key][i],
                    vals[obj.specs["optimization"][key]["order"][i]].values[1],
                )

    def test_reset_objects(self):
        """
        Test functionality of reset_objects.

        """

        obj = CollectedNextDispatch(self.YAMLspec)

        # run return_optimal_next_dispatch to generate some results
        time = pd.to_datetime("2022-05-31 00:05:00")
        x_init = [50.0, 0.0]
        next_dispatch = obj.return_optimal_next_dispatch(time, x_init)

        # run reset_objects and check if it worked correctly
        obj.reset_objects()

        # check if counters are set back to 0
        for key in obj.reward:
            self.assertEqual(
                obj.reward[key].i, 0, f"Reward counter for {key} not reset to 0."
            )

        # check if initial_states and optimal_results are set back to None
        self.assertTrue(
            obj.initial_states is None,
            "ORCA reset_objects did not set initial_states to None.",
        )
        self.assertTrue(
            obj.optimal_results is None,
            "ORCA reset_objects did not set optimal_results to None.",
        )
