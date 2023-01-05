import unittest
import os
import numpy as np
import pyomo
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
import pandas as pd
from ORCA.Optimization.LTIStateSpaceMPCPyomoOptimization import (
    LTIStateSpaceMPCPyomoOptimization,
)
from ..data.SamplePKLFile import generate_matrices_pkl_from_csv
from ORCA.RewardForecast.StaticHistoricalForecast import StaticHistoricalForecast


class TestLTIStateSpaceMPCPyomoOptimization(unittest.TestCase):
    """
    LTIStateSpaceMPCPyomoOptimization tests.

    This tests instantiation and all other methods belonging to StaticHistoricalForecast.

    """

    def setUp(self):
        # generate matrices .pkl file
        generate_matrices_pkl_from_csv()

        # specs for .pkl file
        self.specs = {
            "solver": "glpk",
            "matrices": os.path.join(
                os.path.dirname(__file__), "..", "data", "ABC.pkl"
            ),
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
        }

        # specs for .xml file
        self.specs2 = self.specs.copy()
        self.specs2["matrices"] = os.path.join(
            os.path.dirname(__file__), "..", "data", "RAVENDMDc.xml"
        )

    def tearDown(self):
        os.remove(os.path.join(os.path.dirname(__file__), "..", "data", "ABC.pkl"))

    def test_instantiation(self):
        """
        Tests that LTIStateSpaceMPCPyomoOptimization instantiates correctly with attributes.
        """

        required_attributes = ["solver", "A", "B", "C", "model"]
        required_type = [
            pyomo.solvers.plugins.solvers.GLPK.GLPKSHELL,
            np.ndarray,
            np.ndarray,
            np.ndarray,
            pyo.ConcreteModel,
        ]

        # implicitly tests load_state_space_matrices
        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        # implicitly tests load_matric_from_xml
        obj2 = LTIStateSpaceMPCPyomoOptimization(**self.specs2)

        # for each method, check that attributes exist and are correct type
        for att, typ in zip(required_attributes, required_type):
            self.assertTrue(
                hasattr(obj, att),
                f"Required attribute {att} missing for LTIStateSpaceMPCPyomoOptimization.",
            )
            self.assertTrue(
                isinstance(getattr(obj, att), typ), f"{att} should be {typ}."
            )
            self.assertTrue(
                hasattr(obj2, att),
                f"Required attribute {att} missing for LTIStateSpaceMPCPyomoOptimization.",
            )
            self.assertTrue(
                isinstance(getattr(obj2, att), typ), f"{att} should be {typ}."
            )

    def test_matrices_AssertionError(self):
        """
        Test that AssertionError is thrown when matrices specified incorrectly.
        """

        specs_check = self.specs.copy()
        # check that if matrices is not a valid path AssertionError is thrown
        specs_check["matrices"] = "taco"
        self.assertRaises(
            AssertionError,
            LTIStateSpaceMPCPyomoOptimization,
            **specs_check,
            msg="LTIStateSpaceMPCPyomoOptimization should have AssertionError when matrices is not a valid path.",
        )

        # check that if matrices is not .pkl or .xml AssertionError is thrown
        specs_check["matrices"] = "__init__.py"
        self.assertRaises(
            AssertionError,
            LTIStateSpaceMPCPyomoOptimization,
            **specs_check,
            msg="LTIStateSpaceMPCPyomoOptimization should have AssertionError when matrices extension is not .pkl or .xml.",
        )

    def test_x_bounds(self):
        """
        Test functionality of x_bounds.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        for i in range(2):
            x_bound = obj.x_bounds(None, i, 10)
            self.assertEqual(
                x_bound,
                (self.specs["states"]["lb"][i], self.specs["states"]["ub"][i]),
                "LTIStateSpaceMPCPyomoOptimization x_bounds does not return correct values.",
            )

    def test_u_bounds(self):
        """
        Test functionality of u_bounds.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        for i in range(2):
            u_bound = obj.u_bounds(None, i, 10)
            self.assertEqual(
                u_bound,
                (self.specs["control"]["lb"][i], self.specs["control"]["ub"][i]),
                "LTIStateSpaceMPCPyomoOptimization u_bounds does not return correct values.",
            )

    def test_y_bounds(self):
        """
        Test functionality of y_bounds.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        y_bound = obj.y_bounds(None, 0, 10)
        self.assertEqual(
            y_bound,
            (self.specs["measurements"]["lb"][0], self.specs["measurements"]["ub"][0]),
            "LTIStateSpaceMPCPyomoOptimization y_bounds does not return correct values.",
        )

    def test_initialize_reward(self):
        """
        Test functionality of initialize_reward.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        self.assertEqual(
            0.0,
            obj.initialize_reward(None, 0, 10),
            "LTIStateSpaceMPCPyomoOptimization initialize_reward does not return 0.0.",
        )

    def test_solve_model(self):
        """
        Test functionality of solve_model.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        specs_reward = {
            "history": os.path.join(
                os.path.dirname(__file__), "..", "data", "storage_data.csv"
            ),
            "name": "LMP",
        }
        rewards = StaticHistoricalForecast(**specs_reward)
        results = obj.solve_model({"price": rewards.gen_reward()}, [50.0, 0.0])

        self.assertEqual(
            results.solver.status,
            SolverStatus.ok,
            "LTIStateSpaceMPCPyomoOptimization pyomo solver status not ok.",
        )
        self.assertEqual(
            results.solver.termination_condition,
            TerminationCondition.optimal,
            "LTIStateSpaceMPCPyomoOptimization termination condition not optimal.",
        )

    def test_return_next_dispatch(self):
        """
        Test functionality of return_next_dispatch.
        """

        obj = LTIStateSpaceMPCPyomoOptimization(**self.specs)
        specs_reward = {
            "history": os.path.join(
                os.path.dirname(__file__), "..", "data", "storage_data.csv"
            ),
            "name": "LMP",
        }
        rewards = StaticHistoricalForecast(**specs_reward)

        next_dispatch = obj.return_next_dispatch(
            {"price": rewards.gen_reward()}, [50.0, 0.0]
        )
        history_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "storage_data.csv"
        )
        vals = pd.read_csv(history_path)
        needed_keys = ["states", "control", "measurements"]
        for key in needed_keys:
            self.assertTrue(
                key in next_dispatch,
                f"{key} missing in next_dispatch for LTIStateSpaceMPCPyomoOptimization.",
            )
            self.assertTrue(
                isinstance(next_dispatch[key], list),
                f"{key} in next_dispatch for LTIStateSpaceMPCPyomoOptimization value should be list.",
            )
            self.assertEqual(
                len(next_dispatch[key]),
                len(self.specs[key]["order"]),
                f"{key} in next_dispatch for LTIStateSpaceMPCPyomoOptimization returns wrong number of entries.",
            )
            # check that values are close to what they should be
            for i in range(len(self.specs[key]["order"])):
                self.assertAlmostEqual(
                    next_dispatch[key][i], vals[self.specs[key]["order"][i]].values[1]
                )
