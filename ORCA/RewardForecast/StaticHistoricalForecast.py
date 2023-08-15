import os
import pandas as pd

from ORCA.Basic.RewardForecast import RewardForecast


class StaticHistoricalForecast(RewardForecast):
    """
    Forecast reward data for use with MPC optimization.

    This method uses a static historical set of data.

    Parameters
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    history : pandas.DataFrame
        historical data
    name : str
        column name that has reward/price data

    Attributes
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    n : int
        number of steps to take in time horizon
    i : int
        index for how many times reward/price have been generated
    history : str
        path to csv file containing reward/price data
    name : str
        column name that has reward/price data

    Methods
    -------
    gen_reward()
        generates n sinusoidal reward/price values to use in MPC time horizon optimization

    """

    def __init__(self, history=None, name="LMP", **specs):
        super().__init__(**specs)

        # make sure all inputs can be used
        assert isinstance(history, str), "history must be str"
        assert os.path.isfile(history), f"{history} file could not be located."
        tmp = pd.read_csv(history)
        assert isinstance(name, str), "name must be str"
        assert name in tmp.columns, f"{name} not in history DataFrame columns."
        self.history = tmp
        self.name = name

    def gen_reward(self):
        """
        Generates reward/price data as sinusoid for n steps in time horizon

        Returns
        -------
        rewards : numpy.ndarray
            reward/price data for n steps in time horizon

        """

        # get historical samples
        end = self.i + self.n
        if end > self.history.shape[0]:
            raise ValueError("samples requested beyond historical data.")
        else:
            reward = self.history[self.name].values[self.i : end]

        # increment i
        self.i += 1

        return reward
