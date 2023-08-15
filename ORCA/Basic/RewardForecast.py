import numpy as np


class RewardForecast(object):
    """
    Forecast reward data for use with MPC optimization.

    This is the basic class. The only required method is the gen_reward() method.

    Parameters
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)

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

    Methods
    -------
    gen_reward()
        generates n reward/price values to use in MPC time horizon optimization

    """

    def __init__(self, t_window=60.0 * 12.0, dt=5.0, **specs):
        # get time window, time step, and number of steps to take
        assert isinstance(t_window, float), "t_window must be float."
        assert isinstance(dt, float), "dt must be float."
        self.t_window = t_window
        self.dt = dt
        self.n = int(self.t_window / self.dt)
        self.i = 0

    def gen_reward(self):
        """
        Generates reward/price data for n steps in time horizon

        Returns
        -------
        rewards : numpy.ndarray
            reward/price data for n steps in time horizon

        """

        self.i += 1

        return np.array([10.0] * self.n)
