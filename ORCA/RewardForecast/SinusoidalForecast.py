import numpy as np

from ORCA.Basic.RewardForecast import RewardForecast


class SinusoidalForecast(RewardForecast):
    """
    Forecast reward data for use with MPC optimization.

    This method generates a sinusoidal signal according to the following
    equation:

    amplitude*sin(frequency*x + phase) + offset

    x is generated as np.arange(i, i+n+1)

    Parameters
    ----------
    t_window : float
        look ahead time horizon for MPC (in minutes)
    dt : float
        constant time step (in minutes)
    amplitude : float
        amplitude of sinusoid
    phase : float
        phase of sinusoid
    frequency : float
        frequency of sinusoid w = 2*np.pi*f
    offset : float
        constant offset added to sinusoid

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
    amplitude : float
        amplitude of sinusoid
    phase : float
        phase of sinusoid
    frequency : float
        frequency of sinusoid w = 2*np.pi*f
    offset : float
        constant offset added to sinusoid

    Methods
    -------
    gen_reward()
        generates n sinusoidal reward/price values to use in MPC time horizon optimization

    """

    def __init__(
        self,
        amplitude=10.0,
        phase=np.pi / 4.0,
        frequency=2 * np.pi / 144.0,
        offset=10.0,
        **specs
    ):
        super().__init__(**specs)

        # make sure all inputs can be used
        assert isinstance(amplitude, float), "amplitude must be float."
        self.amplitude = amplitude
        assert isinstance(phase, float), "phase must be float."
        self.phase = phase
        assert isinstance(frequency, float), "frequency must be float."
        self.frequency = frequency
        assert isinstance(offset, float), "offset must be float"
        self.offset = offset

    def gen_reward(self):
        """
        Generates reward/price data as sinusoid for n steps in time horizon

        Returns
        -------
        rewards : numpy.ndarray
            reward/price data for n steps in time horizon

        """

        # generate indices for samples
        x = np.arange(self.i, self.i + self.n)
        # generate sinusoidal samples
        reward = self.offset + self.amplitude * np.sin(self.frequency * x + self.phase)
        # update i
        self.i += 1

        return reward
