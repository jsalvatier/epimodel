"""
:code:`asymmetric_laplace.py`

Asymmetric Laplace Distribution, with location parameter 0. This is used as our NPI Effectiveness prior.

See also: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
"""
import pymc3.distributions.continuous as continuous
from pymc3.distributions import draw_values, generate_samples
import theano.tensor as tt
import numpy as np
from scipy import stats

class AsymmetricLaplace(continuous.Continuous):
    """
    Assymetric Laplace Distribution

    See also: https://en.wikipedia.org/wiki/Asymmetric_Laplace_distribution
    """

    def __init__(self, scale, symmetry, testval=0.0, *args, **kwargs):
        """
        Constructor

        :param scale: scale parameter
        :param symmetry: asymmetry parameter. Reduces to a normal laplace distribution with value 1
        """
        self.scale = tt.as_tensor_variable(scale)
        self.symmetry = tt.as_tensor_variable(symmetry)

        super().__init__(*args, **kwargs, testval=testval)

    def random(self, point=None, size=None):
        """
        Draw random samples from this distribution, using the inverse CDF method.

        :param point: not used
        :param size: size of sample to draw
        :return: Samples
        """
        
        scale, symmetry = draw_values([self.scale, self.symmetry], point=point, size=size)


        u = generate_samples(
                    stats.uniform.rvs, dist_shape=self.shape, size=size
                )
        x = - np.log((1 - u) * (1 + symmetry ** 2)) / (symmetry * scale) * (
                np.greater(u , ((symmetry ** 2) / (1 + symmetry ** 2)))) + symmetry * np.log(
            u * (1 + symmetry ** 2) / (symmetry ** 2)) / scale * (
                    np.less(u , ((symmetry ** 2) / (1 + symmetry ** 2))))


        return x

    def logp(self, value):
        """
        Compute logp.

        :param value: evaluation point
        :return: log probability at evaluation point
        """
        return tt.log(self.scale / (self.symmetry + (self.symmetry ** -1))) + (
                -value * self.scale * tt.sgn(value) * (self.symmetry ** tt.sgn(value)))
