from dataclasses import dataclass

import numpy as np
import pytest

from rl4robot.common.gae import compute_advantage_hist
from rl4robot.types import BoolHistArray, FloatHistArray


@dataclass(frozen=True)
class _TestCase:
    next_reward_hist: FloatHistArray
    next_episode_done_hist: BoolHistArray
    value_hist: FloatHistArray
    next_value_hist: FloatHistArray
    discount_gamma: float
    gae_lambda: float
    vec_advantage_hist: FloatHistArray


test_cases = [
    _TestCase(
        next_reward_hist=np.array([10.0, 12.0, 10.0, 20.0]),
        next_episode_done_hist=np.array([False, False, False, True]),
        value_hist=np.array([1.0, 2.0, 2.0, 3.0]),
        next_value_hist=np.array([2.0, 2.0, 3.0, 1.0]),
        discount_gamma=0.99,
        gae_lambda=0.95,
        vec_advantage_hist=np.array(
            [46.093068329625, 37.33446925, 26.9585, 17.0]
        ),
    ),
    _TestCase(
        next_reward_hist=np.array([1.0, -2.1, 0.1, 0.0, 1.4, 0.1]),
        next_episode_done_hist=np.array(
            [False, True, False, False, True, False]
        ),
        value_hist=np.array([0.2, 1.0, 2.1, 0.4, -1.0, 2.0]),
        next_value_hist=np.array([1.0, 2.1, 0.4, -1.0, 2.0, 0.2]),
        discount_gamma=0.9,
        gae_lambda=1.0,
        vec_advantage_hist=np.array([-1.09, -3.1, -0.866, 0.86, 2.4, -1.72]),
    ),
]


@pytest.mark.parametrize("test_case", test_cases)
def test_compute_advantage_hist(test_case: _TestCase):
    vec_advantage_hist = compute_advantage_hist(
        test_case.next_reward_hist,
        test_case.next_episode_done_hist,
        test_case.value_hist,
        test_case.next_value_hist,
        test_case.discount_gamma,
        test_case.gae_lambda,
    )

    np.testing.assert_array_almost_equal(
        vec_advantage_hist, test_case.vec_advantage_hist
    )
