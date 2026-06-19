"""Zero-amplitude tests for all time integration schemes."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from grid import Grid
from integrators import step

ALL_SCHEMES = ['FTCS','BTCS','CTCS','RK4','SI','EPI2','EPI3']

def test_zero_amplitude_all_schemes():
    g = Grid()
    for scheme in ALL_SCHEMES:
        state = g.allocate_state()
        s_old = g.allocate_state()
        s_new, _ = step(state, g, 0.02, scheme=scheme, state_old=s_old)
        for k in s_new:
            assert np.allclose(s_new[k], 0.0), f"{scheme} failed zero-amp on {k}"
