"""Basic tests for grid.py"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np
from grid import Grid

def test_grid_shapes():
    g = Grid()
    state = g.allocate_state()
    for k in ['u','w','theta','pi']:
        assert state[k].shape == (g.nz, g.nx)

def test_base_state_hydrostatic():
    g = Grid()
    assert g.pi_bar[0] > g.pi_bar[-1]   # pressure decreases with height

def test_sponge_zero_below_threshold():
    g = Grid()
    z_s = 0.8 * g.Lz
    below = g.z_1d < z_s
    assert np.allclose(g.sponge[below], 0.0)
