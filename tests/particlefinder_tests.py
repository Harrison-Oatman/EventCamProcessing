#author = Joanna Van Liew

import sys
import types

#Create a fake metavision_core package to run tests without having to download the metavision_core package on your own system"""
metavision_core = types.ModuleType("metavision_core")
event_io = types.ModuleType("metavision_core.event_io")

# Fake EventsIterator inside event_io
event_io.EventsIterator = object

# Wire package structure together
metavision_core.event_io = event_io

# Register in sys.modules
sys.modules["metavision_core"] = metavision_core
sys.modules["metavision_core.event_io"] = event_io

import numpy as np
import scripts.particlefinder as pf

from skimage.measure import label, regionprops

pf.label = label
pf.regionprops = regionprops

from conftest import array_events

def test_ev_particlefinder_cluster():
    """
    Test that a cluster of dense events can be detected as at least one particle.

    This test starts with a small arbitrary cluster of same-polarity events and passes them
    to the particle finder. The main outcome of this test is that it verifies that one or more
    particles are detected and the output is (x, y, t) that satisfies the minimum area requirement
    as expected.
    """
    events = [(50+i%3, 60+i//3, 1000, 1) for i in range(9)]
    arr = array_events(events)
    particles = pf.ev_particlefinder(arr, min_area=5, h=128, w=128)
    assert len(particles) >=1
    assert 'x' in particles.dtype.names
    assert 'y' in particles.dtype.names
    assert 't' in particles.dtype.names
    assert particles['area'][0] >=5

def test_min_area_particlefinder():
    """
    Tests that clusters smaller than the minimum area are not detected.

    After creating a small arbitrary cluster of events that is smaller than the minimum area,
    the particle finder code is tested to ensure that no particles are returned in this case.
    """
    events = [(10,10,0,1), (11,10,0,1)]
    arr = array_events(events)
    particles = pf.ev_particlefinder(arr, min_area=3, h=128, w=128)
    assert len(particles) ==0
