# -*- coding: utf-8 -*-
"""
Data types provided by plugin

Register data types via the "aiida.data" entry point in setup.json.
"""

from .battery import BatterySample, BatteryState
from .experiment import DummyExperimentSpecs
