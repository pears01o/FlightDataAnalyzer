"""
Data structures used by Analysis Engine.
"""

from analysis_engine.recordtype import recordtype

Segment = recordtype('Segment',
                     'slice type part path hash start_dt go_fast_dt stop_dt precise_timestamp timestamp_configuration',
                     default=None)
