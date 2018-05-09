import unittest
import numpy as np

from analysis_engine.node import (
    Attribute, A, App, ApproachItem, KeyPointValue, KPV,
    KeyTimeInstance, KTI, M, Parameter, P, Section, S
)

from analysis_engine.helicopter.derived_parameters import (
    ApproachRange,
    AltitudeADH,
)


class TestApproachRange(unittest.TestCase):
    def test_can_operate(self):
        operational_combinations = ApproachRange.get_operational_combinations()
        self.assertTrue(('Altitude AAL', 'Latitude Smoothed',
                         'Longitude Smoothed', 'Touchdown') in operational_combinations,
                        msg="Missing 'helicopter' combination")

    def test_derive(self):
        d = 1.0/60.0
        lat = P('Latitude', array=[0.0, d/2.0, d])
        lon = P('Longitude', array=[0.0, 0.0, 0.0])
        alt = P('Altitude AAL', array=[200, 100, 0.0])
        tdn = KTI('Touchdown', items=[KeyTimeInstance(2, 'Touchdown'),])
        ar = ApproachRange()
        ar.derive(alt, lat, lon, tdn)
        result = ar.array
        # Strictly, 1nm is 1852m, but this error arises from the haversine function.
        self.assertEqual(int(result[0]), 1853)


class TestAltitudeADH(unittest.TestCase):
    def test_can_operate(self):
        opts = AltitudeADH.get_operational_combinations()
        self.assertEqual(opts, [('Altitude Radio', 'Vertical Speed')])

    def test_adh_basic(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([z, z[:150:-1], z[50::-1], z[:50], z[150:], z[::-1]]))
        hdot = P('Vertical Speed', np.ma.array([60]*200+[-60]*100+[60]*100+[-60]*200))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(height.array[210], 189.0)
        self.assertEqual(adh.array[210], 89.0)

    def test_adh_no_rig(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([z,z[::-1]]))
        hdot = P('Vertical Speed', np.ma.array([60]*200+[-60]*200))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(np.ma.count(adh.array), 0)

    def test_adh_two_rigs(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([
            z, z[:150:-1], z[50::-1], z[:50], z[150:], z[:150:-1], z[100::-1], z[:100], z[150:], z[::-1]]))
        hdot = P('Vertical Speed', np.ma.array(
            [60]*200 + [-60]*100 + [60]*100 + [-60]*150 + [60]*150 + [-60]*200))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        self.assertEqual(height.array[210]-adh.array[210], 100.0)
        self.assertEqual(height.array[680]-adh.array[680], 50.0)
        
    def test_frequency(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([z, z[:150:-1], z[50::-1], z[:50], z[150:], z[::-1]]),
                   frequency=4.0)
        hdot = P('Vertical Speed', np.ma.array([60]*200+[-60]*100+[60]*100+[-60]*200),
                 frequency=4.0)
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(height.array[210], 189.0)
        self.assertEqual(adh.array[210], 88.25)
