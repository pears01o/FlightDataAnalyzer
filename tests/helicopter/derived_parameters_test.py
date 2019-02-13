import unittest
import numpy as np

from numpy.ma.testutils import assert_array_equal

from analysis_engine.node import (
    M, P, KTI, aeroplane,
    helicopter, KeyTimeInstance,
)

from analysis_engine.library import unique_values

from analysis_engine.helicopter.derived_parameters import (
    ApproachRange,
    AltitudeADH,
    AltitudeAGL,
    AltitudeDensity,
    CyclicAngle,
    CyclicForeAft,
    CyclicLateral,
    MGBOilTemp,
    MGBOilPress,
    Nr,
    TorqueAsymmetry
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
        hdot = P('Vertical Speed', np.concatenate((np.ones(200) * 60, np.ones(100) * -60, np.ones(100) * 60, np.ones(200) * -60)))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(height.array[210], 189.0)
        self.assertEqual(adh.array[210], 89.0)

    def test_adh_no_rig(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([z,z[::-1]]))
        hdot = P('Vertical Speed', np.ma.concatenate((np.ones(200) * 60, np.ones(200) * -60)))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(np.ma.count(adh.array), 0)

    def test_adh_two_rigs(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([
            z, z[:150:-1], z[50::-1], z[:50], z[150:], z[:150:-1], z[100::-1], z[:100], z[150:], z[::-1]]))
        hdot = P('Vertical Speed', np.ma.concatenate((
            np.ones(200) * 60, np.ones(100) * -60, np.ones(100) * 60, np.ones(150) * -60, np.ones(150) * 60, np.ones(200) * -60)))
        adh = AltitudeADH()
        adh.derive(height, hdot)
        self.assertEqual(height.array[210]-adh.array[210], 100.0)
        self.assertEqual(height.array[680]-adh.array[680], 50.0)

    def test_frequency(self):
        z = np.ma.arange(200)
        height = P('Altitude Radio', np.ma.concatenate([z, z[:150:-1], z[50::-1], z[:50], z[150:], z[::-1]]),
                   frequency=4.0)
        hdot = P('Vertical Speed', np.ma.concatenate((np.ones(200) * 60, np.ones(100) * -60, np.ones(100) * 60, np.ones(200) * -60)),
                 frequency=4.0)
        adh = AltitudeADH()
        adh.derive(height, hdot)
        # We confirm that the radio height was 100ft higher than the height above the deck.
        self.assertEqual(height.array[210], 189.0)
        self.assertEqual(adh.array[210], 88.25)


class TestAltitudeAGL(unittest.TestCase):

    def test_basic(self):
        # Although "basic" the synthetic radio altitude tests for noise rejection, transfer from radio to pressure altimetry and use of
        # the gear on ground signals. The wide tolerance is because the noise signal varies from run to run.
        alt_rad = P(name='Altitude Radio', array=(np.minimum(6000,
                                                             (1.0-np.cos(np.arange(100)*3.14/50))*4000 + np.random.rand(100)*300)),
                                           frequency = 2)
        alt_baro = P(name='Altitude STD', array=np.ma.array((1.0-np.cos(np.arange(100)*3.14/50))*4000 + 1000))
        gog = M(name='Gear On Ground', array=np.ma.array([1]*5+[0]*90+[1]*5), values_mapping={0:'Air', 1:'Ground'})
        alt_aal = AltitudeAGL()
        alt_aal.derive(alt_rad, None, alt_baro, gog)
        self.assertLess(abs(np.max(alt_aal.array)-8000), 300)

    def test_negative(self):
        alt_rad = P(name='Altitude Radio', array=np.ma.array([-1, 0, 0, 0, -1]))
        alt_baro = P(name='Altitude STD', array=np.ma.array([0]*5))
        gog = M(name='Gear On Ground', array=np.ma.array([0]*5), values_mapping={0:'Air', 1:'Ground'})
        alt_aal = AltitudeAGL()
        alt_aal.derive(alt_rad, None, alt_baro, gog)
        expected = [0, 0, 0, 0, 0]
        assert_array_equal(alt_aal.array, expected)

    def test_on_ground(self):
        alt_rad = P(name='Altitude Radio', array=np.ma.array([-1, 0, 6, 0, -1, -1, 0, 6, 0, -1, -1, 0, 6, 0, -1, -1, 0, 6, 0, -1]))
        alt_baro = P(name='Altitude STD', array=np.ma.array([0]*20))
        gog = M(name='Gear On Ground', array=np.ma.array([1]*20), values_mapping={0:'Air', 1:'Ground'})
        alt_aal = AltitudeAGL()
        alt_aal.derive(alt_rad, None, alt_baro, gog)
        expected = [0]*20
        assert_array_equal(alt_aal.array, expected)


class TestAltitudeDensity(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeDensity

    def test_can_operate(self):
        available = ('Altitude STD', 'SAT', 'SAT International Standard Atmosphere')
        self.assertTrue(self.node_class.can_operate(available))

    def test_derive(self):

        alt_std = P('Altitude STD', array=np.ma.array([12000, 15000, 3000, 5000, 3500, 2000]))
        sat = P('SAT', array=np.ma.array([0, -45, 35, 40, 32, 8]))
        isa = P('SAT International Standard Atmosphere', array=np.ma.array([-9, -15, 9, 5, 8, 11]))

        node = self.node_class()
        node.derive(alt_std, sat, isa)

        expected = [13080, 11400, 6120, 9200, 6380, 1640]
        assert_array_equal(node.array, expected)


class TestCyclicForeAft(unittest.TestCase):

    def setUp(self):
        self.node_class = CyclicForeAft

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Fore-Aft (1)',),
                                ('Cyclic Fore-Aft (2)',),
                                ('Cyclic Fore-Aft (1)', 'Cyclic Fore-Aft (2)')])

    @unittest.SkipTest
    def test_derive(self):
        self.assertTrue(False)


class TestCyclicLateral(unittest.TestCase):

    def setUp(self):
        self.node_class = CyclicLateral

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Lateral (1)',),
                                ('Cyclic Lateral (2)',),
                                ('Cyclic Lateral (1)', 'Cyclic Lateral (2)')])

    @unittest.SkipTest
    def test_derive(self):
        self.assertTrue(False)


class TestCyclicAngle(unittest.TestCase):

    def setUp(self):
        self.node_class = CyclicAngle

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Fore-Aft', 'Cyclic Lateral')])

    @unittest.SkipTest
    def test_derive(self):
        pitch_array = np.ma.arange(20)
        roll_array = pitch_array[::-1]
        pitch = P('Cyclic Fore-Aft', pitch_array)
        roll = P('Cyclic Lateral', roll_array)
        node = self.node_class()
        node.derive(pitch, roll)

        expected_array = np.ma.sqrt(pitch_array ** 2 + roll_array ** 2)
        assert_array_equal(node.array, expected_array)


class TestMGBOilTemp(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilTemp

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate('MGB Oil Temp (1)',
                                                    ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate('MGB Oil Temp (2)',
                                                    ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('MGB Oil Temp (1)',
                                                     'MGB Oil Temp (2)'),
                                                    ac_type=helicopter))

        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEquals(len(opts), 3)

    def test_derive(self):
        t1 = [78.0]*14 + [78.5,78] + [78.5]*23 + [79.0]*5 + [79.5] + [79.0]*5
        t2 = [78.0]*13 + [78.5,78.0, 77.5] + [78.5]*20 + [79.0]*14

        oil_temp1 = P('MGB Oil Temp (1)', np.ma.array(t1))
        oil_temp2 = P('MGB Oil Temp (2)', np.ma.array(t2))

        mgb_oil_temp = self.node_class()
        mgb_oil_temp.derive(oil_temp1, oil_temp2)

        # array size should be double, 100
        self.assertEqual(mgb_oil_temp.array.size, (oil_temp1.array.size*2))
        self.assertEqual(mgb_oil_temp.array.size, (oil_temp2.array.size*2))
        # Frequency should be doubled 2Hz
        self.assertEqual(mgb_oil_temp.frequency, (oil_temp1.frequency*2))
        self.assertEqual(mgb_oil_temp.frequency, (oil_temp2.frequency*2))
        # Offset should remain the same as the parameter. 0.0
        self.assertEqual(mgb_oil_temp.offset, oil_temp1.offset)
        self.assertEqual(mgb_oil_temp.offset, oil_temp2.offset)
        # Element 44 will become 88 and values 79.5, 79 average to 79.25
        self.assertEqual(oil_temp1.array[44], 79.5)
        self.assertEqual(oil_temp2.array[44], 79)
        self.assertEqual(mgb_oil_temp.array[88],
                         (oil_temp1.array[44]+oil_temp2.array[44])/2)


class TestMGBOilPress(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilPress

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate('MGB Oil Press (1)',
                                                    ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate('MGB Oil Press (2)',
                                                    ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('MGB Oil Press (1)',
                                                     'MGB Oil Press (2)'),
                                                    ac_type=helicopter))

        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEquals(len(opts), 3)

    def test_derive(self):
        p1 = [26.51]*7 + [26.63] + [26.51]*7 + [26.4]*27 + [26.29] + [26.4]*7
        p2 = [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*14
        oil_press1 = P('MGB Oil Press (1)', np.ma.array(p1))
        oil_press2 = P('MGB Oil Press (2)', np.ma.array(p2))

        mgb_oil_press = self.node_class()
        mgb_oil_press.derive(oil_press1, oil_press2)

        # array size should be double, 100
        self.assertEqual(mgb_oil_press.array.size, (oil_press1.array.size*2))
        self.assertEqual(mgb_oil_press.array.size, (oil_press2.array.size*2))
        # Frequency should be doubled 2Hz
        self.assertEqual(mgb_oil_press.frequency, (oil_press1.frequency*2))
        self.assertEqual(mgb_oil_press.frequency, (oil_press2.frequency*2))
        # Offset should remain the same as the parameter. 0.0
        self.assertEqual(mgb_oil_press.offset, oil_press1.offset)
        self.assertEqual(mgb_oil_press.offset, oil_press2.offset)
        # Element 7 will become 14 and values 26.63, 26.51 average to 26.57
        self.assertEqual(oil_press1.array[7], 26.63)
        self.assertEqual(oil_press2.array[7], 26.51)
        self.assertEqual(mgb_oil_press.array[14],
                         (oil_press1.array[7]+oil_press2.array[7])/2)


class TestNr(unittest.TestCase):

    def setUp(self):
        self.node_class = Nr

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertTrue(self.node_class.can_operate(('Nr (1)',), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Nr (2)',), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Nr (1)', 'Nr (2)'), ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Nr (1)', 'Nr (2)'), ac_type=aeroplane))

    def test_derive(self):
        one = P('Nr (1)', np.ma.array([100,200,300]), frequency=0.5, offset=0.0)
        two = P('Nr (2)', np.ma.array([150,250,350]), frequency=0.5, offset=1.0)

        node = self.node_class()
        node.derive(one, two)

        # Note: end samples are not 100 & 350 due to method of merging.
        assert_array_equal(node.array[1:-1], np.array([150, 200, 250, 300]))
        self.assertEqual(node.frequency, 1.0)
        self.assertEqual(node.offset, 0.0)


class TestTorqueAsymmetry(unittest.TestCase):

    def setUp(self):
        self.node_class = TorqueAsymmetry

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Eng (*) Torque Max', 'Eng (*) Torque Min'), ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Eng (*) Torque Max', 'Eng (*) Torque Min'), ac_type=aeroplane))

    def test_derive(self):
        torque_max = P('Eng (*) Torque Max', np.arange(10, 30))
        torque_min = P('Eng (*) Torque Min', np.arange(8, 28))

        node = self.node_class()
        node.derive(torque_max, torque_min)

        self.assertEqual(len(node.array), len(torque_max.array))
        uniq = unique_values(node.array.astype(int))
        # there should be all 20 values being 2 out
        self.assertEqual(uniq, {2: 20})

