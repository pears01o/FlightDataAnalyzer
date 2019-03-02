import unittest
import numpy as np

from analysis_engine.node import (
    A, M, P, aeroplane, helicopter
)

from analysis_engine.helicopter.flight_phase import (
    Airborne,
    Autorotation,
    Hover,
    HoverTaxi,
    RotorsTurning,
    Takeoff,
    OnDeck
)

from analysis_engine.test_utils import buildsection, buildsections


class TestAirborne(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        node = Airborne
        available = ('Altitude AAL For Flight Phases', 'Fast')
        self.assertFalse(node.can_operate(available,
                                          seg_type=A('Segment Type', 'START_AND_STOP')))
        available = ('Altitude Radio', 'Altitude AGL', 'Gear On Ground', 'Rotors Turning')
        self.assertTrue(node.can_operate(available,
                                         seg_type=A('Segment Type', 'START_AND_STOP')))

    def test_airborne_helicopter_basic(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*30+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([2.0]*4+[0.0]*3+[20.0]*30+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 3.5)
        self.assertEqual(node[0].slice.stop, 36.95)

    def test_airborne_helicopter_short(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*10+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([2.0, 0.0, 0.0]+[0.0]*4+[20.0]*10+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 1)

    def test_airborne_helicopter_radio_refinement(self):
        '''
        Confirms that the beginning and end are trimmed to match the radio signal,
        not the (smoothed) AGL data.
        '''
        gog = M(name='Gear On Ground',
                array=np.ma.array([0]*3+[1]*5+[0]*10+[1]*5, dtype=int),
                frequency=1,
                offset=0,
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0.0]*6+[20.0]*12+[0.0]*5, dtype=float))
        rad = P(name='Altitude Radio',
                array=np.ma.array([0.0]*7+[10.0]*10+[0.0]*6, dtype=float))
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(rad, agl, gog, rtr)
        self.assertEqual(node[0].start_edge, 6.1)
        self.assertEqual(node[0].stop_edge, 16.9)

    def test_airborne_helicopter_overlap(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 2, 0], dtype=float),
                frequency=0.2)
        rtr = buildsection('Rotors Turning', 0, 40)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 3.2)
        self.assertEqual(node[0].slice.stop, 6)
        self.assertEqual(node[1].slice.start, 8)
        self.assertEqual(node[1].slice.stop, 10.5)

    def test_airborne_helicopter_cant_fly_without_rotor_turning(self):
        gog = M(name='Gear On Ground',
                array=np.ma.array([1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1], dtype=int),
                values_mapping={1:'Ground', 0:'Air'})
        agl = P(name='Altitude AGL',
                array=np.ma.array([0, 0, 0, 0, 5, 5, 5, 5, 5, 5, 2, 0], dtype=float),
                frequency=0.2)
        rtr = buildsection('Rotors Turning', 0, 0)
        node = Airborne()
        node.derive(agl, agl, gog, rtr)
        self.assertEqual(len(node), 0)


class TestAutorotation(unittest.TestCase):

    def setUp(self):
        self.node_class = Autorotation

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Eng (*) N2 Max', 'Nr', 'Descending'),
                                                    ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Eng (*) N2 Max', 'Nr', 'Descending'),
                                                     ac_type=aeroplane))

    @unittest.SkipTest
    def test_derive(self):
        self.assertTrue(False)

    def test_derive_no_auto(self):
        descs = buildsections('Descending', [4,13], [15,27])
        eng_np = P(name='Eng (*) Np Max', array=np.ma.ones(30) * 100)
        nr = P(name= 'Nr', array=np.ma.ones(30) * 100)
        node = Autorotation()
        node.derive(eng_np , nr, descs)
        self.assertEqual(len(node), 0)

    def test_derive_one_auto(self):
        descs = buildsections('Descending', [4,13], [15,27])
        eng_np = P(name='Eng (*) Np Max', array=np.ma.ones(30) * 100)
        nr = P(name= 'Nr', array=np.ma.ones(30) * 100)
        nr.array[16:19] = 105
        nr.array[22:26] = 102
        node = Autorotation()
        node.derive(eng_np , nr, descs)
        self.assertEqual(len(node), 1)

    def test_derive_two_autos(self):
        descs = buildsections('Descending', [4,13], [15,27])
        eng_np = P(name='Eng (*) Np Max', array=np.ma.ones(30) * 100)
        nr = P(name= 'Nr', array=np.ma.ones(30) * 100)
        nr.array[6:10] = 105
        nr.array[18:20] = 102
        node = Autorotation()
        node.derive(eng_np , nr, descs)
        self.assertEqual(len(node), 2)


class TestHover(unittest.TestCase):

    def setUp(self):
        self.node_class = Hover

    def test_can_operate(self):
        available = ('Altitude AGL', 'Airborne', 'Groundspeed')
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(available, ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(available, ac_type=aeroplane))

    def test_derive_basic(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.concatenate((np.zeros(5), np.ones(30) * 10, np.zeros(5))))
        gspd = P('Groundspeed', array=np.ma.zeros(40))
        airs = buildsections('Airborne', [6, 26])
        t_hf = buildsections('Transition Hover To Flight', [22, 24])
        t_fh = buildsections('Transition Flight To Hover', [8, 10])
        node = Hover()
        node.derive(alt_agl, airs, gspd, t_hf, t_fh)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].slice.start, 11)
        self.assertEqual(node[0].slice.stop, 22)

    def test_derive_null_transitions(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.concatenate((np.zeros(5), np.ones(10) * 10.0, np.zeros(5))))
        gspd = P('Groundspeed', array=np.ma.zeros(20))
        airs = buildsections('Airborne', [6, 16])
        node = Hover()
        node.derive(alt_agl, airs, gspd, None, None)
        self.assertEqual(len(node), 1)

    def test_derive_too_high(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.ones(30) * 310)
        gspd = P('Groundspeed', array=np.ma.zeros(30))
        airs = buildsections('Airborne', [1,13], [15,27])
        node = Hover()
        node.derive(alt_agl, airs, gspd)
        self.assertEqual(len(node), 0)

    def test_derive_too_short(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.zeros(30))
        gspd = P('Groundspeed', array=np.ma.zeros(30))
        airs = buildsections('Airborne', [6,8], [15,27])
        node = Hover()
        node.derive(alt_agl, airs, gspd)
        self.assertEqual(len(node), 1)

    def test_derive_not_without_transition(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.concatenate((np.zeros(5), np.ones(45) * 10, np.ones(50) * 400, np.ones(50) * 250, np.ones(30) * 400, np.zeros(20))))
        gspd = P('Groundspeed', array=np.ma.zeros(200))
        airs = buildsections('Airborne', [6, 200])
        t_hf = buildsections('Transition Hover To Flight', [22, 24])
        t_fh = buildsections('Transition Flight To Hover', [180, 190])
        node = Hover()
        node.derive(alt_agl, airs, gspd, t_hf, t_fh)
        self.assertEqual(len(node), 2)

    def test_derive_not_dip(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.concatenate((np.ones(10) * 310, np.ones(10) * 290, np.ones(10) * 310)))
        gspd = P('Groundspeed', array=np.ma.zeros(30))
        airs = buildsections('Airborne', [0, 30])
        t_hf = buildsections('Transition Hover To Flight', [2, 4])
        t_fh = buildsections('Transition Flight To Hover', [28, 30])
        node = Hover()
        node.derive(alt_agl, airs, gspd, t_hf, t_fh)
        self.assertEqual(len(node), 0)

    def test_derive_too_fast(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.ones(30) * 10, frequency=0.2)
        gspd = P('Groundspeed', array=np.ma.concatenate((np.zeros(10), np.ones(10) * 20, np.zeros(10))))
        airs = buildsections('Airborne', [0, 30])
        t_hf = buildsections('Transition Hover To Flight', [2, 4])
        t_fh = buildsections('Transition Flight To Hover', [28, 30])
        node = Hover()
        node.derive(alt_agl, airs, gspd)
        self.assertEqual(len(node), 2)


class TestHoverTaxi(unittest.TestCase):

    def setUp(self):
        self.node_class = HoverTaxi

    def test_can_operate(self):
        available = ('Altitude AGL', 'Airborne', 'Hover')
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(available, ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(available, ac_type=aeroplane))

    def test_derive_basic(self):
        alt_agl = P(name='Altitude AGL', array=np.ma.concatenate((np.zeros(5), np.ones(30) * 10.0, np.zeros(5))))
        alt_agl.array[14] = 20.0
        alt_agl.array[17] = 60.0
        hovers = buildsections('Hover', [6, 8], [24,26])
        airs = buildsections('Airborne', [6, 26])
        t_hf = buildsections('Transition Hover To Flight', [12, 15])
        t_fh = buildsections('Transition Flight To Hover', [18, 20])
        node = HoverTaxi()
        node.derive(alt_agl, airs, hovers, t_hf, t_fh)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 9)
        self.assertEqual(node[0].slice.stop, 12)
        self.assertEqual(node[1].slice.start, 21)
        self.assertEqual(node[1].slice.stop, 24)


class TestRotorsTurning(unittest.TestCase):
    def setUp(self):
        self.node_class = RotorsTurning

    def test_can_operate(self):
        self.assertTrue(self.node_class.can_operate(('Rotors Running'),
                                                    ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Rotors Running'),
                                                     ac_type=aeroplane))

    def test_derive_basic(self):
        running = M('Rotors Running', np.ma.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
                    values_mapping={0: 'Not Running', 1: 'Running',})
        node=RotorsTurning()
        node.derive(running)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].slice.start, 1)
        self.assertEqual(node[0].slice.stop, 4)
        self.assertEqual(node[1].slice.start, 7)
        self.assertEqual(node[1].slice.stop, 11)


class TestTakeoff(unittest.TestCase):
    def test_can_operate(self):
        # Airborne dependency added to avoid trying to derive takeoff when
        # aircraft's dependency
        available = ('Heading Continuous', 'Altitude AAL For Flight Phases',
                     'Fast', 'Airborne')
        seg_type = A('Segment Type', 'START_AND_STOP')
        #seg_type.value = 'START_ONLY'
        self.assertFalse(Takeoff.can_operate(available, seg_type=seg_type))
        available = ('Altitude AGL', 'Liftoff')
        self.assertTrue(Takeoff.can_operate(available, seg_type=seg_type))

    # TODO: Create testcases for helicopters. All testcases covered aeroplanes
    @unittest.skip('No helicopter testcases prior to split.')
    def test_derive(self):
        self.assertTrue(False, msg='Test not implemented.')


class TestOnDeck(unittest.TestCase):

    def setUp(self):
        self.null = np.array([0.0]*100)
        self.wave = np.sin(np.array(range(100))/3.0)
        self.gnds = buildsections('Grounded', [10, 89])

    def test_can_operate(self):
        self.assertTrue(OnDeck.can_operate(('Grounded', 'Pitch', 'Roll'), ac_type=helicopter))

    def test_basic(self):
        pitch = P('Pitch', self.wave * 2.0)
        roll = P('Roll', self.null)
        phase = OnDeck()
        phase.derive(self.gnds, pitch, roll)
        self.assertEqual(phase.name,'On Deck')
        self.assertEqual(phase.get_first().slice, slice(10, 90))

    def test_roll(self):
        pitch = P('Pitch', self.null)
        roll = P('Roll', self.wave * 2.0)
        phase = OnDeck()
        phase.derive(self.gnds, pitch, roll)
        self.assertEqual(phase.get_first().slice, slice(10, 90))

    def test_roll_and_pitch(self):
        pitch = P('Pitch', self.wave)
        roll = P('Roll', self.wave)
        phase = OnDeck()
        phase.derive(self.gnds, pitch, roll)
        self.assertEqual(phase.get_first().slice, slice(10, 90))

    def test_still_on_ground(self):
        pitch = P('Pitch', self.null)
        roll = P('Roll', self.null)
        phase = OnDeck()
        phase.derive(self.gnds, pitch, roll)
        self.assertEqual(phase.get_first(), None)
