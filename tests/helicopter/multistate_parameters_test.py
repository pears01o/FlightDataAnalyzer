import os
import unittest
import numpy as np

from analysis_engine.node import (
    Attribute, aeroplane, helicopter,
    A,
    App,
    load,
    M,
    P,
    Section,
    S,
)
from analysis_engine.library import (
    vstack_params_where_state,
)
from analysis_engine.multistate_parameters import Eng_AnyRunning
from analysis_engine.helicopter.multistate_parameters import (
    AllEnginesOperative,
    ASEEngaged,
    Eng1OneEngineInoperative,
    Eng2OneEngineInoperative,
    GearOnGround,
    OneEngineInoperative,
    RotorBrakeEngaged,
    RotorsRunning,
)

test_data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              os.pardir, 'test_data')


class TestAllEnginesOperative(unittest.TestCase):

    def setUp(self):
        self.node_class = AllEnginesOperative

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations(ac_type=helicopter)
        expected = [('Eng (*) Any Running', 'One Engine Inoperative', 'Autorotation')]
        self.assertEqual(combinations, expected)

    def test_derive(self):
        run_data = [0]*5 + [1]*30 + [0]*5
        oei_data = [0]*13 + [1]*3 + [0]*24
        oei = M(name='One Engine Inoperative', array=np.ma.array(oei_data), values_mapping=OneEngineInoperative.values_mapping)
        any_running = M(name='Eng (*) Any Running', array=np.ma.array(run_data),
                       values_mapping=Eng_AnyRunning.values_mapping)
        autorotation = []

        node = self.node_class()
        node.derive(any_running, oei, S(items=[]))

        expected_data = [0]*5 + [1]*8 + [0]*3 + [1]*19 + [0]*5
        expected = self.node_class(name='All Engines Operative', array=np.ma.array(expected_data, dtype=int),
                       values_mapping=self.node_class.values_mapping)
        np.testing.assert_array_equal(node.array, expected.array)


class TestASEEngaged(unittest.TestCase):

    def setUp(self):
        self.node_class = ASEEngaged

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        expected = [
            ('ASE (1) Engaged',),
            ('ASE (2) Engaged',),
            ('ASE (3) Engaged',),
            ('ASE (1) Engaged', 'ASE (2) Engaged'),
            ('ASE (1) Engaged', 'ASE (3) Engaged'),
            ('ASE (2) Engaged', 'ASE (3) Engaged'),
            ('ASE (1) Engaged', 'ASE (2) Engaged', 'ASE (3) Engaged'),
        ]
        self.assertEqual(opts, expected)

    def test_single_ase(self):
        ase1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                values_mapping={1:'Engaged',0:'-'},
                name='AP (1) Engaged')
        ase2 = ase3 = None

        node = self.node_class()
        node.derive(ase1, ase2, ase3)

        expected = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                     values_mapping={0: '-', 1: 'Engaged'},
                     name='AP Engaged',
                     frequency=1,
                     offset=0.1)
        np.testing.assert_array_equal(expected.array, node.array)

    def test_dual_ap(self):
        # Two result in just "Engaged" state still
        ase1 = M(array=np.ma.array(data=[0,0,1,1,0,0]),
                values_mapping={1:'Engaged',0:'-'},
                name='AP (1) Engaged')
        ase2 = M(array=np.ma.array(data=[0,0,0,1,1,0]),
                values_mapping={1:'Engaged',0:'-'},
                name='AP (2) Engaged')
        ase3 = None

        node = self.node_class()
        node.derive(ase1, ase2, ase3)

        expected = M(array=np.ma.array(data=[0,0,1,1,1,0]),
                     values_mapping={0: '-', 1: 'Engaged'},
                     name='AP Engaged',
                     frequency=1,
                     offset=0.1)

        np.testing.assert_array_equal(expected.array, node.array)


class TestEng1OneEngineInoperative(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng1OneEngineInoperative
        n2_data = [0.0]*3 + [100.0]*11 + [98.0]*3 + [100.0]*20 + [0.0]*3
        self.n2 = P(name='Eng (2) N2', array=np.ma.array(n2_data))

        nr_data = [0.0]*5 + [100.0]*30 + [0.0]*5
        self.nr = P(name='Nr', array=np.ma.array(nr_data))

        expected_data = [0]*14 + [1]*3 + [0]*23
        self.expected = self.node_class(name='Eng (1) One Engine Inoperative', array=np.ma.array(expected_data, dtype=int),
                                        values_mapping=self.node_class.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations(ac_type=helicopter)
        expected = [('Eng (2) N2', 'Nr', 'Autorotation')]
        self.assertEqual(combinations, expected)

    def test_derive(self):
        node = self.node_class()
        node.derive(self.n2, self.nr, S(items=[]))

        np.testing.assert_array_equal(node.array, self.expected.array)

    def test_derive_mask(self):
        self.n2.array.mask = np.ma.getmaskarray(self.n2.array)
        self.n2.array.mask[20:25] = True

        node = self.node_class()
        node.derive(self.n2, self.nr, S(items=[]))

        np.testing.assert_array_equal(node.array, self.expected.array)


class TestEng2OneEngineInoperative(unittest.TestCase):

    def setUp(self):
        self.node_class = Eng2OneEngineInoperative
        n2_data = [0.0]*3 + [100.0]*11 + [98.0]*3 + [100.0]*20 + [0.0]*3
        self.n2 = P(name='Eng (1) N2', array=np.ma.array(n2_data))

        nr_data = [0.0]*5 + [100.0]*30 + [0.0]*5
        self.nr = P(name='Nr', array=np.ma.array(nr_data))

        expected_data = [0]*14 + [1]*3 + [0]*23
        self.expected = self.node_class(name='Eng (2) One Engine Inoperative', array=np.ma.array(expected_data, dtype=int),
                                        values_mapping=self.node_class.values_mapping)

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations(ac_type=helicopter)
        expected = [('Eng (1) N2', 'Nr', 'Autorotation')]
        self.assertEqual(combinations, expected)

    def test_derive(self):
        node = self.node_class()
        node.derive(self.n2, self.nr, S(items=[]))

        np.testing.assert_array_equal(node.array, self.expected.array)

    def test_derive_mask(self):
        self.n2.array.mask = np.ma.getmaskarray(self.n2.array)
        self.n2.array.mask[20:25] = True

        node = self.node_class()
        node.derive(self.n2, self.nr, S(items=[]))

        np.testing.assert_array_equal(node.array, self.expected.array)


class TestGearOnGround(unittest.TestCase):
    def setUp(self):
        self.node_class = GearOnGround

    def test_can_operate(self):
        helicopter_expected = ('Vertical Speed', 'Eng (*) Torque Avg')
        opts = self.node_class.get_operational_combinations()
        self.assertTrue(helicopter_expected in opts)

    def test_derive__columbia234(self):
        vert_spd = load(os.path.join(test_data_path, "gear_on_ground__columbia234_vert_spd.nod"))
        torque = load(os.path.join(test_data_path, "gear_on_ground__columbia234_torque.nod"))
        collective = load(os.path.join(test_data_path,"gear_on_ground__columbia234_collective.nod"))
        ac_series = A("Series", value="Columbia 234")
        wow = GearOnGround()
        wow.derive(None, None, vert_spd, torque, ac_series, collective)
        self.assertTrue(np.ma.all(wow.array[:252] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[254:540] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1040:1200] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1420:1440] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[1533:1550] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[1615:1622] == 'Air'))
        #self.assertTrue(np.ma.all(wow.array[1696:1730] == 'Ground'))
        self.assertTrue(np.ma.all(wow.array[1900:2150] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2350:2385] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2550:2750] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[2900:3020] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[3366:3376] == 'Air'))
        self.assertTrue(np.ma.all(wow.array[3425:] == 'Ground'))

    def test_derive__columbia234_collective(self):
        vert_spd = load(os.path.join(test_data_path, "gear_on_ground__columbia234_vert_spd_flight2.nod"))
        torque = load(os.path.join(test_data_path, "gear_on_ground__columbia234_torque_flight2.nod"))
        collective = load(os.path.join(test_data_path, "gear_on_ground__columbia234_collective_flight2.nod"))
        ac_series = A("Series", value="Columbia 234")
        wow = GearOnGround()
        wow.derive(None, None, vert_spd, torque, ac_series, collective)
        self.assertTrue(all(wow.array[:277] == 'Ground'))
        self.assertTrue(all(wow.array[300:1272] == 'Air'))
        self.assertTrue(all(wow.array[1275:1470] == 'Ground'))
        self.assertTrue(all(wow.array[1474:1772] == 'Air'))
        self.assertTrue(all(wow.array[1775:1803] == 'Ground'))
        self.assertTrue(all(wow.array[1806:2107] == 'Air'))
        self.assertTrue(all(wow.array[2109:2200] == 'Ground'))
        self.assertTrue(all(wow.array[2203:3894] == 'Air'))
        self.assertTrue(all(wow.array[3896:] == 'Ground'))


class TestOneEngineInoperative(unittest.TestCase):

    def setUp(self):
        self.node_class = OneEngineInoperative

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations(ac_type=helicopter)
        expected = [('Eng (1) One Engine Inoperative', 'Eng (2) One Engine Inoperative', 'Autorotation')]
        self.assertEqual(combinations, expected)

    def test_derive(self):
        data = [0]*10 + [1]*5 + [0]*25
        eng_1 = M(name='Eng (1) One EngineI noperative', array=np.ma.array(data, dtype=int),
                       values_mapping=Eng1OneEngineInoperative.values_mapping)
        eng_2 = M(name='Eng (2) One Engine Inoperative', array=np.ma.array(np.roll(data, 10), dtype=int),
                       values_mapping=Eng2OneEngineInoperative.values_mapping)

        node = self.node_class()
        node.derive(eng_1, eng_2, S(items=[]))

        expected_data = [0]*10 + [1]*5 + [0]*5 + [1]*5 + [0]*15
        expected = self.node_class(name='One Engine Inoperative', array=np.ma.array(expected_data, dtype=int),
                       values_mapping=self.node_class.values_mapping)
        np.testing.assert_array_equal(node.array, expected.array)


class TestRotorBrakeEngaged(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorBrakeEngaged
        self.values_mapping = {1: 'Engaged', 0: '-'}
        self.brk1 = M('Rotor Brake (1) Engaged',
                      np.ma.array(data=[0,0,1,1,0,0]),
                      values_mapping=self.values_mapping)
        self.brk2 = M('Rotor Brake (2) Engaged',
                      np.ma.array(data=[0,0,0,1,1,0]),
                      values_mapping=self.values_mapping)

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        for opt in opts:
            brk1 = 'Rotor Brake (1) Engaged' in opt
            brk2 = 'Rotor Brake (2) Engaged' in opt
            self.assertTrue(brk1 or brk2)

    def test_brk1(self):
        brk1 = self.brk1
        brk2 = None

        node = self.node_class()
        node.derive(brk1, brk2)

        expected = M('Rotor Brake Engaged',
                     np.ma.array(data=[0,0,1,1,0,0]),
                     values_mapping=self.values_mapping)

        np.testing.assert_array_equal(expected.array, node.array)

    def test_brk2(self):
        brk1 = None
        brk2 = self.brk2

        node = self.node_class()
        node.derive(brk1, brk2)

        expected = M('Rotor Brake Engaged',
                     np.ma.array(data=[0,0,0,1,1,0]),
                     values_mapping=self.values_mapping)

        np.testing.assert_array_equal(expected.array, node.array)

    def test_both_brakes(self):
        brk1 = self.brk1
        brk2 = self.brk2

        node = self.node_class()
        node.derive(brk1, brk2)

        expected = M('Rotor Brake Engaged',
                     np.ma.array(data=[0,0,1,1,1,0]),
                     values_mapping=self.values_mapping)

        np.testing.assert_array_equal(expected.array, node.array)


class TestRotorsRunning(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorsRunning

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=helicopter), [('Nr',)])

    @unittest.SkipTest
    def test_derive(self):
        self.assertTrue(False)
