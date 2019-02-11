import os
import numpy as np
import sys
import unittest

from mock import Mock, call, patch

from analysis_engine.node import (
    P, S, M, A, KTI, KPV, helicopter, aeroplane, App, ApproachItem,
    KeyPointValue, KeyTimeInstance, Section, SectionNode,
)

from flightdatautilities import units as ut

from hdfaccess.parameter import MappedArray

from analysis_engine.library import (max_abs_value, max_value, min_value)

from analysis_engine.key_time_instances import (
    AltitudeWhenDescending,
    AltitudeBeforeLevelFlightWhenClimbing,
    AltitudeBeforeLevelFlightWhenDescending,
    EngStart,
    EngStop,
    DistanceFromThreshold,
    DistanceToTouchdown,
    SecsToTouchdown,
)

from analysis_engine.helicopter.key_point_values import (
    Airspeed500To100FtMax,
    Airspeed500To100FtMin,
    Airspeed100To20FtMax,
    Airspeed100To20FtMin,
    Airspeed20FtToTouchdownMax,
    Airspeed2NMToOffshoreTouchdown,
    AirspeedAbove101PercentRotorSpeed,
    AirspeedAbove500FtMin,
    AirspeedAbove500FtMinOffshoreSpecialProcedure,
    AirspeedAt200FtDuringOnshoreApproach,
    AirspeedAtAPGoAroundEngaged,
    AirspeedWhileAPHeadingEngagedMin,
    AirspeedWhileAPVerticalSpeedEngagedMin,
    AirspeedDuringAutorotationMax,
    AirspeedDuringAutorotationMin,
    AltitudeDensityMax,
    AltitudeRadioDuringAutorotationMin,
    AltitudeDuringCruiseMin,
    AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore,
    AltitudeRadioAtNoseDownAttitudeInitiation,
    CollectiveFrom10To60PercentDuration,
    TailRotorPedalWhileTaxiingABSMax,
    TailRotorPedalWhileTaxiingMax,
    TailRotorPedalWhileTaxiingMin,
    CyclicAftDuringTaxiMax,
    CyclicDuringTaxiMax,
    CyclicForeDuringTaxiMax,
    CyclicLateralDuringTaxiMax,
    EngTorqueExceeding100Percent,
    EngTorqueExceeding110Percent,
    EngN2DuringMaximumContinuousPowerMin,
    EngTorqueWithOneEngineInoperativeMax,
    EngTorqueAbove90KtsMax,
    EngTorqueAbove100KtsMax,
    MGBOilTempMax,
    MGBOilPressMax,
    MGBOilPressMin,
    MGBOilPressLowDuration,
    CGBOilTempMax,
    CGBOilPressMax,
    CGBOilPressMin,
    HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxSpecialProcedure,
    HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxStandardApproach,
    TrackVariation100To50Ft,
    HeadingDuringLanding,
    Groundspeed20FtToTouchdownMax,
    Groundspeed20SecToOffshoreTouchdownMax,
    Groundspeed0_8NMToOffshoreTouchdownSpecialProcedure,
    Groundspeed0_8NMToOffshoreTouchdownStandardApproach,
    GroundspeedBelow15FtFor20SecMax,
    GroundspeedWhileAirborneWithASEOff,
    GroundspeedWhileHoverTaxiingMax,
    GroundspeedWithZeroAirspeedFor5SecMax,
    GroundspeedBelow100FtMax,
    PitchBelow1000FtMax,
    PitchBelow1000FtMin,
    PitchBelow5FtMax,
    Pitch5To10FtMax,
    Pitch10To5FtMax,
    Pitch500To100FtMax,
    Pitch500To100FtMin,
    Pitch100To20FtMax,
    Pitch100To20FtMin,
    Pitch50FtToTouchdownMin,
    PitchOnGroundMax,
    PitchOnDeckMax,
    PitchOnDeckMin,
    PitchOnGroundMin,
    RateOfDescent100To20FtMax,
    RateOfDescent500To100FtMax,
    RateOfDescent20FtToTouchdownMax,
    RateOfDescentBelow500FtMax,
    RateOfDescentBelow30KtsWithPowerOnMax,
    VerticalSpeedAtAltitude,
    Roll100To20FtMax,
    RollAbove300FtMax,
    RollBelow300FtMax,
    RollWithAFCSDisengagedMax,
    RollAbove500FtMax,
    RollBelow500FtMax,
    RollOnGroundMax,
    RollOnDeckMax,
    RollRateMax,
    RotorSpeedDuringAutorotationAbove108KtsMin,
    RotorSpeedDuringAutorotationBelow108KtsMin,
    RotorSpeedDuringAutorotationMax,
    RotorSpeedDuringAutorotationMin,
    RotorSpeedWhileAirborneMax,
    RotorSpeedWhileAirborneMin,
    RotorSpeedWithRotorBrakeAppliedMax,
    RotorsRunningDuration,
    RotorSpeedDuringMaximumContinuousPowerMin,
    RotorSpeed36To49Duration,
    RotorSpeed56To67Duration,
    RotorSpeedAt6PercentCollectiveDuringEngStart,
    WindSpeedInCriticalAzimuth,
    SATMin,
    SATRateOfChangeMax,
    CruiseGuideIndicatorMax,
    TrainingModeDuration,
    HoverHeightDuringOffshoreTakeoffMax,
    HoverHeightDuringOnshoreTakeoffMax,
)

from analysis_engine.test_utils import buildsection, buildsections

debug = sys.gettrace() is not None

##############################################################################
# Superclasses


class NodeTest(object):

    def generate_attributes(self, manufacturer):
        if manufacturer == 'boeing':
            _am = A('Model', 'B737-333')
            _as = A('Series', 'B737-300')
            _af = A('Family', 'B737 Classic')
            _et = A('Engine Type', 'CFM56-3B1')
            _es = A('Engine Series', 'CFM56-3')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'airbus':
            _am = A('Model', 'A330-333')
            _as = A('Series', 'A330-300')
            _af = A('Family', 'A330')
            _et = A('Engine Type', 'Trent 772B-60')
            _es = A('Engine Series', 'Trent 772B')
            return (_am, _as, _af, _et, _es)
        if manufacturer == 'beechcraft':
            _am = A('Model', '1900D')
            _as = A('Series', '1900D')
            _af = A('Family', '1900')
            _et = A('Engine Type', 'PT6A-67D')
            _es = A('Engine Series', 'PT6A')
            return (_am, _as, _af, _et, _es)
        raise ValueError('Unexpected lookup for attributes.')

    def test_can_operate(self):
        if not hasattr(self, 'node_class'):
            return
        kwargs = getattr(self, 'can_operate_kwargs', {})
        if getattr(self, 'check_operational_combination_length_only', False):
            self.assertEqual(
                len(self.node_class.get_operational_combinations(**kwargs)),
                self.operational_combination_length,
            )
        else:
            combinations = list(map(set, self.node_class.get_operational_combinations(**kwargs)))
            for combination in map(set, self.operational_combinations):
                self.assertIn(combination, combinations)

    def get_params_from_hdf(self, hdf_path, param_names, _slice=None,
                            phase_name='Phase'):
        import shutil
        import tempfile
        from hdfaccess.file import hdf_file
        from analysis_engine.node import derived_param_from_hdf

        params = []
        phase = None

        with tempfile.NamedTemporaryFile() as temp_file:
            shutil.copy(hdf_path, temp_file.name)

            with hdf_file(temp_file.name) as hdf:
                for param_name in param_names:
                    p = hdf.get(param_name)
                    if p is not None:
                        p = derived_param_from_hdf(p)
                    params.append(p)

        if _slice:
            phase = S(name=phase_name, frequency=1)
            phase.create_section(_slice)
            phase = phase.get_aligned(params[0])

        return params, phase


class CreateKPVsWhereTest(NodeTest):
    '''
    Basic test for KPVs created with `create_kpvs_where()` method.

    The rationale for this class is to be able to use very simple test case
    boilerplate for the "multi state parameter duration in given flight phase"
    scenario.

    This test checks basic mechanics of specific type of KPV: duration of a
    given state in multistate parameter.

    The test supports multiple parameters and optionally a phase name
    within which the duration is measured.

    What is tested this class:
        * kpv.can_operate() results
        * parameter and KPV names
        * state names
        * basic logic to measure the time of a state duration within a phase
          (slice)

    What isn't tested:
        * potential edge cases of specific parameters
    '''
    def basic_setup(self):
        '''
        Setup for test_derive_basic.

        In the most basic use case the test which derives from this class
        should declare the attributes used to build the test case and then call
        self.basic_setup().

        You need to declare:

        self.node_class::
            class of the KPV node to be used to derive

        self.param_name::
            name of the parameter to be passed to the KPVs `derive()` method

        self.phase_name::
            name of the flight phase to be passed to the `derive()` or None if
            the KPV does not use phases

        self.values_mapping::
            "state to state name" mapping for multistate parameter

        Optionally:

        self.additional_params::
            list of additional parameters to be passed to the `derive()` after
            the main parameter. If unset, only one parameter will be used.


        The method performs the following operations:

            1. Builds the main parameter using self.param_name,
               self.values_array and self.values_mapping

            2. Builds self.params list from the main parameter and
               self.additional_params, if given
            3. Optionally builds self.phases with self.phase_name if given
            4. Builds self.operational_combinations from self.params and
               self.phases
            5. Builds self.expected list of expected values using
               self.node_class and self.phases

        Any of the built attributes can be overridden in the derived class to
        alter the expected test results.
        '''
        if not hasattr(self, 'values_array'):
            self.values_array = np.ma.concatenate((np.zeros(3), np.ones(6), np.zeros(3)))

        if not hasattr(self, 'phase_slice'):
            self.phase_slice = slice(2, 7)

        if not hasattr(self, 'expected_index'):
            self.expected_index = 3

        if not hasattr(self, 'params'):
            self.params = [
                MultistateDerivedParameterNode(
                    self.param_name,
                    array=self.values_array,
                    values_mapping=self.values_mapping
                )
            ]

            if hasattr(self, 'additional_params'):
                self.params += self.additional_params

        if hasattr(self, 'phase_name') and self.phase_name:
            self.phases = buildsection(self.phase_name,
                                       self.phase_slice.start,
                                       self.phase_slice.stop)
        else:
            self.phases = []

        if not hasattr(self, 'operational_combinations'):
            combinations = [p.name for p in self.params]

            self.operational_combinations = [combinations]
            if self.phases:
                combinations.append(self.phases.name)

        if not hasattr(self, 'expected'):
            self.expected = []
            if self.phases:
                # TODO: remove after intervals have been implemented
                if hasattr(self, 'complex_where'):
                    slices = self.phases.get_slices()
                else:
                    slices = [p.slice for p in self.phases]
            else:
                slices = [slice(None)]

            for sl in slices:
                expected_value = np.count_nonzero(
                    self.values_array[sl])
                if expected_value:
                    self.expected.append(
                        KeyPointValue(
                            name=self.node_class().get_name(),
                            index=self.expected_index,
                            value=expected_value
                        )
                    )

    def test_can_operate(self):
        '''
        Test the operational combinations.
        '''
        # sets of sorted tuples of node combinations must match exactly
        kpv_operational_combinations = \
            self.node_class.get_operational_combinations()

        kpv_combinations = set(
            tuple(sorted(c)) for c in kpv_operational_combinations)

        expected_combinations = set(
            tuple(sorted(c)) for c in self.operational_combinations)

        self.assertSetEqual(kpv_combinations, expected_combinations)

    def test_derive_basic(self):
        if hasattr(self, 'node_class'):
            node = self.node_class()
            params = self.params
            if self.phases:
                params.append(self.phases)
            node.derive(*(params))
            self.assertEqual(node, self.expected)


class CreateKPVsAtKPVsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKPVsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_kpvs = Mock()
        node.derive(mock1, mock2)
        node.create_kpvs_at_kpvs.assert_called_once_with(mock1.array, mock2)


class CreateKPVsAtKTIsTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestAltitudeAtLiftoff(unittest.TestCase, CreateKPVsAtKTIsTest):
            def setUp(self):
                self.node_class = AltitudeAtLiftoff
                self.operational_combinations = [('Altitude STD', 'Liftoff')]
    '''
    def test_derive_mocked(self):
        mock1, mock2 = Mock(), Mock()
        mock1.array = Mock()
        node = self.node_class()
        node.create_kpvs_at_ktis = Mock()
        node.derive(mock1, mock2)
        kwargs = {}
        if hasattr(self, 'interpolate'):
            kwargs = {'interpolate': self.interpolate}
        node.create_kpvs_at_ktis.assert_called_once_with(mock1.array, mock2, **kwargs)


class CreateKPVsWithinSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        mock1.frequency = 1.0
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpvs_within_slices = Mock()
        node.derive(mock1, mock2)
        if hasattr(self, 'second_param_method_calls'):
            mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
            node.create_kpvs_within_slices.assert_called_once_with(
                mock1.array, mock3.return_value, self.function)
        else:
            self.assertEqual(mock2.method_calls, [])
            node.create_kpvs_within_slices.assert_called_once_with(
                mock1.array, mock2, self.function)


class CreateKPVsWithinSlicesSecondWindowTest(CreateKPVsWithinSlicesTest):
    '''
    '''
    @patch('analysis_engine.key_point_values.second_window')
    def test_derive_mocked(self, second_window):
        # Not interested in testing functionallity of second window, this is
        # handled in library tests. Here we just want to check it was called
        # with the correct duration.
        second_window.side_effect = lambda *args, **kw: args[0]
        super(CreateKPVsWithinSlicesSecondWindowTest, self).test_derive_mocked()
        self.assertEqual(second_window.call_count, 1)
        # check correct duration used.
        self.assertEqual(second_window.call_args[0][2], self.duration, msg="Incorrect duration used.")


class CreateKPVFromSlicesTest(NodeTest):
    '''
    Example of subclass inheriting tests::

        class TestRollAbove1500FtMax(unittest.TestCase, CreateKPVsWithinSlicesTest):
            def setUp(self):
                self.node_class = RollAbove1500FtMax
                self.operational_combinations = [('Roll', 'Altitude AAL For Flight Phases')]
                # Function passed to create_kpvs_within_slices
                self.function = max_abs_value
                # second_param_method_calls are method calls made on the second
                # parameter argument, for example calling slices_above on a Parameter.
                # It is optional.
                self.second_param_method_calls = [('slices_above', (1500,), {})]

    TODO: Implement in a neater way?
    '''
    def test_derive_mocked(self):
        mock1, mock2, mock3 = Mock(), Mock(), Mock()
        mock1.array = Mock()
        if hasattr(self, 'second_param_method_calls'):
            mock3 = Mock()
            setattr(mock2, self.second_param_method_calls[0][0], mock3)
            mock3.return_value = Mock()
        node = self.node_class()
        node.create_kpv_from_slices = Mock()
        node.derive(mock1, mock2)
        ####if hasattr(self, 'second_param_method_calls'):
        ####    mock3.assert_called_once_with(*self.second_param_method_calls[0][1])
        ####    node.create_kpv_from_slices.assert_called_once_with(\
        ####        mock1.array, mock3.return_value, self.function)
        ####else:
        ####    self.assertEqual(mock2.method_calls, [])
        ####    node.create_kpv_from_slices.assert_called_once_with(\
        ####        mock1.array, mock2, self.function)


class TestAirspeed500To100FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed500To100FtMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertTrue(self.node_class.can_operate, [('Airspeed', 'Altitude AGL', 'Final Approach')])

    def test_derive_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_ph = P('Altitude AAL For Flight Phases', array * 3)
        final_app = buildsections('Final Approach', [31, 58], [95, 120])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 40)
        self.assertAlmostEqual(kpv[0].value, 165, places=0)
        self.assertEqual(kpv[1].index, 103)
        self.assertAlmostEqual(kpv[1].value, 164, places=0)


class TestAirspeed500To100FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed500To100FtMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertTrue(self.node_class.can_operate, [('Airspeed', 'Altitude AGL', 'Final Approach')])

    def test_derive_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_ph = P('Altitude AAL For Flight Phases', array * 3)
        final_app = buildsections('Final Approach', [31, 58], [95, 120])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 54)
        self.assertAlmostEqual(kpv[0].value, 37, places=0)
        self.assertEqual(kpv[1].index, 117)
        self.assertAlmostEqual(kpv[1].value, 35, places=0)


class TestAirspeed100To20FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed100To20FtMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertTrue(self.node_class.can_operate, [('Airspeed', 'Altitude AGL', 'Approach And Landing')])

    def test_derive_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_ph = P('Altitude AGL', array)
        final_app = buildsections('Approach And Landing', [31, 58], [95, 120])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 48)
        self.assertAlmostEqual(kpv[0].value, 91, places=0)
        self.assertEqual(kpv[1].index, 110)
        self.assertAlmostEqual(kpv[1].value, 100, places=0)


class TestAirspeed100To20FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed100To20FtMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertTrue(self.node_class.can_operate, [('Airspeed', 'Altitude AGL', 'Approach And Landing')])

    def test_derive_basic(self):
        array = (np.ma.cos(np.arange(0, 12.6, 0.1)) * -100) + 100
        spd = P('Airspeed', array)
        alt_ph = P('Altitude AGL', array)
        final_app = buildsections('Approach And Landing', [31, 58], [95, 120])
        kpv = self.node_class()
        kpv.derive(spd, alt_ph, final_app)
        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].index, 56)
        self.assertAlmostEqual(kpv[0].value, 22, places=0)
        self.assertEqual(kpv[1].index, 119)
        self.assertAlmostEqual(kpv[1].value, 21, places=0)


class TestAirspeed20FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed20FtToTouchdownMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Altitude AGL For Flight Phases', 'Touchdown')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.concatenate((np.arange(90, 0, -1), np.zeros(10))))
        spd = P('Airspeed', np.ma.arange(100, 0, -1))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown')])

        node = self.node_class()
        node.derive(spd, alt, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 71)
        self.assertEqual(node[0].value, 29)


class TestAirspeed2NMToOffshoreTouchdown(unittest.TestCase):

    def setUp(self):
        self.node_class = Airspeed2NMToOffshoreTouchdown

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Airspeed 2 NM To Touchdown')
        self.assertEqual(node.units, 'kt')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 3)
        self.assertIn('Airspeed', opts[0])
        self.assertIn('Distance To Touchdown', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])

    def test_derive(self):
        air_spd = np.linspace(64, 7, 25).tolist()
        air_spd += np.linspace(84, 28, 11).tolist()
        airspeed = P('Airspeed', np.ma.array(air_spd))
        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(24, 'Offshore Touchdown'),
                                                     KeyTimeInstance(35, 'Offshore Touchdown')])

        dtts = DistanceToTouchdown('Distance To Touchdown',
                   items=[KeyTimeInstance(16, '0.8 NM To Touchdown'),
                          KeyTimeInstance(14, '1.0 NM To Touchdown'),
                          KeyTimeInstance(9, '1.5 NM To Touchdown'),
                          KeyTimeInstance(4, '2.0 NM To Touchdown'),
                          KeyTimeInstance(32, '0.8 NM To Touchdown'),
                          KeyTimeInstance(31, '1.0 NM To Touchdown'),
                          KeyTimeInstance(29, '1.5 NM To Touchdown'),
                          KeyTimeInstance(27, '2.0 NM To Touchdown'),
                          KeyTimeInstance(37, '2.0 NM To Touchdown')])

        node = self.node_class()
        node.derive(airspeed, dtts, touchdown)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 4)
        self.assertAlmostEqual(node[0].value, 54.5, places=1)
        self.assertEqual(node[1].index, 27)
        self.assertAlmostEqual(node[1].value, 72.8, places=1)

class TestAirspeedAbove101PercentRotorSpeed(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedAbove101PercentRotorSpeed

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Airborne', 'Nr')])

    def test_derive(self):
        air_spd_array = np.ma.repeat([0, 15, 35, 80, 90], 6)
        air_spd = P('Airspeed', air_spd_array)
        nr_array = np.ma.concatenate((np.ones(24) * 101.0, np.ones(6) * 100.0))
        nr = P('Nr', nr_array)
        airborne = buildsection('Airborne', 5, 29)
        node = self.node_class()
        node.derive(air_spd, airborne, nr)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 18)
        self.assertEqual(node[0].value, 80)

    def test_derive_masked(self):
        air_spd_array = np.ma.array([90.0]*30)
        air_spd = P('Airspeed', air_spd_array)
        nr_array = np.ma.concatenate((
            np.ones(10) * 100.0,
            [101.0, 102.0, 103.0, 104.0, 105.0],
            np.ones(15) * 100.0))
        nr_array[10:15] = np.ma.masked
        nr = P('Nr', nr_array)
        airborne = buildsection('Airborne', 5, 25)
        node = self.node_class()
        node.derive(air_spd, airborne, nr)
        self.assertEqual(len(node), 0)

    def test_derive_no_exceeding(self):
        air_spd_array = np.ma.array(np.repeat([0, 15, 35, 80, 90],6))
        air_spd = P('Airspeed', air_spd_array)
        nr_array = np.ma.ones(30) * 100.0
        nr = P('Nr', nr_array)
        airborne = buildsection('Airborne', 5, 25)
        node = self.node_class()
        node.derive(air_spd, airborne, nr)
        self.assertEqual(len(node), 0)

    def test_derive_multiple_exceeds(self):
        air_spd_array = np.ma.array([90.0]*30)
        air_spd = P('Airspeed', air_spd_array)
        nr_array = np.ma.concatenate((
            np.ones(7) * 100,
            np.ones(4) * 101,
            [102.0, 103.0, 104.0, 105.0],
            np.ones(5) * 100,
            [101.0, 102.0, 103.0, 104.0, 105.0],
            np.ones(5) * 100,
        ))
        nr = P('Nr', nr_array)
        airborne = buildsection('Airborne', 5, 28)
        node = self.node_class()
        node.derive(air_spd, airborne, nr)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 7)
        self.assertEqual(node[0].value, 90.0)
        self.assertEqual(node[1].index, 20)
        self.assertEqual(node[1].value, 90.0)


class TestAirspeedAbove500FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedAbove500FtMin

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Altitude AGL For Flight Phases', 'Approach Information')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.array(np.tile(np.linspace(200, 1000, 40),2)))
        spd = P('Airspeed', np.ma.array(np.tile(np.linspace(90, 100, 40), 2)))

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(25, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(32, 38, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(spd, alt, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 55)
        self.assertAlmostEqual(node[0].value, 93.84, places=1)


class TestAirspeedAbove500FtMinOffshoreSpecialProcedure(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedAbove500FtMinOffshoreSpecialProcedure

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Altitude AGL For Flight Phases', 'Approach Information')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.array(np.tile(np.linspace(200, 1000, 40),2)))
        spd = P('Airspeed', np.ma.array(np.tile(np.linspace(90, 100, 40),2)))

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(25, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(32, 38, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(spd, alt, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertAlmostEqual(node[0].value, 93.84, places=1)


class TestAirspeedAt200FtDuringOnshoreApproach(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = AirspeedAt200FtDuringOnshoreApproach
        self.can_operate_kwargs = {'ac_type': helicopter}
        self.operational_combinations = [
            ('Airspeed', 'Altitude AGL For Flight Phases', 'Approach Information', 'Offshore'),]

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Airspeed At 200 Ft During Onshore Approach')
        self.assertEqual(node.units, 'kt')

    def test_derive(self):
        x = np.linspace(3, 141, 17).tolist() + [140] + \
            np.linspace(140, 2, 17).tolist() + \
            np.linspace(5, 139, 17).tolist() + [138] + \
            np.linspace(138, 0, 17).tolist()
        air_spd = P('Airspeed', x)
        approaches = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(25, 30)),
                          ApproachItem('LANDING', slice(60, 65))])
        y = np.linspace(190, 403, 17).tolist() + \
            np.linspace(415, 20, 18).tolist() + \
            np.linspace(230, 534, 17).tolist() + \
            np.linspace(503, 50, 18).tolist()
        alt_agl = P('Altitude AGL For Flight Phases', y)

        offshore = M(name='Offshore', array=np.ma.array([0]*70, dtype=int),
                 values_mapping={0: 'Onshore', 1: 'Offshore'})

        node = self.node_class()
        node.derive(air_spd, alt_agl, approaches, offshore)

        self.assertEqual(len(node), 2)
        self.assertAlmostEqual(node[0].index, 26, places=0)
        self.assertAlmostEqual(node[0].value, 68.8, places=1)

        self.assertAlmostEqual(node[1].index, 63, places=0)
        self.assertAlmostEqual(node[1].value, 48.6, places=1)

    def test_not_offshore(self):
        x = np.linspace(3, 141, 17).tolist() + [140] + \
            np.linspace(140, 2, 17).tolist() + \
            np.linspace(5, 139, 17).tolist() + [138] + \
            np.linspace(138, 0, 17).tolist()
        air_spd = P('Airspeed', x)
        approaches = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(25, 30)),
                          ApproachItem('LANDING', slice(60, 65))])
        y = np.linspace(190, 403, 17).tolist() + \
            np.linspace(415, 20, 18).tolist() + \
            np.linspace(230, 534, 17).tolist() + \
            np.linspace(503, 50, 18).tolist()
        alt_agl = P('Altitude AGL For Flight Phases', y)

        offshore = M(name='Offshore', array=np.ma.ones(70, dtype=int),
                 values_mapping={0: 'Onshore', 1: 'Offshore'})

        node = self.node_class()
        node.derive(air_spd, alt_agl, approaches, offshore)

        self.assertEqual(len(node), 0)

    def test_short_hop(self):
        x = np.linspace(3, 141, 17).tolist() + [140] + \
            np.linspace(140, 2, 17).tolist() + \
            np.linspace(5, 139, 17).tolist() + [138] + \
            np.linspace(138, 0, 17).tolist()
        air_spd = P('Airspeed', x)
        approaches = App('Approach Information',
                   items=[ApproachItem('LANDING', slice(25, 30)),
                          ApproachItem('LANDING', slice(34, 65))])
        y = np.linspace(190, 403, 17).tolist() + \
            np.linspace(415, 20, 18).tolist() + \
            np.linspace(230, 534, 17).tolist() + \
            np.linspace(503, 50, 18).tolist()
        alt_agl = P('Altitude AGL For Flight Phases', y)

        offshore = M(name='Offshore', array=np.ma.zeros(70, dtype=int),
                 values_mapping={0: 'Onshore', 1: 'Offshore'})

        node = self.node_class()
        node.derive(air_spd, alt_agl, approaches, offshore)

        self.assertEqual(len(node), 2)
        self.assertAlmostEqual(node[0].index, 26, places=0)
        self.assertAlmostEqual(node[0].value, 68.8, places=1)

        self.assertAlmostEqual(node[1].index, 63, places=0)
        self.assertAlmostEqual(node[1].value, 48.6, places=1)



class TestAirspeedAtAPGoAroundEngaged(unittest.TestCase):
    '''
    '''

    # Set up the autopilot values mapping.
    vm = {0:'Off', 1:'IAS', 2:'Alt', 3:'Alt.A',
          4:'V/S', 5:'Glideslope', 6:'Go Around',
          8:'Hover', 9:'Overfly', 10:'Trans-Up',}

    def setUp(self):
        self.node_class = AirspeedAtAPGoAroundEngaged


    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Airborne', 'AP Pitch Mode (1)')])

    def test_derive(self):
        aspd = P('Airspeed', np.ma.ones(10) * 34)
        airs = buildsection('Airborne', 3, 9)
        mode = M(name='AP Pitch Mode (1)', array=np.ma.arange(10, dtype=int),
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 34)

    def test__airborne_phase_and_first_sample(self):
        aspd = P('Airspeed', np.ma.arange(10))
        airs = buildsection('Airborne', 5, 9)
        mode = M(name='AP Pitch Mode (1)', array=np.ma.array([5,6,6,5,5,5,6,6,6,6,],dtype=int),
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 6)


class TestAirspeedWhileAPHeadingEngagedMin(unittest.TestCase):
    '''
    '''

    # Set up the autopilot values mapping.
    vm = {0:'Off', 1:'Heading', 2:'Nav', 3:'VOR',
          4:'Loc', 5:'VOR Approach', 6:'Go Around',
          8:'Hover', 9:'Overfly', 10:'Trans-Up',}

    def setUp(self):
        self.node_class = AirspeedWhileAPHeadingEngagedMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Airborne', 'AP Roll-Yaw Mode (1)')])

    def test_derive(self):
        aspd = P('Airspeed', np.ma.ones(10) * 34)
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Roll-Yaw Mode (1)', array=np.ma.arange(10, dtype=int),
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 1)
        self.assertEqual(node[0].value, 34)

    def test_check_min(self):
        aspd = P('Airspeed', np.ma.array([34.0]*5+[33]+[34]*4))
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Roll-Yaw Mode (1)', array=np.ma.ones(10, dtype=int),
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 33)

    def test_no_mode(self):
        aspd = P('Airspeed', np.ma.ones(10) * 34)
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Roll-Yaw Mode (1)', array=np.ma.ones(10, dtype=int) * 7,
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 0)


class TestAirspeedWhileAPVerticalSpeedEngagedMin(unittest.TestCase):
    '''
    '''

    # Set up the autopilot values mapping.
    vm = {0:'Off', 1:'CR.HT', 2:'Alt', 3:'Alt.A',
          4:'V/S', 5:'Glideslope', 6:'Go Around',
          8:'HHT', 9:'Overfly', 10:'Trans-Up',}

    def setUp(self):
        self.node_class = AirspeedWhileAPVerticalSpeedEngagedMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Airborne', 'AP Collective Mode (1)')])

    def test_derive(self):
        aspd = P('Airspeed', np.ma.ones(10) * 34)
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Collective Mode (1)', array=np.ma.arange(10, dtype=int),
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 4)
        self.assertEqual(node[0].value, 34)

    def test_check_min(self):
        aspd = P('Airspeed', np.ma.array([34.0] * 5 + [33] + [34] * 4))
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Collective Mode (1)', array=np.ma.ones(10, dtype=int) * 4,
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 33)

    def test_no_mode(self):
        aspd = P('Airspeed', np.ma.ones(10) * 34.0)
        airs = buildsection('Airborne', 1, 9)
        mode = M(name='AP Collective Mode (1)', array=np.ma.ones(10, dtype=int) * 7,
                 values_mapping=self.vm)
        node = self.node_class()
        node.derive(aspd, airs, mode)
        self.assertEqual(len(node), 0)


class TestAirspeedDuringAutorotationMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedDuringAutorotationMax

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Autorotation')])

    def test_derive(self):
        name = 'Autorotation'
        section = Section(name, slice(70, 100), 70, 100)
        autorotation = SectionNode(name, items=[section])

        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = P('Airspeed', np.ma.array(testwave))

        node = self.node_class()
        node.derive(spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 94)
        self.assertAlmostEqual(node[0].value, 200, places=0)


class TestAirspeedDuringAutorotationMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AirspeedDuringAutorotationMin

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Airspeed', 'Autorotation')])

    def test_derive(self):
        name = 'Autorotation'
        section = Section(name, slice(70, 100), 70, 100)
        autorotation = SectionNode(name, items=[section])

        testline = np.arange(0, 12.6, 0.1)
        testwave = (np.cos(testline) * -100) + 100
        spd = P('Airspeed', np.ma.array(testwave))

        node = self.node_class()
        node.derive(spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 70)
        self.assertAlmostEqual(node[0].value, 25, places=0)


class TestAltitudeDensityMax(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeDensityMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Altitude Density', 'Airborne')])

    def test_derive(self):
        alt_std = P('Altitude Density', np.ma.arange(0, 11))
        name = 'Airborne'
        section = Section(name, slice(0, 9), 0, 9)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(alt_std, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8)
        self.assertEqual(node[0].value, 8)


class TestAltitudeRadioDuringAutorotationMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeRadioDuringAutorotationMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Altitude Radio', 'Autorotation')])

    def test_derive(self):
        alt_rad = P(
            name='Altitude Radio',
            array=np.arange(4000, 0, -16),
        )
        name = 'Autorotation'
        section = Section(name, slice(10, 240), 10.2, 240.5)
        autorotation = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(alt_rad, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 240.5)
        self.assertEqual(node[0].value, 152)


class TestAltitudeDuringCruiseMin(unittest.TestCase):

    def setUp(self):
        self.node_class = AltitudeDuringCruiseMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Altitude During Cruise Min')
        self.assertEqual(node.units, 'ft')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Altitude AGL', opts[0])
        self.assertIn('Cruise', opts[0])

    def test_derive(self):
        alt_agl = P('Altitude AGL',
                    np.ma.array([0, 100, 400, 1000, 1003,
                                 1010, 999, 1000, 500, 100,
                                 0, 0, 100, 500, 1100,
                                 1080, 1090, 1070, 500, 100]))
        cruise = buildsections('Cruise', [3, 7], [14, 17])

        node = self.node_class()
        node.derive(alt_agl, cruise)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 6)
        self.assertEqual(node[0].value, 999)
        self.assertEqual(node[1].index, 17)
        self.assertEqual(node[1].value, 1070)


class TestAltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore(unittest.TestCase):

    def setUp(self):
        self.offshore_mapping = {0: 'Onshore', 1: 'Offshore'}
        self.node_class = AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore

    def test_can_operate(self):
        expected = [('Offshore', 'Liftoff', 'Hover',
                     'Nose Down Attitude Adoption', 'Altitude Radio',
                     'Altitude AAL For Flight Phases')]

        opts_h175 = self.node_class.get_operational_combinations(
                    ac_type=helicopter, family=A('Family', 'H175'))

        opts_aeroplane = self.node_class.get_operational_combinations(ac_type=aeroplane)

        self.assertEqual(opts_h175, expected)
        self.assertNotEqual(opts_aeroplane, expected)

    def test_derive(self):
        node = AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore()
        offshore_data = np.concatenate([np.zeros(10), np.ones(80),
                                        np.zeros(10)])

        offshore_array = MappedArray(offshore_data,
                                     values_mapping=self.offshore_mapping)
        offshore_multistate = M(name='Offshore', array=offshore_array)

        liftoff = KTI('Liftoff', items=[
            KeyTimeInstance(15, 'Liftoff'),
        ])

        hover = buildsection('Hover', 20, 30)
        nose_down = buildsection('Nose Down Attitude Adoption', 29, 32)

        rad_alt = np.concatenate([np.zeros(13), np.linspace(0, 30, num=7),
                                  np.linspace(30, 5, num=10),
                                  np.linspace(5, 1000, num=70)])

        alt_aal = np.concatenate([np.zeros(13), np.linspace(0, 25, num=7),
                                  np.linspace(25, 3, num=10),
                                  np.linspace(3, 1000, num=70)])

        node.derive(offshore_multistate, liftoff, hover, nose_down,
                    P('Altitude Radio', rad_alt),
                    P('Altitude AAL For Flight Phases', alt_aal))

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 28)
        self.assertEqual(round(node[0].value, 2), 7.78)

    def test_derive_multiple_liftoffs_and_offshore_clumps(self):
        node = AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore()
        offshore_data = np.concatenate([np.zeros(10), np.ones(35),
                                         np.zeros(3), np.ones(35),
                                         np.zeros(17)])

        offshore_array = MappedArray(offshore_data,
                                     values_mapping=self.offshore_mapping)
        offshore_multistate = M(name='Offshore', array=offshore_array)

        liftoffs = KTI('Liftoff', items=[
            KeyTimeInstance(15, 'Liftoff'),
            KeyTimeInstance(50, 'Liftoff'),
        ])

        hovers = buildsections('Hover', [20, 30], [52, 65])
        nose_downs = buildsections('Nose Down Attitude Adoption', [29, 32],
                                                                  [59, 67])

        rad_alt = np.concatenate([np.zeros(13), np.linspace(0, 30, num=7),
                                  np.linspace(30, 5, num=10),
                                  np.zeros(18), np.linspace(5, 30, num=8),
                                  np.linspace(5, 1000, num=44)])

        alt_aal = np.concatenate([np.zeros(13), np.linspace(0, 25, num=7),
                                  np.linspace(25, 3, num=10),
                                  np.zeros(18), np.linspace(3, 24, num=8),
                                  np.linspace(5, 1000, num=44)])

        node.derive(offshore_multistate, liftoffs, hovers, nose_downs,
                    P('Altitude Radio', rad_alt),
                    P('Altitude AAL For Flight Phases', alt_aal))

        self.assertEqual(len(node), 2)

        self.assertEqual(node[0].index, 28)
        self.assertEqual(round(node[0].value, 2), 7.78)
        self.assertEqual(node[1].index, 56)
        self.assertEqual(round(node[1].value, 2), 5)

    def test_derive_fallback(self):
        node = AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore()
        offshore_data = np.concatenate([np.zeros(10), np.ones(80),
                                        np.zeros(10)])

        offshore_array = MappedArray(offshore_data,
                                     values_mapping=self.offshore_mapping)
        offshore_multistate = M(name='Offshore', array=offshore_array)

        liftoff = KTI('Liftoff', items=[
            KeyTimeInstance(15, 'Liftoff'),
        ])

        hover = buildsection('Hover', 20, 30)
        nose_down = buildsection('Nose Down Attitude Adoption', 29, 32)

        rad_alt = np.concatenate([np.zeros(13), np.linspace(30, 5, num=17),
                                  np.linspace(5, 1000, num=70)])

        alt_aal = np.concatenate([np.zeros(13), np.linspace(0, 25, num=17),
                                  np.zeros(70)])

        node.derive(offshore_multistate, liftoff, hover, nose_down,
                    P('Altitude Radio', rad_alt),
                    P('Altitude AAL For Flight Phases', alt_aal))

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 15)
        self.assertEqual(round(node[0].value, 2), 26.88)


class TestAltitudeRadioAtNoseDownAttitudeInitation(unittest.TestCase,
                                                   NodeTest):

    def setUp(self):
        self.node_class = AltitudeRadioAtNoseDownAttitudeInitiation

    def test_can_operate(self):
        expected = [('Altitude Radio', 'Nose Down Attitude Adoption')]
        opts_h175 = self.node_class.get_operational_combinations(
                    ac_type=helicopter, family=A('Family', 'H175'))
        opts_aeroplane = self.node_class.get_operational_combinations(
                         ac_type=aeroplane)

        self.assertEqual(opts_h175, expected)
        self.assertNotEqual(opts_aeroplane, expected)

    def test_derive(self):
        node = AltitudeRadioAtNoseDownAttitudeInitiation()
        rad_alt = np.linspace(0, 50, num=25)
        nose_downs = buildsection('Nose Down Attitude Adoption', 10, 15)
        node.derive(P('Altitude Radio', rad_alt), nose_downs)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(round(node[0].value, 2), 20.83)

    def test_derive_multiple(self):
        node = AltitudeRadioAtNoseDownAttitudeInitiation()
        descent_alt = np.linspace(50, 0, num=10)
        ascent_alt = np.linspace(0, 50, num=15)
        rad_alt = np.concatenate([ascent_alt, descent_alt, ascent_alt])
        nose_downs = buildsections('Nose Down Attitude Adoption',
                                   [5, 20], [23, 39])
        node.derive(P('Altitude Radio', rad_alt), nose_downs)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(round(node[0].value, 2), 17.86)
        self.assertEqual(node[1].index, 23)
        self.assertEqual(round(node[1].value, 2), 5.56)


##############################################################################
# Collective


class TestCollectiveFrom10To60PercentDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = CollectiveFrom10To60PercentDuration

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Collective', 'Rotors Turning')])

    def test_derive(self):
        collective = P(
            name='Collective',
            array=np.ma.array([5]*5 + [15, 30, 60, 90] + [100]*5),
        )
        rtr = buildsection('Rotors Turning', 0, 15)
        node = self.node_class()
        node.derive(collective, rtr)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 3)

    def test_derive__no_occurence(self):
        collective = P(
            name='Collective',
            array=np.ma.array([5]*5 + [15, 30, 60, 30] + [5]*5),
        )
        rtr = buildsection('Rotors Turning', 0, 15)
        node = self.node_class()
        node.derive(collective, rtr)
        self.assertEqual(len(node), 0)

    def test_derive__not_turning(self):
        collective = P(
            name='Collective',
            array=np.ma.array([5]*5 + [15, 30, 60, 90] + [5]*5),
        )
        rtr = buildsection('Rotors Turning', 0, 5)
        node = self.node_class()
        node.derive(collective, rtr)
        self.assertEqual(len(node), 0)

    def test_derive__reject_very_slow(self):
        collective = P(
            name='Collective',
            array=np.ma.array([5]*5 + [15, 30, 60, 90] + [5]*5),
            frequency=0.1
        )
        rtr = buildsection('Rotors Turning', 0, 15)
        node = self.node_class()
        node.derive(collective, rtr)
        self.assertEqual(len(node), 0)


class TestTailRotorPedalWhileTaxiingABSMax(unittest.TestCase):

    def setUp(self):
        self.node_class = TailRotorPedalWhileTaxiingABSMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Tail Rotor Pedal', 'Taxiing')])

    def test_derive(self):
        name = 'Taxiing'
        section = Section(name, slice(0, 150), 0, 150)
        taxiing = SectionNode(name, items=[section])

        x = np.linspace(0, 10, 200)
        pedal = P('Tail Rotor Pedal',(-x*np.sin(x)))

        node = self.node_class()
        node.derive(pedal, taxiing)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, -6.990, places=3)


class TestTailRotorPedalWhileTaxiingMax(unittest.TestCase):

    def setUp(self):
        self.node_class = TailRotorPedalWhileTaxiingMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Tail Rotor Pedal', 'Taxiing')])

    def test_derive(self):
        name = 'Taxiing'
        section = Section(name, slice(0, 150), 0, 150)
        taxiing = SectionNode(name, items=[section])

        x = np.linspace(0, 10, 200)
        pedal = P('Tail Rotor Pedal',(-x*np.sin(x)))

        node = self.node_class()
        node.derive(pedal, taxiing)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 98)
        self.assertAlmostEqual(node[0].value, 4.814, places=3)


class TestTailRotorPedalWhileTaxiingMin(unittest.TestCase):

    def setUp(self):
        self.node_class = TailRotorPedalWhileTaxiingMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Tail Rotor Pedal', 'Taxiing')])

    def test_derive(self):
        name = 'Taxiing'
        section = Section(name, slice(0, 150), 0, 150)
        taxiing = SectionNode(name, items=[section])

        x = np.linspace(0, 10, 200)
        pedal = P('Tail Rotor Pedal',(-x*np.sin(x)))

        node = self.node_class()
        node.derive(pedal, taxiing)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, -6.990, places=3)


##############################################################################
# Cyclic


class TestCyclicDuringTaxiMax(unittest.TestCase):

    x = np.linspace(0, 10, 200)
    cyclic = P(
        name='Cyclic Angle',
        array=np.ma.abs(x*np.sin(x)),
    )
    name = 'Taxiing'
    section = Section(name, slice(0, 150), 0, 150)
    taxi = SectionNode(name, items=[section])

    def setUp(self):
        self.node_class = CyclicDuringTaxiMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Angle', 'Taxiing', 'Rotors Turning')])

    def test_derive(self):

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, self.taxi)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, 6.990, places=3)

    def test_not_stationary(self):

        name = 'Rotors Turning'
        section = Section(name, slice(0, 135), 0, 135)
        rtr = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, rtr)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 98)
        self.assertAlmostEqual(node[0].value, 4.814, places=3)


class TestCyclicLateralDuringTaxiMax(unittest.TestCase):

    x = np.linspace(0, 10, 200)
    cyclic = P(
        name='Cyclic Lateral',
        array=np.ma.abs(x*np.sin(x)),
    )
    name = 'Taxiing'
    section = Section(name, slice(0, 150), 0, 150)
    taxi = SectionNode(name, items=[section])

    def setUp(self):
        self.node_class = CyclicLateralDuringTaxiMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Lateral', 'Taxiing', 'Rotors Turning')])

    def test_derive(self):

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, self.taxi)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, 6.990, places=3)

    def test_not_stationary(self):

        name = 'Rotors Turning'
        section = Section(name, slice(0, 135), 0, 135)
        rtr = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, rtr)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 98)
        self.assertAlmostEqual(node[0].value, 4.814, places=3)


class TestCyclicAftDuringTaxiMax(unittest.TestCase):

    x = np.linspace(0, 10, 200)
    cyclic = P(
        name='Cyclic Fore-Aft',
        array=np.ma.abs(x*np.sin(x)),
    )
    name = 'Taxiing'
    section = Section(name, slice(0, 150), 0, 150)
    taxi = SectionNode(name, items=[section])

    def setUp(self):
        self.node_class = CyclicAftDuringTaxiMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Fore-Aft', 'Taxiing', 'Rotors Turning')])

    def test_derive(self):

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, self.taxi)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, 6.990, places=3)

    def test_not_stationary(self):

        name = 'Rotors Turning'
        section = Section(name, slice(0, 135), 0, 135)
        rtr = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, rtr)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 98)
        self.assertAlmostEqual(node[0].value, 4.814, places=3)


class TestCyclicForeDuringTaxiMax(unittest.TestCase):

    x = np.linspace(0, 10, 200)
    cyclic = P(
        name='Cyclic Fore-Aft',
        array=-np.ma.abs(x*np.sin(x)),
    )
    name = 'Taxiing'
    section = Section(name, slice(0, 150), 0, 150)
    taxi = SectionNode(name, items=[section])

    def setUp(self):
        self.node_class = CyclicForeDuringTaxiMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Cyclic Fore-Aft', 'Taxiing', 'Rotors Turning')])

    def test_derive(self):

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, self.taxi)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 149)
        self.assertAlmostEqual(node[0].value, -6.990, places=3)

    def test_not_stationary(self):

        name = 'Rotors Turning'
        section = Section(name, slice(0, 135), 0, 135)
        rtr = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(self.cyclic, self.taxi, rtr)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 98)
        self.assertAlmostEqual(node[0].value, -4.814, places=3)


class TestEngTorqueExceeding100Percent(unittest.TestCase):


    def setUp(self):
        self.node_class = EngTorqueExceeding100Percent


    def test_derive_exceeding_one_second_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([95, 96, 97, 98, 99, 100, 99, 98]))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)

    def test_derive_exceeding_two_seconds_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([95, 96, 97, 98, 99, 100, 101, 99, 98]))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)

    def test_derive_exceeding_one_longer_than_two_seconds_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 104,
                                              103, 104, 105, 104, 103, 102, 101, 100, 99, 98]))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].value, 14)
        self.assertEqual(kpv[0].index, 5)


    def test_derive_exceeding_multiple_times(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                                              103, 102, 101, 100, 99, 98, 99, 100, 101, 102,
                                              103, 104, 105, 104, 103, 102, 101, 100, 99, 98]))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].value, 9)
        self.assertEqual(kpv[0].index, 5)
        self.assertEqual(kpv[1].value, 11)
        self.assertEqual(kpv[1].index, 17)


    def test_derive_not_exceeding(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([95, 96, 97, 98, 99, 98, 97, 96]))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


    def test_derive_exceeding_masked_data(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                               array=np.ma.masked_greater_equal([95, 96, 97, 98, 99, 100, 101, 102, 103, 104,
                                                  103, 104, 105, 104, 103, 102, 101, 100, 99, 98], 100))

        kpv = EngTorqueExceeding100Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


class TestEngTorqueExceeding110Percent(unittest.TestCase):


    def setUp(self):
        self.node_class = EngTorqueExceeding110Percent


    def test_derive_exceeding_one_second_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([105, 106, 107, 108, 109, 110, 109, 108]))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


    def test_derive_exceeding_two_seconds_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                            array=np.ma.array([105, 106, 107, 108, 109, 110, 111, 109, 108]))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


    def test_derive_exceeding_one_longer_than_two_seconds_period(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 114,
                                              113, 114, 115, 114, 113, 112, 111, 110, 109, 108]))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 1)
        self.assertEqual(kpv[0].value, 14)
        self.assertEqual(kpv[0].index, 5)


    def test_derive_exceeding_multiple_times(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                                              113, 112, 111, 110, 109, 108, 109, 110, 111, 112,
                                              113, 114, 115, 114, 113, 112, 111, 110, 109, 108]))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 2)
        self.assertEqual(kpv[0].value, 9)
        self.assertEqual(kpv[0].index, 5)
        self.assertEqual(kpv[1].value, 11)
        self.assertEqual(kpv[1].index, 17)


    def test_derive_not_exceeding(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                           array=np.ma.array([105, 106, 107, 108, 109, 108, 107, 106]))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


    def test_derive_exceeding_masked_data(self):

        eng_avg_torque = P(name='Eng (*) Avg Torque',
                               array=np.ma.masked_greater_equal([105, 106, 107, 108, 109, 110, 111, 112, 113, 114,
                                                  113, 114, 115, 114, 113, 112, 111, 110, 109, 108], 110))

        kpv = EngTorqueExceeding110Percent()
        kpv.derive(eng_avg_torque)

        self.assertEqual(len(kpv), 0)


class TestEngN2MaximumContinuousPowerMin(unittest.TestCase, CreateKPVsWithinSlicesTest):

    def setUp(self):
        self.node_class = EngN2DuringMaximumContinuousPowerMin
        self.operational_combinations = [('Eng (*) N2 Min', 'Maximum Continuous Power')]
        self.can_operate_kwargs = {'ac_type': helicopter}
        self.function = min_value


class TestEngTorqueWithOneEngineInoperativeMax(unittest.TestCase, NodeTest):

    def setUp(self):
        self.node_class = EngTorqueWithOneEngineInoperativeMax
        self.operational_combinations = [('Eng (*) Torque Max', 'Airborne', 'One Engine Inoperative')]
        self.can_operate_kwargs = {'ac_type': helicopter}
        self.function = max_value

    def test_derive_heli(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))

        airs = buildsection('Airborne', 1, 28)
        one_eng = M('One Engine Inoperative', np.ma.array([1]*14 + [0]*16), values_mapping={0:'-', 1:'OEI'})

        node = self.node_class()
        node.derive(eng, airs, one_eng)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 73)


class TestEngTorqueAbove90KtsMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTorqueAbove90KtsMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(
            node.name,
            'Eng Torque Above 90 Kts Max'
        )
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Eng (*) Torque Max', 'Airspeed')])

    def test_derive_exceeding_once(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 79,
            65, 62, 63, 59, 58, 57, 43, 46, 52, 58,
        ]))
        air_spd = P('Airspeed', np.ma.array([
            136, 132, 131, 131, 132, 135, 131, 132, 131, 131,
            120, 120, 110, 110, 110, 110, 100, 100, 100, 100,
            90,  89,  89,  89,  89,  88,  88,  88,  87,  87,
        ]))

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 19)
        self.assertEqual(node[0].value, 79)


    def test_derive_exeeding_twice_seperated_by_less_than_ten_seconds(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 79,
            65, 62, 63, 59, 58, 57, 43, 46, 52, 58,
            68, 62, 64, 65, 67, 70, 72, 73, 76, 81,
            36, 44, 23, 40, 50, 37, 70, 75, 89, 17,
        ]))
        air_spd = P('Airspeed', np.ma.array([
            136, 132, 131, 131, 132, 135, 131, 132, 131, 131,
            120, 120, 110, 110, 110, 110, 100, 100, 100, 100,
            90,  89,  89,  89,  89,  89,  89,  89,  89,  89,
            132, 131, 132, 131, 131, 132, 130, 121, 113, 95,
            97,  89,  81,  73,  65,  57,  49,  41,  33,  25
        ]))

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 39)
        self.assertEqual(node[0].value, 81)


    def test_derive_exeeding_twice_seperated_by_more_than_ten_seconds(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 79,
            65, 62, 63, 59, 58, 57, 43, 46, 52, 58,
            68, 62, 64, 65, 67, 70, 72, 73, 76, 81,
            36, 44, 23, 40, 50, 37, 70, 75, 89, 17,
        ]))
        air_spd = P('Airspeed', np.ma.array([
            136, 132, 131, 131, 132, 135, 131, 132, 131, 131,
            120, 120, 110, 110, 110, 110, 100, 100, 90, 89,
            89,  89,  89,  88,  88,  88,  89,  89,  89,  89,
            89,  90,  132, 131, 131, 132, 130, 121, 113, 95,
            97,  89,  81,  73,  65,  57,  49,  41,  33,  25
        ]))

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 73)
        self.assertEqual(node[1].index, 39)
        self.assertEqual(node[1].value, 81)


    def test_derive_not_exceeding(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 79,
            65, 62, 63, 59, 58, 57, 43, 46, 52, 58,
        ]))
        air_spd = P('Airspeed', np.ma.array([
            89,  89,  89,  88,  87,  88,  89,  89,  89,  89,
            89,  89,  89,  88,  88,  88,  89,  89,  89,  89,
            89,  89,  89,  88,  88,  88,  77,  76,  77,  79,
        ]))

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 0)


    def test_derive_exceeding_masked_data(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 74, 73, 72, 71, 70, 70, 70, 68, 72,
            73, 74, 75, 76, 70, 78, 70, 70, 68, 72,
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
        ]))
        air_spd = P('Airspeed', np.ma.masked_greater_equal([
            89,  89,  89,  88,  87,  88,  89,  89,  90,  91,
            91,  90,  89,  88,  88,  88,  89,  89,  89,  89,
            89,  89,  89,  88,  88,  88,  77,  76,  77,  79,
        ], 90) )

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 0)



class TestEngTorqueAbove100KtsMax(unittest.TestCase):

    def setUp(self):
        self.node_class = EngTorqueAbove100KtsMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(
            node.name,
            'Eng Torque Above 100 Kts Max'
        )
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Eng (*) Torque Max', 'Airspeed')])

    def test_derive(self):
        eng = P('Eng (*) Torque Max', np.ma.array([
            70, 70, 70, 70, 70, 70, 70, 70, 68, 72,
            67, 73, 66, 59, 60, 58, 45, 60, 40, 49,
            36, 44, 23, 40, 50, 37, 70, 75, 17, 17,
        ]))
        air_spd = P('Airspeed', np.ma.array([
            136, 132, 131, 131, 132, 135, 131, 132, 131, 131,
            132, 131, 132, 131, 131, 132, 130, 121, 113, 99,
            97,  89,  81,  73,  65,  57,  49,  41,  33,  25
        ]))

        node = self.node_class()
        node.derive(eng, air_spd)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 73)


class TestMGBOilTempMax(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilTempMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.CELSIUS)
        self.assertEqual(node.name, 'MGB Oil Temp Max')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),7)
        for opt in opts:
            self.assertIn('Airborne', opt)
            mgb = 'MGB Oil Temp' in opt
            mgb_fwd = 'MGB (Fwd) Oil Temp' in opt
            mgb_aft = 'MGB (Aft) Oil Temp' in opt
            self.assertTrue(mgb or mgb_fwd or mgb_aft)

    def test_derive(self):
        temp = [78.0]*14 + [78.5,78] + [78.5]*23 + [79.0]*5 + [79.5] + [79.0]*5
        mgb_oil_temp = P('MGB Oil Temp', temp)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(mgb_oil_temp, None, None, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 79)
        self.assertEqual(node[0].index, 39)

    def test_derive_fwd_aft(self):
        t1 = [78.0]*14 + [78.5,78] + [78.5]*23 + [79.0]*5 + [79.5] + [79.0]*5
        t2 = [78.0]*14 + [78.5,78] + [78.0]*23 + [78.5]*5 + [79.5] + [79.0]*5

        mgb_fwd_oil_temp = P('MGB (Fwd) Oil Temp', t1)
        mgb_aft_oil_temp = P('MGB (Aft) Oil Temp', t2)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(None, mgb_fwd_oil_temp, mgb_aft_oil_temp, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 79)
        self.assertEqual(node[0].index, 39)


class TestMGBOilPressMax(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilPressMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.PSI)
        self.assertEqual(node.name, 'MGB Oil Press Max')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),7)
        for opt in opts:
            self.assertIn('Airborne', opt)
            mgb = 'MGB Oil Press' in opt
            mgb_fwd = 'MGB (Fwd) Oil Press' in opt
            mgb_aft = 'MGB (Aft) Oil Press' in opt
            self.assertTrue(mgb or mgb_fwd or mgb_aft)

    def test_derive(self):
        press = [26.7] + [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*13
        mgb_oil_press = P('MGB Oil press', press)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(mgb_oil_press, None, None, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.63)
        self.assertEqual(node[0].index, 15)

    def test_derive_fwd_aft(self):
        p1 = [26.7] + [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*13
        p2 = [26.7] + [26.51]*15 + [26.4]*20 + [26.11] + [26.4]*13
        mgb_fwd_oil_press = P('MGB Oil press', p1)
        mgb_aft_oil_press = P('MGB Oil press', p2)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(None, mgb_fwd_oil_press, mgb_aft_oil_press, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.63)
        self.assertEqual(node[0].index, 15)

class TestMGBOilPressMin(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilPressMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.PSI)
        self.assertEqual(node.name, 'MGB Oil Press Min')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),7)
        for opt in opts:
            self.assertIn('Airborne', opt)
            mgb = 'MGB Oil Press' in opt
            mgb_fwd = 'MGB (Fwd) Oil Press' in opt
            mgb_aft = 'MGB (Aft) Oil Press' in opt
            self.assertTrue(mgb or mgb_fwd or mgb_aft)

    def test_derive(self):
        press = [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*13 + [26.1]
        mgb_oil_press = P('MGB Oil press', press)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(mgb_oil_press, None, None, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.29)
        self.assertEqual(node[0].index, 35)

    def test_derive_fwd_aft(self):
        p1 = [26.51]*14 + [26.63] + [26.4]*20 + [26.51] + [26.4]*13 + [26.1]
        p2 = [26.51]*15 + [26.4]*20 + [26.29] + [26.4]*13 + [26.1]
        mgb_fwd_oil_press = P('MGB Oil press', p1)
        mgb_aft_oil_press = P('MGB Oil press', p2)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(None, mgb_fwd_oil_press, mgb_aft_oil_press, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.29)
        self.assertEqual(node[0].index, 35)


class TestMGBOilPressLowDuration(unittest.TestCase):
    def setUp(self):
        self.node_class = MGBOilPressLowDuration

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, 's')
        self.assertEqual(node.name, 'MGB Oil Press Low Duration')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),7)
        for opt in opts:
            self.assertIn('Airborne', opt)
            mgb = 'MGB Oil Press Low' in opt
            mgb1 = 'MGB Oil Press Low (1)' in opt
            mgb2 = 'MGB Oil Press Low (2)' in opt
            self.assertTrue(mgb or mgb1 or mgb2)


    def test_derive(self):
        warn = np.ma.concatenate((np.zeros(5), np.ones(20), np.zeros(5)))
        warn_param = M('MGB Oil Press Low',
                       array=warn,
                       values_mapping={0: '-', 1: 'Low Press'})
        airs = buildsection('Airborne', 1, 38)
        node = self.node_class()
        node.derive(mgb=warn_param, mgb1=None, mgb2=None, airborne=airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 20)

    def test_derive_2(self):
        warn_param_1 = M('MGB Oil Press Low (1)',
                       array=np.ma.concatenate((np.zeros(5), np.ones(6), np.zeros(19))),
                       values_mapping={0: '-', 1: 'Low Press'})
        warn_param_2 = M('MGB Oil Press Low (2)',
                       array=np.ma.concatenate((np.zeros(10), np.ones(5), np.zeros(15))),
                       values_mapping={0: '-', 1: 'Low Press'})
        airs = buildsection('Airborne', 1, 38)
        node = self.node_class()
        node.derive(None, mgb1=warn_param_1, mgb2=warn_param_2, airborne=airs)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[0].value, 10)


class TestCGBOilTempMax(unittest.TestCase):
    def setUp(self):
        self.node_class = CGBOilTempMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.CELSIUS)
        self.assertEqual(node.name, 'CGB Oil Temp Max')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),1)
        self.assertIn('Airborne', opts[0])
        self.assertIn('CGB Oil Temp', opts[0])

    def test_derive(self):
        temp = [78.0]*14 + [78.5,78] + [78.5]*23 + [79.0]*5 + [79.5] + [79.0]*5
        cgb_oil_temp = P('CGB Oil Temp', temp)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(cgb_oil_temp, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 79)
        self.assertEqual(node[0].index, 39)


class TestCGBOilPressMax(unittest.TestCase):
    def setUp(self):
        self.node_class = CGBOilPressMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.PSI)
        self.assertEqual(node.name, 'CGB Oil Press Max')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),1)
        self.assertIn('Airborne', opts[0])
        self.assertIn('CGB Oil Press', opts[0])

    def test_derive(self):
        press = [26.7] + [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*13
        cgb_oil_press = P('CGB Oil press', press)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(cgb_oil_press, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.63)
        self.assertEqual(node[0].index, 15)


class TestCGBOilPressMin(unittest.TestCase):
    def setUp(self):
        self.node_class = CGBOilPressMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.units, ut.PSI)
        self.assertEqual(node.name, 'CGB Oil Press Min')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)

        self.assertEqual(len(opts),1)
        self.assertIn('Airborne', opts[0])
        self.assertIn('CGB Oil Press', opts[0])

    def test_derive(self):
        press = [26.51]*14 + [26.63] + [26.4]*20 + [26.29] + [26.4]*13 + [26.1]
        cgb_oil_press = P('CGB Oil press', press)
        airborne = buildsection('Airborne', 1, 43)

        node = self.node_class()
        node.derive(cgb_oil_press, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 26.29)
        self.assertEqual(node[0].index, 35)


class TestHeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxSpecialProcedure(unittest.TestCase):

    def setUp(self):
        self.node_class = HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxSpecialProcedure

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(
            node.name,
            'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Special Procedure'
        )
        self.assertEqual(node.units, 'deg')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 4)
        self.assertIn('Heading Continuous', opts[0])
        self.assertIn('Distance To Touchdown', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive(self):

        heading = P('Heading Continuous', np.ma.array([
            -210, -209, -207, -206, -204, -201, -200, -199, -198, -197,
            -197, -196, -195, -195, -195, -194, -193, -193, -193, -193,
            -193, -193, -193, -193, -193, -193, -193, -193, -194, -194,
            -195, -195, -195, -195, -196, -197, -198, -200, -202, -204,
            -205, -207, -209, -211, -211, -210, -211, -211
        ]))

        dtts = DistanceToTouchdown('Distance To Touchdown',
                   items=[KeyTimeInstance(14, '0.8 NM To Touchdown'),
                          KeyTimeInstance(13, '1.0 NM To Touchdown'),
                          KeyTimeInstance(3, '1.5 NM To Touchdown'),
                          KeyTimeInstance(2, '2.0 NM To Touchdown')])

        tdwns = KTI(name='Offshore Touchdown', items=[KeyTimeInstance(index=20, name='Offshore Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                       slice(30, 36, None),
                                           runway_change=False,
                                           offset_ils=False,
                                           airport=None,
                                           landing_runway=None,
                                           approach_runway=None,
                                           gs_est=None,
                                           loc_est=None,
                                           ils_freq=None,
                                           turnoff=None,
                                           lowest_lat=-20.863417177,
                                           lowest_lon=115.404442795,
                                           lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(heading, dtts, tdwns, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 13)
        self.assertEqual(node[0].value, 11)

    def test_derive_no_special_procedures(self):

        heading = P('Heading Continuous', np.ma.array([
            -210, -209, -207, -206, -204, -201, -200, -199, -198, -197,
            -197, -196, -195, -195, -195, -194, -193, -193, -193, -193,
            -193, -193, -193, -193, -193, -193, -193, -193, -194, -194,
            -195, -195, -195, -195, -196, -197, -198, -200, -202, -204,
            -205, -207, -209, -211, -211, -210, -211, -211
        ]))

        dtts = DistanceToTouchdown('Distance To Touchdown',
                                   items=[KeyTimeInstance(14, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(13, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(3, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(2, '2.0 NM To Touchdown')])

        tdwns = KTI(name='Offshore Touchdown', items=[KeyTimeInstance(index=20, name='Offshore Touchdown')])

        approaches = App()
        approaches.create_approach('LANDING',
                                   slice(30, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(heading, dtts, tdwns, approaches)

        self.assertEqual(len(node), 0)


class TestHeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxStandardApproach(unittest.TestCase):

    def setUp(self):
        self.node_class = HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxStandardApproach

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(
            node.name,
            'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Standard Approach'
        )
        self.assertEqual(node.units, 'deg')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 4)
        self.assertIn('Heading Continuous', opts[0])
        self.assertIn('Distance To Touchdown', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive_no_offshore_touchdown(self):

        heading = P('Heading Continuous', np.ma.array([
            -210, -209, -207, -206, -204, -201, -200, -199, -198, -197,
            -197, -196, -195, -195, -195, -194, -193, -193, -193, -193,
            -193, -193, -193, -193, -193, -193, -193, -193, -194, -194,
            -195, -195, -195, -195, -196, -197, -198, -200, -202, -204,
            -205, -207, -209, -211, -211, -210, -211, -211
        ]))

        dtts = DistanceToTouchdown('Distance To Touchdown',
                   items=[KeyTimeInstance(14, '0.8 NM To Touchdown'),
                          KeyTimeInstance(13, '1.0 NM To Touchdown'),
                          KeyTimeInstance(3, '1.5 NM To Touchdown'),
                          KeyTimeInstance(2, '2.0 NM To Touchdown')])

        tdwns = KTI(name='Offshore Touchdown', items=[KeyTimeInstance(index=20, name='Offshore Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(30, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(heading, dtts, tdwns, approaches)

        self.assertEqual(len(node), 0)

    def test_derive_one_offshore_touchdown(self):

        heading = P('Heading Continuous', np.ma.array([
            -210, -209, -207, -206, -204, -201, -200, -199, -198, -197,
            -197, -196, -195, -195, -195, -194, -193, -193, -193, -193,
            -193, -193, -193, -193, -193, -193, -193, -193, -194, -194,
            -195, -195, -195, -195, -196, -197, -198, -200, -202, -204,
            -205, -207, -209, -211, -211, -210, -211, -211
        ]))

        dtts = DistanceToTouchdown('Distance To Touchdown',
                   items=[KeyTimeInstance(14, '0.8 NM To Touchdown'),
                          KeyTimeInstance(13, '1.0 NM To Touchdown'),
                          KeyTimeInstance(3, '1.5 NM To Touchdown'),
                          KeyTimeInstance(2, '2.0 NM To Touchdown'),
                          KeyTimeInstance(24, '0.8 NM To Touchdown'),
                          KeyTimeInstance(23, '1.0 NM To Touchdown'),
                          KeyTimeInstance(17, '1.5 NM To Touchdown'),
                          KeyTimeInstance(16, '2.0 NM To Touchdown')])

        tdwns = KTI(name='Offshore Touchdown', items=[KeyTimeInstance(index=32, name='Offshore Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(30, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(heading, dtts, tdwns, approaches)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 23)
        self.assertEqual(node[0].value, 0)


class TestTrackVariation100To50Ft(unittest.TestCase):

    def setUp(self):
        self.node_class = TrackVariation100To50Ft

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Track', 'Altitude AGL')])

    def test_derive(self):
        x = np.linspace(0, 10, 50)
        track = P(
            name='Track',
            array=-x*np.sin(x),
        )
        array = np.ma.arange(0, 250, 10)
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AGL', array)

        node = self.node_class()
        node.derive(track, alt)
        self.assertEqual(len(node), 1)
        self.assertEqual(node.name, 'Track Variation 100 To 50 Ft')
        self.assertEqual(node[0].index, 44) # index at 50ft
        self.assertAlmostEqual(node[0].value, 2.47, places=3) # PTP of section


class TestHeadingDuringLanding(unittest.TestCase):
    def setUp(self):
        self.node_class = HeadingDuringLanding

    def test_can_operate(self):
        combinations = self.node_class.get_operational_combinations()
        expected_combinations = [('Heading Continuous', 'Transition Flight To Hover'),]
        self.assertEqual(combinations, expected_combinations)

    def test_derive_basic(self):
        head = P('Heading Continuous',
                 np.ma.array([0,1,2,3,4,5,6,7,8,9,10,-1,-1,
                              7,-1,-1,-1,-1,-1,-1,-1,-10]))
        landing = buildsection('Transition Flight To Hover',11,15)
        head.array[13] = np.ma.masked
        kpv = self.node_class()
        kpv.derive(head, landing,)
        expected = [KeyPointValue(index=11, value=359.0,
                                  name='Heading During Landing')]
        self.assertEqual(kpv, expected)


class TestGroundspeed20FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Groundspeed20FtToTouchdownMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Groundspeed', 'Altitude AGL', 'Touchdown')])

    def test_derive(self):
        alt = P('Altitude AGL', np.ma.array(list(range(90, 0, -1))+[0]*10))
        spd = P('Groundspeed', np.ma.arange(100, 0, -1))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown')])

        node = self.node_class()
        node.derive(spd, alt, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 71)
        self.assertEqual(node[0].value, 29)


class TestGroundspeed20SecToOffshoreTouchdownMax(unittest.TestCase):
    def setUp(self):
        self.node_class = Groundspeed20SecToOffshoreTouchdownMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Groundspeed 20 Sec To Offshore Touchdown Max')
        self.assertEqual(node.units, 'kt')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 3)
        self.assertIn('Groundspeed', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])
        self.assertIn('Secs To Touchdown', opts[0])

    def test_derive(self):
        groundspeed = P('Groundspeed',
                        np.ma.array([45, 43, 39, 34, 30,
                                     22, 15,  7,  2,  1,
                                     1,   0,  0,  0,  0,
                                     50, 47, 44, 39, 32,
                                     30, 32, 31, 16,  8,
                                     8,   8,  7,  8,  8]))
        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(10, 'Offshore Touchdown'),
                                                     KeyTimeInstance(25, 'Offshore Touchdown')])
        secs_tdwn = SecsToTouchdown('Secs To Touchdown',
                        items=[KeyTimeInstance(1, '90 Secs To Touchdown'),
                               KeyTimeInstance(7, '30 Secs To Touchdown'),
                               KeyTimeInstance(8, '20 Secs To Touchdown'),
                               KeyTimeInstance(16, '90 Secs To Touchdown'),
                               KeyTimeInstance(22, '30 Secs To Touchdown'),
                               KeyTimeInstance(23, '20 Secs To Touchdown'),
                               KeyTimeInstance(29, '20 Secs To Touchdown')])

        node = self.node_class()
        node.derive(groundspeed, touchdown, secs_tdwn)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 8)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(node[1].index, 23)
        self.assertEqual(node[1].value, 16)


class TestGroundspeed0_8NMToOffshoreTouchdownSpecialProcedure(unittest.TestCase):

    def setUp(self):
        self.node_class = Groundspeed0_8NMToOffshoreTouchdownSpecialProcedure

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Groundspeed 0.8 NM To Offshore Touchdown Special Procedure')
        self.assertEqual(node.units, 'kt')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 4)
        self.assertIn('Groundspeed', opts[0])
        self.assertIn('Distance To Touchdown', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive(self):

        gnd_spd = np.linspace(57, 2, 25).tolist()
        gnd_spd += np.linspace(111, 7, 11).tolist()
        groundspeed = P('Groundspeed', np.ma.array(gnd_spd))

        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(23, 'Offshore Touchdown')])
        dtts = DistanceToTouchdown('Distance To Touchdown',
                                   items=[KeyTimeInstance(16, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(15, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(14, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(13, '2.0 NM To Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(34, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(groundspeed, dtts, touchdown, approaches)

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 16.0, places=1)
        self.assertAlmostEqual(node[0].value, 20.3, places=1)

    def test_derive_no_special_procedure(self):

        gnd_spd = np.linspace(57, 2, 25).tolist()
        gnd_spd += np.linspace(111, 7, 11).tolist()
        groundspeed = P('Groundspeed', np.ma.array(gnd_spd))

        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(23, 'Offshore Touchdown')])
        dtts = DistanceToTouchdown('Distance To Touchdown',
                                   items=[KeyTimeInstance(16, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(15, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(14, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(13, '2.0 NM To Touchdown')])

        approaches = App()
        approaches.create_approach('LANDING',
                                   slice(34, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(groundspeed, dtts, touchdown, approaches)

        self.assertEqual(len(node), 0)


class TestGroundspeed0_8NMToOffshoreTouchdownStandardApproach(unittest.TestCase):

    def setUp(self):
        self.node_class = Groundspeed0_8NMToOffshoreTouchdownStandardApproach

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Groundspeed 0.8 NM To Offshore Touchdown Standard Approach')
        self.assertEqual(node.units, 'kt')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(len(opts[0]), 4)
        self.assertIn('Groundspeed', opts[0])
        self.assertIn('Distance To Touchdown', opts[0])
        self.assertIn('Offshore Touchdown', opts[0])
        self.assertIn('Approach Information', opts[0])

    def test_derive_offshore_standard_landing(self):

        gnd_spd = np.linspace(57, 2, 25).tolist()
        gnd_spd += np.linspace(111, 7, 11).tolist()
        groundspeed = P('Groundspeed', np.ma.array(gnd_spd))

        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(33, 'Offshore Touchdown')])
        dtts = DistanceToTouchdown('Distance To Touchdown',
                                   items=[KeyTimeInstance(16, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(15, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(14, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(13, '2.0 NM To Touchdown'),
                                          KeyTimeInstance(25, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(26, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(27, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(28, '2.0 NM To Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                   slice(30, 36, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-20.863417177,
                                   lowest_lon=115.404442795,
                                   lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(groundspeed, dtts, touchdown, approaches)

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 25.0, places=1)
        self.assertAlmostEqual(node[0].value, 111.0, places=1)

    def test_derive_no_offshore_touchdown(self):

        gnd_spd = np.linspace(57, 2, 25).tolist()
        gnd_spd += np.linspace(111, 7, 11).tolist()
        groundspeed = P('Groundspeed', np.ma.array(gnd_spd))

        touchdown = KTI('Offshore Touchdown', items=[KeyTimeInstance(23, 'Offshore Touchdown')])
        dtts = DistanceToTouchdown('Distance To Touchdown',
                                   items=[KeyTimeInstance(16, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(15, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(14, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(13, '2.0 NM To Touchdown'),
                                          KeyTimeInstance(25, '0.8 NM To Touchdown'),
                                          KeyTimeInstance(26, '1.0 NM To Touchdown'),
                                          KeyTimeInstance(27, '1.5 NM To Touchdown'),
                                          KeyTimeInstance(28, '2.0 NM To Touchdown')])

        approaches = App()
        approaches.create_approach('AIRBORNE_RADAR',
                                   slice(19, 29, None),
                                   runway_change=False,
                                   offset_ils=False,
                                   airport=None,
                                   landing_runway=None,
                                   approach_runway=None,
                                   gs_est=None,
                                   loc_est=None,
                                   ils_freq=None,
                                   turnoff=None,
                                   lowest_lat=-19.92955434,
                                   lowest_lon=115.385025548,
                                   lowest_hdg=206.713600159)

        approaches.create_approach('LANDING',
                                       slice(30, 36, None),
                                       runway_change=False,
                                       offset_ils=False,
                                       airport=None,
                                       landing_runway=None,
                                       approach_runway=None,
                                       gs_est=None,
                                       loc_est=None,
                                       ils_freq=None,
                                       turnoff=None,
                                       lowest_lat=-20.863417177,
                                       lowest_lon=115.404442795,
                                       lowest_hdg=208.701438904)

        node = self.node_class()
        node.derive(groundspeed, dtts, touchdown, approaches)

        self.assertEqual(len(node), 0)


class TestGroundspeedBelow15FtFor20SecMax(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedBelow15FtFor20SecMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Groundspeed', 'Altitude AAL For Flight Phases', 'Airborne')])

    def test_derive(self):
        gspd = P('Groundspeed', np.ma.arange(0, 100))
        alt = P('Altitude AAL For Flight Phases', np.ma.array([0]*10 + list(range(0, 80)) + [80]*10))
        name = 'Airborne'
        section = Section(name, slice(10, 80), 10, 80)
        airborne = SectionNode(name, items=[section])
        node = self.node_class()
        node.derive(gspd, alt, airborne)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 25)
        self.assertEqual(node[0].value, 25)


class TestGroundspeedWhileAirborneWithASEOff(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedWhileAirborneWithASEOff

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Groundspeed', 'ASE Engaged', 'Airborne')])

    def test_derive(self):
        gspd = P('Groundspeed', np.ma.arange(0, 100))
        ase = M('ASE Engaged', np.ma.repeat([0,1,1,1,1,1,0,1,1,0], 10), values_mapping={1:'Engaged'})
        name = 'Airborne'
        section = Section(name, slice(10, 90), 9.5, 90.5)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(gspd, ase, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 69)
        self.assertEqual(node[0].value, 69)

    def test_derive__none(self):
        gspd = P('Groundspeed', np.ma.arange(0, 100))
        ase = M('ASE Engaged', np.ma.repeat([0,1,1,1,1,1,1,1,1,0], 10), values_mapping={1:'Engaged'})
        name = 'Airborne'
        section = Section(name, slice(10, 90), 9.5, 90.5)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(gspd, ase, airborne)

        self.assertEqual(len(node), 0)


class TestGroundspeedWhileHoverTaxiingMax(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedWhileHoverTaxiingMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Groundspeed', 'Hover Taxi')])

    def test_derive(self):
        gspd = P('Groundspeed', np.ma.arange(0, 100))
        name = 'Hover Taxi'
        section = Section(name, slice(2, 11), 1.5, 11)
        hover_taxi = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(gspd, hover_taxi)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertEqual(node[0].value, 10)


class TestGroundspeedWithZeroAirspeedFor5SecMax(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedWithZeroAirspeedFor5SecMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Wind Speed', 'Wind Direction', 'Groundspeed', 'Heading', 'Airborne')])

    def test_derive(self,):
        gnd_spd = P(
            name='Groundspeed',
            array=np.ma.ravel(np.ma.reshape(np.ma.array([4, 5, 6, 0, 0, 0, 0, 20, 10, 50]*10),(10,-1)).T),
            frequency=2
            )
        heading = P(
            name='Heading',
            array=np.ma.ravel(np.ma.reshape(np.ma.array([0, 90, 180, 270, 0, 0, 0, 0, 300, 300]*10),(10,-1)).T),
            frequency=2
            )
        wind_dir = P(
            name='Wind Direction',
            array=np.ma.ravel(np.ma.reshape(np.ma.array([0, 0, 0, 0, 90, 180, 270, 0, 180, 180]*10),(10,-1)).T),
            frequency=2
            )
        windspeed = P(
            name='Wind Speed',
            array=np.ma.ravel(np.ma.reshape(np.ma.array([10, 10, 10, 10, 10, 10, 10, 10, 100, 100]*10),(10,-1)).T),
            frequency=2
            )
        name = 'Airborne'
        section = Section(name, slice(0, 100), 0, 100)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(windspeed, wind_dir, gnd_spd, heading, airborne)

        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].index, 20)
        self.assertEqual(node[0].value, 6)
        self.assertEqual(node[1].index, 50)
        self.assertEqual(node[1].value, 0)
        self.assertEqual(node[2].index, 80)
        self.assertEqual(node[2].value, 10)


class TestGroundspeedBelow100FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = GroundspeedBelow100FtMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Groundspeed Below 100 Ft Max')
        self.assertEqual(node.units, 'kt')

    def test_can_operate(self):
        self.assertEqual(
            self.node_class.get_operational_combinations(ac_type=aeroplane),
            []
        )
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Groundspeed', opts[0])
        self.assertIn('Altitude AGL For Flight Phases', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        alt_agl_data = np.ma.array([
            500, 300, 100,  90,  60,  70,  75,  80,  90, 101,  80,  70,  80
        ])
        gnd_spd_data = np.ma.array([
            140, 145, 146, 100, 120, 122, 123, 119, 125, 146, 130, 125, 118
        ])

        alt_agl = P('Altitude AGL', alt_agl_data)
        gnd_spd = P('Groundspeed', gnd_spd_data)
        airborne = buildsection('Airborne', 1, 10)

        node = self.node_class()
        node.derive(gnd_spd, alt_agl, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[0].value, 146)


class TestPitchBelow1000FtMax(unittest.TestCase):

    def test_can_operate(self):
        opts = PitchBelow1000FtMax.get_operational_combinations(
            ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Altitude AGL')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 11, 12, 13, 8, 14, 6, 20])
        agl = P('Altitude AGL',
                array=[100, 200, 700, 1010, 4000, 1200, 1100, 900, 800])
        node = PitchBelow1000FtMax()
        node.derive(pitch, agl)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 11)
        self.assertEqual(node[1].value, 20)

class TestPitchBelow1000FtMin(unittest.TestCase):

    def test_can_operate(self):
        opts = PitchBelow1000FtMin.get_operational_combinations(
            ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Altitude AGL')])

    def test_derive(self):
        pitch = P('Pitch', array=[10, 10, 11, 12, 13, 8, 14, 6, 20])
        agl = P('Altitude AGL',
                array=[100, 200, 700, 1010, 4000, 1200, 1100, 900, 800])
        node = PitchBelow1000FtMin()
        node.derive(pitch, agl)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 10)
        self.assertEqual(node[1].value, 6)


class TestPitchBelow5FtMax(unittest.TestCase):
    def setUp(self):
        self.node_class = PitchBelow5FtMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEquals(node.name, 'Pitch Below 5 Ft Max')
        self.assertEquals(node.units, 'deg')

    def test_can_operate(self):
        self.assertEquals(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEquals(len(opts), 1)
        self.assertIn('Pitch',opts[0])
        self.assertIn('Altitude AGL', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        pitch = P('Pitch', np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1]))
        alt_agl = P('Altitude AGL', np.ma.array(np.linspace(0, 15, 9)))
        airborne = buildsection('Airborne', 1.4, 8)

        node = self.node_class()
        node.derive(pitch, alt_agl, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 4)
        self.assertEqual(node[0].index, 2)


class TestPitch5To10FtMax(unittest.TestCase):
    def setUp(self):
        self.node_class = Pitch5To10FtMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEquals(node.name, 'Pitch 5 To 10 Ft Max')
        self.assertEquals(node.units, 'deg')

    def test_can_operate(self):
        self.assertEquals(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEquals(len(opts), 1)
        self.assertIn('Pitch',opts[0])
        self.assertIn('Altitude AGL', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        arr = np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1])
        arr = np.ma.append(arr, arr[::-1])
        pitch = P('Pitch', arr)
        arr_alt = np.ma.array(np.linspace(0, 15, 9))
        arr_alt = np.ma.append(arr_alt, arr_alt[::-1])
        alt_agl = P('Altitude AGL', arr_alt)
        airborne = buildsection('Airborne', 1, 16)

        node = self.node_class()
        node.derive(pitch, alt_agl, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 9)
        self.assertEqual(node[0].index, 4)


class TestPitch10To5FtMax(unittest.TestCase):
    def setUp(self):
        self.node_class = Pitch10To5FtMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEquals(node.name, 'Pitch 10 To 5 Ft Max')
        self.assertEquals(node.units, 'deg')

    def test_can_operate(self):
        self.assertEquals(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEquals(len(opts), 1)
        self.assertIn('Pitch',opts[0])
        self.assertIn('Altitude AGL', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        arr = np.ma.array([0, 2, 4, 7, 9, 8, 6, 3, -1])
        arr = np.ma.append(arr, arr[::-1])
        pitch = P('Pitch', arr)
        arr_alt = np.ma.array(np.linspace(0, 15, 9))
        arr_alt = np.ma.append(arr_alt, arr_alt[::-1])
        alt_agl = P('Altitude AGL', arr_alt)
        airborne = buildsection('Airborne', 1, 16)

        node = self.node_class()
        node.derive(pitch, alt_agl, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].value, 9)
        self.assertEqual(node[0].index, 13)


class TestPitch500To100FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch500To100FtMax
        self.operational_combinations = [('Pitch', 'Altitude AGL', 'Descent')]
        self.can_operate_kwargs = {'ac_type': helicopter}
        self.function = min_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL For Flight Phases', 'Descent', 'Aircraft Type'), ac_type=helicopter))

    def test_derive(self):
        alt = P('Altitude AAL For Flight Phases', np.ma.arange(500, 0, -5))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        name = 'Descent'
        section = Section(name, slice(0, 100), 0, 100)
        descending = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, alt, descending)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 49)
        self.assertAlmostEqual(node[0].value, 4.811, places=3)


class TestPitch500To100FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch500To100FtMin
        self.operational_combinations = [('Pitch', 'Altitude AGL For Flight Phases', 'Descent')]
        self.can_operate_kwargs = {'ac_type': helicopter}
        self.function = min_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL For Flight Phases', 'Descent', 'Aircraft Type'), ac_type=helicopter))

    def test_derive(self):
        alt = P('Altitude AAL For Flight Phases', np.ma.arange(500, 0, -5))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        name = 'Descent'
        section = Section(name, slice(0, 100), 0, 100)
        descending = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, alt, descending)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)


class TestPitch100To20FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch100To20FtMax
        self.operational_combinations = [('Pitch', 'Altitude AGL', 'Descent')]
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.function = max_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL For Flight Phases', 'Descent', 'Aircraft Type'), ac_type=helicopter))

    def test_derive(self):
        alt = P('Altitude AAL For Flight Phases', np.ma.arange(500, 0, -5))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        name = 'Descent'
        section = Section(name, slice(0, 100), 0, 100)
        descending = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, alt, descending)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 96)
        self.assertAlmostEqual(node[0].value, 2.607, places=3)


class TestPitch100To20FtMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch100To20FtMin
        self.operational_combinations = [('Pitch', 'Altitude AGL For Flight Phases', 'Descent')]
        self.can_operate_kwargs = {'ac_type': aeroplane}
        self.function = min_value

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Pitch', 'Altitude AAL For Flight Phases', 'Final Approach', 'Aircraft Type'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Pitch', 'Altitude AGL For Flight Phases', 'Descent', 'Aircraft Type'), ac_type=helicopter))

    def test_derive(self):
        alt = P('Altitude AAL For Flight Phases', np.ma.arange(500, 0, -5))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        name = 'Descent'
        section = Section(name, slice(0, 100), 0, 100)
        descending = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(pitch, alt, descending)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 80)
        self.assertAlmostEqual(node[0].value, -7.874, places=3)


class TestPitch50FtToTouchdownMin(unittest.TestCase):

    def setUp(self):
        self.node_class = Pitch50FtToTouchdownMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Altitude AGL', 'Touchdown')])

    def test_derive(self):
        alt = P('Altitude AGL', np.ma.array(list(range(90, 0, -1))+[0]*10))
        x = np.linspace(0, 10, 100)
        pitch = P('Pitch', -x*np.sin(x))
        tdwns = KTI('Touchdown', items=[KeyTimeInstance(90, 'Touchdown')])

        node = self.node_class()
        node.derive(pitch, alt, tdwns)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)


class TestPitchOnGroundMax(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchOnGroundMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Pitch',  'Collective', 'Grounded', 'On Deck')])

    def test_derive(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'Grounded'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        grounded = SectionNode(name, items=[section, section2])
        on_deck = buildsection('On Deck', 98, 99)
        node = self.node_class()
        node.derive(pitch, coll, grounded, on_deck)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 2.5)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[1].index, 9)
        self.assertEqual(node[1].value, 0)

    def test_not_on_deck(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'Grounded'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        grounded = SectionNode(name, items=[section, section2])
        on_deck = buildsection('On Deck', 5, 99)
        node = self.node_class()
        node.derive(pitch, coll, grounded, on_deck)
        self.assertEqual(len(node), 1)

    def test_not_with_collective(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*5+[55.0]*5))
        name = 'Grounded'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        grounded = SectionNode(name, items=[section, section2])
        on_deck = buildsection('On Deck', 98, 99)
        node = self.node_class()
        node.derive(pitch, coll, grounded, on_deck)
        self.assertEqual(len(node), 1)


class TestPitchOnDeckMax(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchOnDeckMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Collective', 'On Deck')])

    def test_basic(self):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'On Deck'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        on_deck = SectionNode(name, items=[section, section2])
        node = self.node_class()
        node.derive(pitch, coll, on_deck)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 2.5)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[1].index, 9)
        self.assertEqual(node[1].value, 0)

    def test_with_coll_applied(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*5+[50.0]*5))
        name = 'On Deck'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        on_deck = SectionNode(name, items=[section, section2])
        node = self.node_class()
        node.derive(pitch, coll, on_deck)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 2.5)
        self.assertEqual(node[0].value, 3)


class TestPitchOnDeckMin(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchOnDeckMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Collective', 'On Deck')])

    def test_derive(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'On Deck'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        on_deck = SectionNode(name, items=[section, section2])
        node = self.node_class()
        node.derive(pitch, coll, on_deck)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)
        self.assertEqual(node[1].index, 8)
        self.assertEqual(node[1].value, -1)


class TestPitchOnGroundMin(unittest.TestCase):

    def setUp(self):
        self.node_class = PitchOnGroundMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Pitch', 'Collective', 'Grounded', 'On Deck')])

    def test_derive(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'Grounded'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        grounded = SectionNode(name, items=[section, section2])
        on_deck = buildsection('On Deck', 11, 99)
        node = self.node_class()
        node.derive(pitch, coll, grounded, on_deck)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)
        self.assertEqual(node[1].index, 8)
        self.assertEqual(node[1].value, -1)

    def test_not_on_deck(self,):
        pitch = P(
            name='Pitch',
            array=np.ma.array([0, 0, 2, 4, 7, 6, 4, 3, -1, 0]),
        )
        coll = P('Collective', np.ma.array([10.0]*10))
        name = 'Grounded'
        section = Section(name, slice(0, 3), 0, 2.5)
        section2 = Section(name, slice(8, 9), 8, 10)
        grounded = SectionNode(name, items=[section, section2])
        on_deck = buildsection('On Deck', 5, 99)
        node = self.node_class()
        node.derive(pitch, coll, grounded, on_deck)
        self.assertEqual(len(node), 1)


class TestRateOfDescent100To20FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent100To20FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AGL', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AGL', 'Descent'), ac_type=helicopter))

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 50, 5), np.arange(50, 500, 100)))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AGL', array, frequency=0.25)
        roc_array = np.ma.concatenate((np.ones(2) * 25, [62, 81, 100, 100, 50, 47, 35, 10, 35, 12, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed Inertial', roc_array, frequency=0.25)
        name = 'Descent'
        section = Section(name, slice(19, 27), 19, 27)
        descents = SectionNode(name, items=[section], frequency=0.25)

        node = self.node_class()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 100 To 20 Ft Max', items=[
            KeyPointValue(index=24.0, value=-100.0, name='Rate Of Descent 100 To 20 Ft Max')
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent500To100FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent500To100FtMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AGL', 'Final Approach'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed', 'Altitude AAL For Flight Phases', 'Final Approach'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AGL', 'Descent'), ac_type=helicopter))

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 50, 25), np.arange(50, 500, 100), [550, 450, 540], np.ones(5) * 590))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AGL', array, frequency=0.25)
        roc_array = np.ma.concatenate((np.ones(2) * 25, [62, 81, 100, 100, 50, 47, 35, 10, 35, 12, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed Inertial', roc_array, frequency=0.25)
        name = 'Descent'
        section = Section(name, slice(19, 27), 19, 27)
        descents = SectionNode(name, items=[section], frequency=0.25)

        node = self.node_class()
        node.derive(vert_spd, alt, descents)

        expected = KPV('Rate Of Descent 500 To 100 Ft Max', items=[
            KeyPointValue(index=24.0, value=-100.0, name='Rate Of Descent 500 To 100 Ft Max')
        ])
        self.assertEqual(node, expected)


class TestRateOfDescent20FtToTouchdownMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescent20FtToTouchdownMax

    def test_can_operate(self):
        self.assertFalse(self.node_class.can_operate([], ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate([], ac_type=helicopter))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AGL', 'Touchdown'), ac_type=aeroplane))
        self.assertFalse(self.node_class.can_operate(('Vertical Speed Inertial', 'Altitude AAL For Flight Phases', 'Touchdown'), ac_type=helicopter))
        self.assertTrue(self.node_class.can_operate(('Vertical Speed Inertial', 'Touchdown', 'Altitude AGL',), ac_type=helicopter))

    def test_derive(self):
        array = np.ma.concatenate((np.arange(0, 50, 5), [55, 45, 54], np.ones(5) * 59))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AAL For Flight Phases', array)
        roc_array = np.ma.concatenate((np.ones(8) * 5, [6, 2, 3, 3, 1, 3, 1, 0, 0, 0]))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        roc_array[33] = -26
        vert_spd = P('Vertical Speed Inertial', roc_array)

        touch_down = KTI('Touchdown', items=[KeyTimeInstance(34, 'Touchdown')])

        node = self.node_class()
        node.derive(vert_spd, touch_down, alt)

        expected = KPV('Rate Of Descent 20 Ft To Touchdown Max', items=[
            KeyPointValue(name='Rate Of Descent 20 Ft To Touchdown Max', index=33, value=-26),
        ])
        self.assertEqual(node, expected)


class TestRateOfDescentBelow500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescentBelow500FtMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Vertical Speed Inertial', 'Altitude AGL For Flight Phases', 'Descending')])

    def test_derive(self,):
        array = np.ma.concatenate((np.arange(0, 2000, 100), np.arange(5000, 10000, 1000)))
        array = np.ma.concatenate((array, array[::-1]))
        alt = P('Altitude AGL For Flight Phases', array, frequency=0.25)
        roc_array = np.ma.concatenate(([437, 625, 812, 1000, 1125, 625, 475, 500, 125, 375, 275], np.ones(14) * 250))
        roc_array = np.ma.concatenate((roc_array, -roc_array[::-1]))
        vert_spd = P('Vertical Speed Inertial', roc_array, frequency=0.25)
        name = 'Descending'
        section = Section(name, slice(25, 48), 25, 48)
        descent = SectionNode(name, items=[section], frequency=0.25)

        node = self.node_class()
        node.derive(vert_spd, alt, descent)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 45)
        self.assertEqual(node[0].value, -1125)


class TestRateOfDescentBelow30KtsWithPowerOnMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RateOfDescentBelow30KtsWithPowerOnMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Vertical Speed Inertial', 'Airspeed', 'Descending', 'Eng (*) Torque Avg')])
        opts = self.node_class.get_operational_combinations(ac_type=aeroplane)
        self.assertEqual(opts, [])

    def test_derive(self,):
        x = np.linspace(0, 10, 62)
        vrt_spd = P('Vertical Speed', -x*np.sin(x) * 100)
        air_spd = P(
            name='Airspeed',
            array=np.ma.array(list(range(-2, 90, 3)) + list(range(90, -2, -3))),
        )

        air_spd.array[0] = 0
        name = 'Descending'
        section_1 = Section(name, slice(2, 12), 1.5, 11.5)
        section_2 = Section(name, slice(39, 49), 38.5, 48.5)
        descending = SectionNode(name, items=[section_1, section_2])

        power = P(name='Eng (*) Torque Avg', array=np.ma.array([35.0]*62,dtype=float))
        node = self.node_class()
        node.derive(vrt_spd, air_spd, descending, power)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertAlmostEqual(node[0].value, -164, places=0)

    def test_low_power(self,):
        x = np.linspace(0, 10, 62)
        vrt_spd = P('Vertical Speed', -x*np.sin(x) * 100)
        air_spd = P(
            name='Airspeed',
            array=np.ma.array(list(range(-2, 90, 3)) + list(range(90, -2, -3))),
        )

        air_spd.array[0] = 0
        name = 'Descending'
        section_1 = Section(name, slice(2, 12), 1.5, 11.5)
        section_2 = Section(name, slice(39, 49), 38.5, 48.5)
        descending = SectionNode(name, items=[section_1, section_2])

        power = P(name='Eng (*) Torque Avg', array=np.ma.array([15.0]*62,dtype=float))
        node = self.node_class()
        node.derive(vrt_spd, air_spd, descending, power)

        self.assertEqual(len(node), 0)


class TestVerticalSpeedAtAltitude(unittest.TestCase):
    def setUp(self):
        self.node_class = VerticalSpeedAtAltitude

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Vertical Speed At Altitude')
        self.assertIn('Vertical Speed At 300 Ft', node.names())
        self.assertIn('Vertical Speed At 500 Ft', node.names())
        self.assertEqual(node.units, 'fpm')

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(
            ac_type=aeroplane)
        self.assertEqual(opts, [])

        opts = self.node_class.get_operational_combinations(
            ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertEqual(opts[0], ('Vertical Speed',
                                   'Altitude AGL',
                                   'Approach'))

    def test_derive(self):
        x = np.linspace(0, 12, 70)
        vert_spd = P('Vertical Speed', x*np.sin(x) * 50)
        approaches = buildsections('Approach', [25,30], [60, 65])
        y = np.linspace(190, 403, 17).tolist() + \
            np.linspace(415, 201, 18).tolist() + \
            np.linspace(230, 534, 17).tolist() + \
            np.linspace(503, 208, 18).tolist()
        alt_agl = P('Altitude AGL', y)

        node = self.node_class()
        node.derive(vert_spd, alt_agl, approaches)

        self.assertEqual(len(node), 4)
        self.assertAlmostEqual(node[0].index, 25, places=0)
        self.assertAlmostEqual(node[1].index, 26, places=0)
        self.assertAlmostEqual(node[2].index, 60, places=0)
        self.assertAlmostEqual(node[3].index, 64, places=0)
        self.assertTrue(node[0].name == node[2].name == \
                        'Vertical Speed At 500 Ft')
        self.assertTrue(node[1].name == node[3].name == \
                        'Vertical Speed At 300 Ft')


class TestRoll100To20FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = Roll100To20FtMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Altitude AGL For Flight Phases', 'Descent')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.arange(500, 0, -5), frequency=0.25)
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x), frequency=0.25)
        name = 'Descent'
        section = Section(name, slice(1, 95), 1, 95)
        descent = SectionNode(name, items=[section], frequency=0.25)

        node = self.node_class()
        node.derive(roll, alt, descent)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 80)
        self.assertAlmostEqual(node[0].value, -7.874, places=3)


class TestRollAbove300FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollAbove300FtMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Altitude AGL For Flight Phases')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.arange(0, 5000, 50))
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))

        node = self.node_class()
        node.derive(roll, alt)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)


class TestRollBelow300FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollBelow300FtMax

    def test_attribute(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Roll Below 300 Ft Max')
        self.assertEqual(node.units, 'deg')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Roll', opts[0])
        self.assertIn('Altitude AGL For Flight Phases', opts[0])
        self.assertIn('Airborne', opts[0])
        #self.assertEqual(opts, [('Roll', 'Altitude AGL')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.arange(0, 5000, 50))
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))
        airborne = buildsections('Airborne', [2, 49])

        node = self.node_class()
        node.derive(roll, alt, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 6)
        self.assertAlmostEqual(node[0].value, -0.345, places=3)


class TestRollWithAFCSDisengagedMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollWithAFCSDisengagedMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Roll With AFCS Disengaged Max')
        self.assertEqual(node.units, 'deg')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'AFCS (1) Engaged',
                                 'AFCS (2) Engaged')])

    def test_derive(self):
        afcs1 = M('AFCS (1) Engaged',
                array=np.ma.array([0]*5 + [1]*10 + [0]*50 + [1]*30 + [0]*5),
                values_mapping={0: '-', 1: 'Engaged'})
        afcs2 = M('AFCS (2) Engaged',
                array=np.ma.array([0]*5 + [1]*30 + [0]*65),
                values_mapping={0: '-', 1: 'Engaged'})
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))

        node = self.node_class()
        node.derive(roll, afcs1, afcs2)

        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].index, 4)
        self.assertAlmostEqual(node[0].value, -0.1588, places=3)
        self.assertEqual(node[1].index, 49)
        self.assertAlmostEqual(node[1].value, 4.8110, places=3)
        self.assertEqual(node[2].index, 99)
        self.assertAlmostEqual(node[2].value, 5.4402, places=3)

class TestRollAbove500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollAbove500FtMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Altitude AGL For Flight Phases')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.arange(0, 5000, 50))
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))

        node = self.node_class()
        node.derive(roll, alt)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 79)
        self.assertAlmostEqual(node[0].value, -7.917, places=3)

class TestRollBelow500FtMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollBelow500FtMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Altitude AGL For Flight Phases')])

    def test_derive(self):
        alt = P('Altitude AGL For Flight Phases', np.ma.arange(0, 5000, 50))
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))

        node = self.node_class()
        node.derive(roll, alt)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 10)
        self.assertAlmostEqual(node[0].value, -0.855, places=3)


class TestRollOnGroundMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollOnGroundMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Collective', 'Grounded', 'On Deck')])

    def test_derive(self,):
        x = np.linspace(0, 10, 100)
        roll = P('Roll', x*np.sin(x))
        coll = P('Collective', np.ma.array([10.0]*100))
        name = 'Grounded'
        section = Section(name, slice(10, 50), 10, 50)
        grounded = SectionNode(name, items=[section])

        section = Section('On Deck', slice(80, 90), 80, 90)
        on_deck = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, coll, grounded, on_deck)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 49)
        self.assertAlmostEqual(node[0].value, -4.811, places=3)

    def test_not_on_deck(self,):
        x = np.linspace(0, 10, 100)
        roll = P('Roll', -x*np.sin(x))
        coll = P('Collective', np.ma.array([10.0]*100))
        name = 'Grounded'
        section = Section(name, slice(10, 50), 10, 50)
        grounded = SectionNode(name, items=[section])

        section = Section('On Deck', slice(10, 50), 10, 50)
        on_deck = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, coll, grounded, on_deck)

        self.assertEqual(len(node), 0)


class TestRollOnDeckMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollOnDeckMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll', 'Collective', 'On Deck')])

    def test_derive(self,):
        x = np.linspace(0, 10, 100)
        roll = P('Roll', x*np.sin(x))
        coll = P('Collective', np.ma.array([10.0]*100))
        name = 'On Deck'
        section = Section(name, slice(10, 50), 10, 50)
        on_deck = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(roll, coll, on_deck)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 49)
        self.assertAlmostEqual(node[0].value, -4.811, places=3)


class TestRollRateMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RollRateMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Roll Rate', 'Airborne')])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        roll_rate= P('Roll Rate', np.sin(x)*20+x)
        airborne = buildsection('Airborne', 0, 180)
        node = self.node_class()
        node.derive(roll_rate, airborne)
        self.assertEqual(len(node), 3)
        self.assertEqual(node[0].index, 32)
        self.assertEqual(node[1].index, 93)
        self.assertEqual(node[2].index, 157)

    def test_multiple_flights(self):
        x = np.linspace(0, 10, 200)
        roll_rate= P('Roll Rate', np.sin(x)*20+x)
        airborne = buildsections('Airborne', [9,60], [130,180])
        node = self.node_class()
        node.derive(roll_rate, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 32)
        self.assertEqual(node[1].index, 157)

    def test_not_below_five(self):
        x = np.linspace(0, 10, 200)
        roll_rate= P('Roll Rate', np.sin(x)*6+x/4.0)
        airborne = buildsection('Airborne', 0, 200)
        node = self.node_class()
        node.derive(roll_rate, airborne)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 32)
        self.assertEqual(node[1].index, 157)


##############################################################################
# Rotor


class TestRotorSpeedDuringAutorotationAbove108KtsMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedDuringAutorotationAbove108KtsMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'Rotor Speed During Autorotation Above 108 Kts Min')
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])
        self.assertIn('Autorotation', opts[0])
        self.assertIn('Airspeed', opts[0])

    def test_derive(self,):
        air_spd = P('Airspeed',
                    np.ma.array(list(range(50, 150)) + list(range(150, 50, -1))))
        x = np.linspace(0, 10, 200)
        rtr_spd = P('Nr', array=np.ma.array(np.sin(x)+100))
        autorotation = buildsection('Autorotation', 115, 152)

        node = self.node_class()
        node.derive(rtr_spd, air_spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 115)
        self.assertAlmostEqual(node[0].value, 99.517, places=3)


class TestRotorSpeedDuringAutorotationBelow108KtsMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedDuringAutorotationBelow108KtsMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name,
                         'Rotor Speed During Autorotation Below 108 Kts Min')
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])
        self.assertIn('Autorotation', opts[0])
        self.assertIn('Airspeed', opts[0])

    def test_derive(self):
        air_spd = P('Airspeed',
                    np.ma.array(list(range(50, 150)) + list(range(150, 50, -1))))
        x = np.linspace(0, 10, 200)
        rtr_spd = P('Nr', array=np.ma.array(np.sin(x)+100))
        autorotation = buildsection('Autorotation', 115, 152)

        node = self.node_class()
        node.derive(rtr_spd, air_spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 142)
        self.assertAlmostEqual(node[0].value, 100.753, places=3)


class TestRotorSpeedDuringAutorotationMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedDuringAutorotationMax

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Rotor Speed During Autorotation Max')
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])
        self.assertIn('Autorotation', opts[0])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        rtr_spd = P('Nr', array=np.ma.array(np.sin(x)+100))
        autorotation = buildsection('Autorotation', 115, 152)

        node = self.node_class()
        node.derive(rtr_spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 151)
        self.assertAlmostEqual(node[0].value, 100.965, places=3)


class TestRotorSpeedDuringAutorotationMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedDuringAutorotationMin

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Rotor Speed During Autorotation Min')
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])
        self.assertIn('Autorotation', opts[0])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        rtr_spd = P('Nr', array=np.ma.array(np.sin(x)+100))
        autorotation = buildsection('Autorotation', 115, 152)

        node = self.node_class()
        node.derive(rtr_spd, autorotation)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 115)
        self.assertAlmostEqual(node[0].value, 99.5, places=1)


class TestRotorSpeedWhileAirborneMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedWhileAirborneMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Nr', 'Airborne', 'Autorotation')])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        name = 'Airborne'
        section = Section(name, slice(115, 152), 115, 152)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(rotor, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 151)
        self.assertAlmostEqual(node[0].value, 100.965, places=3)

    def test_excluding_autorotation(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        airborne = buildsection('Airborne', 70, 199)
        auto = buildsection('Autorotation', 136, 178)
        node = self.node_class()
        node.derive(rotor, airborne, auto)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 135)
        self.assertAlmostEqual(node[0].value, 100.480, places=3)


class TestRotorSpeedWhileAirborneMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedWhileAirborneMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Nr', 'Airborne', 'Autorotation')])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        name = 'Airborne'
        section = Section(name, slice(115, 152), 115, 152)
        airborne = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(rotor, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 115)
        self.assertAlmostEqual(node[0].value, 99.517, places=3)

    def test_excluding_autorotation(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        airborne = buildsection('Airborne', 30, 155)
        auto = buildsection('Autorotation', 72, 114)
        node = self.node_class()
        node.derive(rotor, airborne, auto)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 114)
        self.assertAlmostEqual(node[0].value, 99.473, places=3)

class TestRotorSpeedWithRotorBrakeAppliedMax(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedWithRotorBrakeAppliedMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Nr', 'Rotor Brake Engaged')])  # TODO: check naming "Rotor Brake"/"Rotor Brake On" and "Rotor"/"Nr"

    def test_derive(self,):
        values_mapping = {0: '-', 1: 'Engaged'}
        rotor_brk = M(
            'Rotor Brake', values_mapping=values_mapping,
            array=np.ma.array(
                [0] * 40 + [1] * 20 + [0] * 80 + [1] * 20 + [0] * 40))
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        node = self.node_class()
        node.derive(rotor, rotor_brk)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].index, 40)
        self.assertAlmostEqual(node[0].value, 100.905, places=3)
        self.assertEqual(node[1].index, 156)
        self.assertAlmostEqual(node[1].value, 101.000, places=3)

    def test_two_samples_not_one(self,):
        values_mapping = {0: '-', 1: 'Engaged'}
        rotor_brk = M(
            'Rotor Brake', values_mapping=values_mapping,
            array=np.ma.array(
                [0] * 40 + [1] * 1 + [0] * 80 + [1] * 2 + [0] * 40))
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        node = self.node_class()
        node.derive(rotor, rotor_brk)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 122)


class TestRotorsRunningDuration(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorsRunningDuration

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Rotors Running',)])

    def test_derive(self):
        running = M('Rotors Running', np.ma.array([0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0]),
                    values_mapping={0: 'Not Running', 1: 'Running',})

        node = self.node_class()
        node.derive(running)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 11)
        self.assertEqual(node[0].value, 7)


class TestRotorSpeedDuringMaximumContinuousPowerMin(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeedDuringMaximumContinuousPowerMin

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Nr', 'Maximum Continuous Power', 'Autorotation')])

    def test_derive(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        name = 'Maximum Continuous Power'
        section = Section(name, slice(115, 152), 115, 152)
        mcp = SectionNode(name, items=[section])

        node = self.node_class()
        node.derive(rotor, mcp)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 115)
        self.assertAlmostEqual(node[0].value, 99.517, places=3)

    def test_excluding_autorotation(self):
        x = np.linspace(0, 10, 200)
        rotor = P('Rotor', array=np.ma.array(np.sin(x)+100))
        mcp = buildsection('Maximum Continuous Power', 30, 155)
        auto = buildsection('Autorotation', 72, 114)
        node = self.node_class()
        node.derive(rotor, mcp, auto)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 114)
        self.assertAlmostEqual(node[0].value, 99.473, places=3)


class TestRotorSpeed36To49Duration(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeed36To49Duration

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Rotor Speed 36 To 49 Duration')
        self.assertEqual(node.units, 's')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(
            ac_type=helicopter, family=A('Family', 'S92'))
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])

    def test_derive(self):
        nr = P('Nr', np.ma.array([10, 13, 24, 30, 36, 40, 46,
                                  49, 55, 60, 48, 45, 35, 20]))
        node = self.node_class()
        node.derive(nr)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[1].value, 2)
        self.assertEqual(node[1].index, 10)


class TestRotorSpeed56To67Duration(unittest.TestCase):

    def setUp(self):
        self.node_class = RotorSpeed56To67Duration

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(node.name, 'Rotor Speed 56 To 67 Duration')
        self.assertEqual(node.units, 's')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(
            ac_type=helicopter, family=A('Family', 'S92'))
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])

    def test_derive(self):
        nr = P('Nr', np.ma.array([10, 23, 34, 40, 56, 60, 66,
                                  67, 68, 69, 66, 58, 54, 40]))
        node = self.node_class()
        node.derive(nr)

        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 2)
        self.assertEqual(node[0].index, 5)
        self.assertEqual(node[1].value, 2)
        self.assertEqual(node[1].index, 10)


class TestRotorSpeedAt6PercentCollectiveDuringEngStart(unittest.TestCase):
    def setUp(self):
        self.node_class = RotorSpeedAt6PercentCollectiveDuringEngStart

    def test_attributes(self):
        node = self.node_class()
        self.assertEqual(
            node.name,
            'Rotor Speed At 6 Percent Collective During Eng Start'
        )
        self.assertEqual(node.units, '%')

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(
            ac_type=helicopter, family=A('Family', 'S92'))
        self.assertEqual(len(opts), 1)
        self.assertIn('Nr', opts[0])

    def test_derive(self):
        nr = P('Nr', np.ma.array([
            0, 0, 0, 0, 0, 0, 0.2, 0, 0.5, 0.5,
            0.8, 0.8, 0.8, 1, 1.2, 1.5, 1.8, 2.2, 2.5, 3,
            3.5, 4, 4.2, 4.8, 5.2, 6, 6.5, 7.5, 8.5, 9.5,
            10.2, 11.2, 12, 13, 14, 15, 15.8, 16.5, 17.5, 18.2, 19,
        ]))
        collective = P('Collective', np.ma.array([
            61.5, 61.25, 61.25, 61.38, 61.5, 61.38, 61.38, 61.38, 61.25, 61.5,
            61.38, 61.38, 61.25, 61.3, 61.38, 61.25, 61.5, 61.38, 61.38, 61.38,
            61.5, 61.5, 61.5, 61.25, 61.38, 60.62, 55.5, 52.62, 43.88, 35.75,
            25.38, 14.38, 4.5, 3.88, 4, 4.38, 4.25, 4, 4.12, 4, 4,
        ]))
        firsts = KTI('First Eng Fuel Flow Start', items=[
            KeyTimeInstance(2, 'First Eng Fuel Flow Start'),
        ])

        node = self.node_class()
        node.derive(nr, collective, firsts)

        self.assertEqual(len(node), 1)
        self.assertAlmostEqual(node[0].index, 31.85, places=2)
        self.assertAlmostEqual(node[0].value, 11.88, places=2)

    def test_derive__rotors_running_data_start(self):
        nr = P('Nr', np.ma.array([102*41]))
        collective = P('Collective', 65 - np.ma.array([
            61.5, 61.25, 61.25, 61.38, 61.5, 61.38, 61.38, 61.38, 61.25, 61.5,
            61.38, 61.38, 61.25, 61.3, 61.38, 61.25, 61.5, 61.38, 61.38, 61.38,
            61.5, 61.5, 61.5, 61.25, 61.38, 60.62, 55.5, 52.62, 43.88, 35.75,
            25.38, 14.38, 4.5, 3.88, 4, 4.38, 4.25, 4, 4.12, 4, 4,
        ]))
        firsts = KTI('First Eng Fuel Flow Start', items=[
            KeyTimeInstance(2, 'First Eng Fuel Flow Start'),
        ])

        node = self.node_class()
        node.derive(nr, collective, firsts)

        self.assertEqual(len(node), 0)


class TestWindSpeedInCriticalAzimuth(unittest.TestCase):

    def setUp(self):
        self.node_class = WindSpeedInCriticalAzimuth

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('Wind Speed', 'Wind Direction', 'Airspeed True', 'Heading', 'Airborne')])

    def test_derive(self):
        airspeed = P(
            name='Airspeed True',
            array=np.ma.array([0, 0, 0, 0, 0, 0, 0, 20, 0, 50]),
            frequency=2
            )
        heading = P(
            name='Heading',
            array=np.ma.array([0, 90, 180, 270, 0, 0, 0, 0, 300, 300]),
            frequency=2
            )
        wind_dir = P(
            name='Wind Direction',
            array=np.ma.array([0, 0, 0, 0, 90, 180, 270, 0, 180, 180]),
            frequency=2
            )
        windspeed = P(
            name='Wind Speed',
            array=np.ma.array([10, 10, 10, 10, 10, 10, 10, 10, 100, 100]),
            frequency=2
            )
        name = 'Airborne'
        section = Section(name, slice(0, 10), 0, 10)
        airborne = SectionNode(name, items=[section], frequency=2)

        node = self.node_class(frequency=2)
        node.derive(windspeed, wind_dir, airspeed, heading, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 8)
        self.assertAlmostEqual(node[0].value, 100, places=0)


class TestSATMin(unittest.TestCase):

    def setUp(self):
        self.node_class = SATMin

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('SAT', 'Family', 'Rotors Turning')])

    def test_derive(self,):
        sat = P('SAT', np.ma.arange(0, 11))
        node = self.node_class()
        node.derive(sat)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 0)
        self.assertEqual(node[0].value, 0)

    def test_derive_s92(self,):
        sat = P('SAT', np.ma.arange(15,26))
        family = A('Family', value='S92')
        rotors_turning = buildsection('Rotors Turning', 3, 9)

        node = self.node_class()
        node.derive(sat, family, rotors_turning)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 3)
        self.assertEqual(node[0].value, 18)

class TestSATRateOfChangeMax(unittest.TestCase):

    def setUp(self):
        self.node_class = SATRateOfChangeMax

    def test_can_operate(self):
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, [('SAT', 'Airborne')])

    def test_basic(self):
        sat = P('SAT', np.ma.arange(20))
        air = buildsection('Airborne', 0, 20)
        node = self.node_class()
        node.derive(sat, air)
        self.assertEqual(node[0].value, 1.0)

    def test_pulses(self):
        sat = P('SAT', np.ma.array([0.0]*10+[20.0]+[0.0]*10+[30.0]+[0.0]*10))
        air = buildsection('Airborne', 12, 30)
        node = self.node_class()
        node.derive(sat, air)
        # The 4-sec differentiation window makes the peak slope arise
        # 2 samples before the peak at t=21. Hence this index.
        self.assertEqual(node[0].index, 19.0)
        # The 4-sec differentiation window makes the peak slope appear
        # 30C/4 = 7.5 C/sec. Hence this index.
        self.assertEqual(node[0].value, 7.5)


class TestCruiseGuideIndicatorMax(unittest.TestCase):

    def setUp(self):
        self.node_class = CruiseGuideIndicatorMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(len(opts), 1)
        self.assertIn('Cruise Guide', opts[0])
        self.assertIn('Airborne', opts[0])

    def test_derive(self):
        cgi = P('CGI', array=np.ma.array([-60, 0, 10, 20, 30, 40, -30, -50, 30, 20, 10, 0]))
        airborne = buildsection('Airborne', 1,10)
        node = self.node_class()
        node.derive(cgi, airborne)

        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 7)
        self.assertEqual(node[0].value, -50)

class TestTrainingModeDuration(unittest.TestCase):

    def test_can_operate(self):
        self.assertEqual(TrainingModeDuration.get_operational_combinations(),
                         [('Training Mode',), ('Eng (1) Training Mode', 'Eng (2) Training Mode')])

    def test_derive_S92A(self):
        trg=P('Training Mode', np.array([0,0,1,1,1,0,0,1,1,0,0]))
        node = TrainingModeDuration()
        node.derive(trg, None, None)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[1].value, 2)
        self.assertEqual(node[1].index, 7)

    def test_derive_H225(self):
        trg1=P('Eng (1) Training Mode', np.array([0,0,1,1,1,0,0,0,0,0,0]))
        trg2=P('Eng (2) Training Mode', np.array([0,0,0,0,0,0,0,1,1,0,0]))
        node = TrainingModeDuration()
        node.derive(None, trg1, trg2)
        self.assertEqual(len(node), 2)
        self.assertEqual(node[0].value, 3)
        self.assertEqual(node[0].index, 2)
        self.assertEqual(node[1].value, 2)
        self.assertEqual(node[1].index, 7)


class TestHoverHeightDuringOnshoreTakeoffMax(unittest.TestCase):

    def setUp(self):
        self.node_class = HoverHeightDuringOnshoreTakeoffMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, ([('Altitude Radio', 'Offshore', 'Hover', 'Takeoff')]))

    def test_derive(self):
        alt=P('Altitude Radio', np.ma.array(data=[2,3,4,5,3,3,4,3,2,7,8.0],
                                            mask=[0,0,0,1,0,0,0,0,0,0,0]))
        offshore = M(name='Offshore', array=np.ma.array([0]*12, dtype=int),
                         values_mapping={0: 'Onshore', 1: 'Offshore'})
        toff = buildsection('Takeoff', 5, 8)
        hover=buildsections('Hover', [2, 4], [7, 10])
        node=self.node_class()
        node.derive(alt, offshore, hover, toff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 9)
        self.assertEqual(node[0].value, 7)

class TestHoverHeightDuringOffshoreTakeoffMax(unittest.TestCase):

    def setUp(self):
        self.node_class = HoverHeightDuringOffshoreTakeoffMax

    def test_can_operate(self):
        self.assertEqual(self.node_class.get_operational_combinations(
            ac_type=aeroplane), [])
        opts = self.node_class.get_operational_combinations(ac_type=helicopter)
        self.assertEqual(opts, ([('Altitude Radio', 'Offshore', 'Hover', 'Takeoff')]))

    def test_derive(self):
        alt=P('Altitude Radio', np.ma.array(data=[2,3,4,5,3,3,4,3,2,7,8.0],
                                            mask=[0,0,0,1,0,0,0,0,0,0,0]))
        offshore = M(name='Offshore', array=np.ma.array([1]*12, dtype=int),
                         values_mapping={0: 'Onshore', 1: 'Offshore'})
        toff = buildsection('Takeoff', 5, 8)
        hover=buildsections('Hover', [2, 4], [7, 10])
        node=self.node_class()
        node.derive(alt, offshore, hover, toff)
        self.assertEqual(len(node), 1)
        self.assertEqual(node[0].index, 9)
        self.assertEqual(node[0].value, 7)
