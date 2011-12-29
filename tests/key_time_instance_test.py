import unittest
import numpy as np

from analysis.plot_flight import plot_parameter
from analysis.node import KeyTimeInstance, Parameter, P, Section, S
from analysis.flight_phase import (ApproachAndLanding,
                                   ClimbCruiseDescent,
                                   Climbing,
                                   DescentLowClimb,
                                   )
from analysis.derived_parameters import (ClimbForFlightPhases,
                                         )
from analysis.key_time_instances import (BottomOfDescent,
                                         ClimbStart,
                                         GoAround,
                                         AltitudeInApproach,
                                         AltitudeInFinalApproach,
                                         AltitudeWhenClimbing,
                                         AltitudeWhenDescending,
                                         InitialClimbStart,
                                         LandingPeakDeceleration,
                                         LandingStart,
                                         LandingStartDeceleration,
                                         LandingTurnOffRunway,
                                         Liftoff,
                                         TakeoffStartAcceleration,
                                         TakeoffTurnOntoRunway,
                                         TopOfClimb,
                                         TopOfDescent,
                                         Touchdown
                                         )

import sys
debug = sys.gettrace() is not None

class TestBottomOfDescent(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_bottom_of_descent_basic(self):
        testwave = np.cos(np.arange(0,12.6,0.1))*(-2500)+12500
        alt_ph = Parameter('Altitude For Flight Phases', np.ma.array(testwave))
        alt_std = Parameter('Altitude STD', np.ma.array(testwave))
        dlc = DescentLowClimb()
        dlc.derive(alt_ph)
        bod = BottomOfDescent()
        bod.derive(dlc, alt_std)    
        expected = [KeyTimeInstance(index=63, name='Bottom Of Descent')]        
        self.assertEqual(bod, expected)
        
        
class TestClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Altitude AAL', 'Climbing')]
        opts = ClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_climb_start_basic(self):
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc, [Section('Fast',slice(0,8,None))])
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        # These values give an result with an index of 4.5454 recurring.
        expected = [KeyTimeInstance(index=5/1.1, name='Climb Start')]
        self.assertEqual(kpi, expected)


    def test_climb_start_cant_climb_when_slow(self):
        roc = Parameter('Rate Of Climb', np.ma.array([1200]*8))
        climb = Climbing()
        climb.derive(roc, []) #  No Fast phase found in this data
        alt = Parameter('Altitude AAL', np.ma.array(range(0,1600,220)))
        kpi = ClimbStart()
        kpi.derive(alt, climb)
        expected = [] #  Even though the altitude climbed, the a/c can't have
        self.assertEqual(kpi, expected)


class TestGoAround(unittest.TestCase):
    def test_can_operate(self):
        
        expected = [('Altitude AAL For Flight Phases',
                     'Approach And Landing',
                     'Climb For Flight Phases'),
                    ('Altitude AAL For Flight Phases',
                     'Altitude Radio For Flight Phases',
                     'Approach And Landing',
                     'Climb For Flight Phases'),
                    ]
        opts = GoAround.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_go_around_basic(self):
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,1000,501))
        aal = [Section('Approach And Landing',slice(10,18))]
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), aal)
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   aal, climb)
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_multiple_go_arounds(self):
        # This tests for three go-arounds, but the fourth part of the curve
        # does not produce a go-around as it ends in mid-descent.
        alt = np.ma.array(np.cos(np.arange(0,21,0.02))*(1000)+2500)

        if debug:
            plot_parameter(alt)
            
        aal = ApproachAndLanding()
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
            
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), 
                     [Section('Fast',slice(0,len(alt),None))])
        
        goa = GoAround()
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt),
                   aal, climb)
                   
        expected = [KeyTimeInstance(index=157, name='Go Around'), 
                    KeyTimeInstance(index=471, name='Go Around'), 
                    KeyTimeInstance(index=785, name='Go Around')]
        self.assertEqual(goa, expected)

    def test_go_around_insufficient_climb(self):
        # 500 ft climb is not enough to trigger the go-around. 
        # Compare to 501 ft for the "basic" test.
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,700,499))
        aal = ApproachAndLanding()
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio For Flight Phases',alt))
            
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt),  
                     [Section('Fast',slice(0,len(alt),None))])
        
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        goa.derive(Parameter('Altitude AAL For Flight Phases',alt),
                   Parameter('Altitude Radio',alt),
                   aal, climb)
        expected = []
        self.assertEqual(goa, expected)

    def test_go_around_no_rad_alt(self):
        # This tests that the go-around works without a radio altimeter.
        alt = np.ma.array(range(0,4000,500)+range(4000,0,-500)+range(0,1000,501))
        aal = ApproachAndLanding()

        # Call derive method. Note: "None" required to replace rad alt argument.
        aal.derive(Parameter('Altitude AAL For Flight Phases',alt),None) 
        climb = ClimbForFlightPhases()
        climb.derive(Parameter('Altitude STD', alt), aal)
        goa = GoAround()
        # Pretend we are flying over flat ground, so the altitudes are equal.
        alt_aal=Parameter('Altitude AAL For Flight Phases',alt)
        
        # !!! None is positional argument in place of alt_rad !!!
        goa.derive(alt_aal, None, aal, climb)
        
        expected = [KeyTimeInstance(index=16, name='Go Around')]
        self.assertEqual(goa, expected)


class TestAltitudeWhenClimbing(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeWhenClimbing.get_operational_combinations(),
                         [('Climbing', 'Altitude AAL')])
    
    def test_derive(self):
        climbing = S('Climbing', items=[Section('a', slice(4, 10)),
                                        Section('b', slice(12, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(0, 200, 20) + \
                                       range(0, 200, 20),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        altitude_when_climbing = AltitudeWhenClimbing()
        altitude_when_climbing.derive(climbing, alt_aal)
        self.assertEqual(altitude_when_climbing,
          [KeyTimeInstance(index=4.0, name='75 Ft Climbing', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=12.0, name='35 Ft Climbing', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=12.75, name='50 Ft Climbing', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=14.0, name='75 Ft Climbing', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=15.25, name='100 Ft Climbing', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=17.75, name='150 Ft Climbing', datetime=None,
                           latitude=None, longitude=None)]
        )


##class AltitudeInApproach(KeyTimeInstanceNode):
    ##'''
    ##Creates KTIs at certain altitudes when the aircraft is in the approach phase.
    ##'''
    ##NAME_FORMAT = '%(altitude)d Ft In Approach'
    ##ALTITUDES = [1000, 1500, 2000, 3000]
    ##NAME_VALUES = {'altitude': ALTITUDES}
    
    ##def derive(self, approaches=S('Approach And Landing'),
               ##alt_aal=P('Altitude AAL')):
        ##alt_array = hysteresis(alt_aal.array, 10)
        ##for approach in approaches:
            ##for alt_threshold in self.ALTITUDES:
                ### Will trigger a single KTI per height (if threshold is crossed)
                ### per climbing phase.
                ##index = index_at_value(alt_array, approach.slice, alt_threshold)
                ##if index:
                    ##self.create_kti(index, altitude=alt_threshold)



class TestAltitudeInApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInApproach.get_operational_combinations(),
                         [('Approach And Landing', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach And Landing', items=[Section('a', slice(4, 7)),
                                                      Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(1950, 0, -200) + \
                                       range(1950, 0, -200)))
        altitude_in_approach = AltitudeInApproach()
        altitude_in_approach.derive(approaches, alt_aal)
        self.assertEqual(altitude_in_approach,
          [KeyTimeInstance(index=4.7750000000000004, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.775, name='1000 Ft In Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=12.275, name='1500 Ft In Approach',
                           datetime=None, latitude=None, longitude=None)])



class TestAltitudeInFinalApproach(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeInFinalApproach.get_operational_combinations(),
                         [('Approach And Landing', 'Altitude AAL')])
    
    def test_derive(self):
        approaches = S('Approach And Landing', items=[Section('a', slice(2, 7)),
                                                      Section('b', slice(10, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(950, 0, -100) + \
                                       range(950, 0, -100)))
        altitude_in_approach = AltitudeInFinalApproach()
        altitude_in_approach.derive(approaches, alt_aal)
        
        self.assertEqual(altitude_in_approach,
          [KeyTimeInstance(index=4.5499999999999998,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=18.550000000000001,
                           name='100 Ft In Final Approach',
                           datetime=None, latitude=None, longitude=None),
           KeyTimeInstance(index=14.550000000000001,
                           name='500 Ft In Final Approach', datetime=None,
                           latitude=None, longitude=None)])


class TestAltitudeWhenDescending(unittest.TestCase):
    def test_can_operate(self):
        self.assertEqual(AltitudeWhenDescending.get_operational_combinations(),
                         [('Descending', 'Altitude AAL')])
    
    def test_derive(self):
        descending = S('Descending', items=[Section('a', slice(0, 10)),
                                            Section('b', slice(12, 20))])
        alt_aal = P('Altitude AAL',
                    np.ma.masked_array(range(100, 0, -10) + \
                                       range(100, 0, -10),
                                       mask=[False] * 6 + [True] * 3 + \
                                            [False] * 11))
        altitude_when_descending = AltitudeWhenDescending()
        altitude_when_descending.derive(descending, alt_aal)
        self.assertEqual(altitude_when_descending,
          [KeyTimeInstance(index=3.0, name='75 Ft Descending', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=18.5, name='20 Ft Descending', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=17.0, name='35 Ft Descending', datetime=None,
                           latitude=None, longitude=None),
           KeyTimeInstance(index=13.0, name='75 Ft Descending', datetime=None,
                           latitude=None, longitude=None)])




class TestInitialClimbStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = InitialClimbStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_climb_start_basic(self):
        instance = InitialClimbStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Takeoff',slice(0,3.5,None))])
        expected = [KeyTimeInstance(index=3.5, name='Initial Climb Start')]
        self.assertEqual(instance, expected)


class TestLandingPeakDeceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing','Heading Continuous', 
                     'Acceleration Forwards For Flight Phases')]
        opts = LandingPeakDeceleration.get_operational_combinations()
        self.assertEqual(opts, expected) 
        
    def test_landing_peak_deceleration_basic(self):
        head = P('Heading Continuous',np.ma.array([0,2,4,7,9,8,6,3]))
        acc = P('Acceleration Forwards For Flight Phases',
                np.ma.array([0,0,-.1,-.1,-.2,-.1,0,0]))
        landing = [Section('Landing',slice(2,5,None))]
        kti = LandingPeakDeceleration()
        kti.derive(landing, head, acc)
        expected = [KeyTimeInstance(index=4, name='Landing Peak Deceleration')]
        self.assertEqual(kti, expected)


class TestLiftoff(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Rate Of Climb')]
        opts = Liftoff.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_liftoff_basic(self):
        # Linearly increasing climb rate with the 5 fpm threshold set between 
        # the 5th and 6th sample points.
        rate_of_climb = Parameter('Rate Of Climb', np.ma.arange(10)-0.5)
        # Takeoff section encloses the test point.
        takeoff = [Section('Takeoff',slice(0,9,None))]
        lift = Liftoff()
        lift.derive(takeoff, rate_of_climb)
        expected = [KeyTimeInstance(index=5.5, name='Liftoff')]
        self.assertEqual(lift, expected)
    

class TestTakeoffTurnOntoRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff',)]
        opts = TakeoffTurnOntoRunway.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_turn_onto_runway_basic(self):
        instance = TakeoffTurnOntoRunway()
        # This just needs the takeoff slice startpoint, so trivial to test
        takeoff = [Section('Takeoff',slice(1.7,3.5,None))]
        instance.derive(takeoff)
        expected = [KeyTimeInstance(index=1.7, name='Takeoff Turn Onto Runway')]
        self.assertEqual(instance, expected)


class TestTakeoffStartAcceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Takeoff','Acceleration Longitudinal')]
        opts = TakeoffStartAcceleration.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_start_acceleration(self):
        instance = TakeoffStartAcceleration()
        # This just needs the takeoff slice startpoint, so trivial to test
        takeoff = [Section('Takeoff',slice(1,5,None))]
        accel = P('AccelerationLongitudinal',np.ma.array([0,0,0,0.2,0.3,0.3,0.3]))
        instance.derive(takeoff,accel)
        expected = [KeyTimeInstance(index=2.5, name='Takeoff Start Acceleration')]
        self.assertEqual(instance, expected)

    
class TestTopOfClimb(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD','Climb Cruise Descent')]
        opts = TopOfClimb.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_climb_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)

    def test_top_of_climb_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)

    def test_top_of_climb_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfClimb()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=8, name='Top Of Climb')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)


class TestTopOfDescent(unittest.TestCase):
    # Based closely on the level flight condition, but taking only the
    # outside edges of the envelope.
    def test_can_operate(self):
        expected = [('Altitude STD','Climb Cruise Descent')]
        opts = TopOfDescent.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_top_of_descent_basic(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=13, name='Top Of Descent')]
        self.assertEqual(phase, expected)

    def test_top_of_descent_truncated_start(self):
        alt_data = np.ma.array([400]*5+range(400,0,-50))
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = [KeyTimeInstance(index=5, name='Top Of Descent')]
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),1)

    def test_top_of_descent_truncated_end(self):
        alt_data = np.ma.array(range(0,400,50)+[400]*5)
        alt = Parameter('Altitude STD', np.ma.array(alt_data))
        phase = TopOfDescent()
        in_air = ClimbCruiseDescent()
        in_air.append(Section(name='Climb Cruise Descent',
                              slice=slice(0,len(alt.array))))
        phase.derive(alt, in_air)
        expected = []
        self.assertEqual(phase, expected)
        self.assertEqual(len(phase),0)
    
        
class TestTouchdown(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing','Rate Of Climb')]
        opts = Touchdown.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_touchdown_basic(self):
        rate_of_climb = Parameter('Rate Of Climb', np.ma.array([-30,-20,-11,-1,0,0,0]))
        land = [Section('Landing',slice(1,5))]                        
        tdown = Touchdown()
        tdown.derive(land, rate_of_climb)
        # and the real answer is this KTI
        expected = [KeyTimeInstance(index=2.1, name='Touchdown')]
        self.assertEqual(tdown, expected)
    
    
class TestLandingStart(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingStart.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_start_basic(self):
        instance = LandingStart()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None))])
        expected = [KeyTimeInstance(index=66, name='Landing Start')]
        self.assertEqual(instance, expected)


class TestLandingTurnOffRunway(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing',)]
        opts = LandingTurnOffRunway.get_operational_combinations()
        self.assertEqual(opts, expected)        

    def test_initial_landing_turn_off_runway_basic(self):
        instance = LandingTurnOffRunway()
        # This just needs the takeoff slice endpoint, so trivial to test
        instance.derive([Section('Landing',slice(66,77,None))])
        expected = [KeyTimeInstance(index=77, name='Landing Turn Off Runway')]
        self.assertEqual(instance, expected)


class TestLandingStartDeceleration(unittest.TestCase):
    def test_can_operate(self):
        expected = [('Landing','Acceleration Longitudinal')]
        opts = LandingStartDeceleration.get_operational_combinations()
        self.assertEqual(opts, expected)

    def test_takeoff_start_acceleration(self):
        takeoff = [Section('Landing',slice(2,6,None))]
        accel = P('AccelerationLongitudinal',np.ma.array([0,0,0,0,-0.1,-0.3,-0.3]))
        kpv = LandingStartDeceleration()
        kpv.derive(takeoff,accel)
        expected = [KeyTimeInstance(index=4.5, name='Landing Start Deceleration')]
        self.assertEqual(kpv, expected)
