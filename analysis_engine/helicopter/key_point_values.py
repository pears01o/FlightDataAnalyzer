import numpy as np

from analysis_engine.node import (
    KeyPointValueNode, KPV, KTI, P, S, A, M, App, Section,
    helicopter, helicopter_only
)

from flightdatautilities import units as ut

from analysis_engine.settings import (ACCEL_LAT_OFFSET_LIMIT,
                                      ACCEL_LON_OFFSET_LIMIT,
                                      ACCEL_NORM_OFFSET_LIMIT,
                                      AIRSPEED_THRESHOLD,
                                      BUMP_HALF_WIDTH,
                                      CLIMB_OR_DESCENT_MIN_DURATION,
                                      CONTROL_FORCE_THRESHOLD,
                                      GRAVITY_IMPERIAL,
                                      GRAVITY_METRIC,
                                      HOVER_MIN_DURATION,
                                      HYSTERESIS_FPALT,
                                      MIN_HEADING_CHANGE,
                                      NAME_VALUES_CONF,
                                      NAME_VALUES_ENGINE,
                                      NAME_VALUES_LEVER,
                                      NAME_VALUES_RANGES,
                                      HEADING_RATE_FOR_TAXI_TURNS,
                                      REVERSE_THRUST_EFFECTIVE_EPR,
                                      REVERSE_THRUST_EFFECTIVE_N1,
                                      SPOILER_DEPLOYED,
                                      TCAS_SCAN_TIME,
                                      TCAS_THRESHOLD,
                                      VERTICAL_SPEED_FOR_LEVEL_FLIGHT)

from analysis_engine.library import (ambiguous_runway,
                                     align,
                                     all_deps,
                                     all_of,
                                     any_of,
                                     bearings_and_distances,
                                     bump,
                                     closest_unmasked_value,
                                     clump_multistate,
                                     coreg,
                                     cycle_counter,
                                     cycle_finder,
                                     cycle_select,
                                     find_edges,
                                     find_edges_on_state_change,
                                     first_valid_parameter,
                                     first_valid_sample,
                                     hysteresis,
                                     ils_established,
                                     index_at_value,
                                     index_of_first_start,
                                     index_of_last_stop,
                                     integrate,
                                     is_index_within_slice,
                                     is_index_within_slices,
                                     lookup_table,
                                     nearest_neighbour_mask_repair,
                                     mask_inside_slices,
                                     mask_outside_slices,
                                     max_abs_value,
                                     max_continuous_unmasked,
                                     max_maintained_value,
                                     max_value,
                                     median_value,
                                     min_value,
                                     most_common_value,
                                     moving_average,
                                     repair_mask,
                                     np_ma_masked_zeros_like,
                                     np_ma_zeros_like,
                                     peak_curvature,
                                     prev_unmasked_value,
                                     rate_of_change_array,
                                     runs_of_ones,
                                     runway_deviation,
                                     runway_distance_from_end,
                                     runway_heading,
                                     second_window,
                                     shift_slice,
                                     shift_slices,
                                     slice_duration,
                                     slice_midpoint,
                                     slice_samples,
                                     slices_above,
                                     slices_and_not,
                                     slices_below,
                                     slices_between,
                                     slices_duration,
                                     slices_from_ktis,
                                     slices_int,
                                     slices_from_to,
                                     slices_not,
                                     slices_overlap,
                                     slices_or,
                                     slices_and,
                                     slices_remove_overlaps,
                                     slices_remove_small_slices,
                                     slices_remove_small_gaps,
                                     trim_slices,
                                     level_off_index,
                                     valid_slices_within_array,
                                     value_at_index,
                                     vstack_params_where_state,
                                     vstack_params)

class Airspeed500To100FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach between 500ft AAL and 100ft AAL.
    Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed500To100FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach between 500ft AAL and 100ft AAL.
    Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed100To20FtMax(KeyPointValueNode):
    '''
    Maximum airspeed during the final approach and landing between 100 AAL and 20ft AAL.
    Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed100To20FtMin(KeyPointValueNode):
    '''
    Minimum airspeed during the final approach and landing between 100 AAL and 20ft AAL.
    Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_descent_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            air_spd.array,
            alt_descent_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=air_spd.frequency)


class Airspeed20FtToTouchdownMax(KeyPointValueNode):
    '''
    Maximum airspeed between 20ft AGL and touchdown.
    Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Airspeed'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            max_value,
        )


class Airspeed2NMToOffshoreTouchdown(KeyPointValueNode):
    '''
    Airspeed 2NM from offshore touchdown. Helicopter only.
    '''

    units = ut.KT

    name = 'Airspeed 2 NM To Touchdown'

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), dtts=P('Distance To Touchdown'),
               touchdown=KTI('Offshore Touchdown')):

        for tdwn in touchdown:
            dist_to_touchdown = dtts.get_previous(tdwn.index, name='2.0 NM To Touchdown')
            if dist_to_touchdown:
                self.create_kpvs_at_ktis(airspeed.array, [dist_to_touchdown])


class AirspeedAbove101PercentRotorSpeed(KeyPointValueNode):
    '''
    Airspeed At or Above 101% Rotor Speed. Helicopter only.
    '''
    name = 'Airspeed Above 101 Percent Rotor Speed'
    can_operate = helicopter_only

    def derive(self,
               airspeed=P('Airspeed'),
               airborne=S('Airborne'),
               nr=P('Nr'),
               ):
        nr_above_101 = nr.array >= 101.0
        airborne_slices = airborne.get_slices()
        slices_above_101_pct = runs_of_ones(nr_above_101)
        above_101_while_airborne = slices_and(airborne_slices, slices_above_101_pct)
        self.create_kpvs_within_slices(airspeed.array, above_101_while_airborne, max_value)


class AirspeedAbove500FtMin(KeyPointValueNode):
    '''
    Minimum airspeed above 500ft AGL during standard approaches. Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd= P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information')):

        app_slices = []
        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                app_slices.append(approach.slice)

        for alt_agl_slice in alt_agl.slices_above(500):
            if any([slices_overlap(app_slice, alt_agl_slice) for app_slice in app_slices]):
                continue
            else:
                self.create_kpv_from_slices(air_spd.array,[alt_agl_slice], min_value)


class AirspeedAbove500FtMinOffshoreSpecialProcedure(KeyPointValueNode):
    '''
    Minimum airspeed above 500ft AGL during a Shuttling Approach, ARDA
    or AROA. Helicopter only.
    '''

    units = ut.KT
    can_operate = helicopter_only
    def derive(self, air_spd= P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information')):

        app_slices = []
        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                app_slices.append(approach.slice)

        for alt_agl_slice in alt_agl.slices_above(500):
            for app_slice in app_slices:
                if slices_overlap(app_slice, alt_agl_slice):
                    self.create_kpv_from_slices(air_spd.array,[alt_agl_slice], min_value)


class AirspeedAt200FtDuringOnshoreApproach(KeyPointValueNode):
    '''
    Approach airspeed at 200ft AGL. Helicopter only.
    '''

    units = ut.KT
    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               approaches=App('Approach Information'), offshore=M('Offshore')):
        for approach in approaches:
            # check if landed/lowest point of approach is Offshore. May trigger incorrectly
            # close to coast. Will be able to improve once we have implemented Rig locations
            if value_at_index(offshore.array, approach.slice.stop, interpolate=False) == 'Offshore':
                continue

            index = index_at_value(alt_agl.array, 200,
                                   _slice=slice(approach.slice.stop, approach.slice.start, -1))
            if not index:
                continue
            value = value_at_index(air_spd.array, index)
            self.create_kpv(index, value)


class AirspeedAtAPGoAroundEngaged(KeyPointValueNode):
    '''
    Airspeed at which AP Go Around mode has been engaged. Helicopter only.
    '''

    name = 'Airspeed At AP Go Around Engaged'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Pitch Mode (1)')):

        sections = slices_and(airs.get_slices(),
                              clump_multistate(ap_mode.array, 'Go Around'))
        for section in sections:
            index = section.start
            value = air_spd.array[index]
            self.create_kpv(index, value)


class AirspeedWhileAPHeadingEngagedMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed at which AP Heading mode was engaged. Helicopter only.
    '''

    name = 'Airspeed While AP Heading Engaged Min'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Roll-Yaw Mode (1)')):

        heads = clump_multistate(ap_mode.array, 'Heading')
        if heads:
            sections = slices_and(airs.get_slices(), heads)
            self.create_kpv_from_slices(air_spd.array, sections, min_value)


class AirspeedWhileAPVerticalSpeedEngagedMin(KeyPointValueNode):
    '''
    Minimum recorded airspeed at which AP VS mode was engaged. Helicopter only.
    '''

    name = 'Airspeed While AP Vertical Speed Engaged Min'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, air_spd=P('Airspeed'), airs=S('Airborne'),
               ap_mode=M('AP Collective Mode (1)')):

        vss = clump_multistate(ap_mode.array, 'V/S')
        if vss:
            sections = slices_and(airs.get_slices(), vss)
            self.create_kpv_from_slices(air_spd.array, sections, min_value)


##############################################################################
# Airspeed Autorotation


class AirspeedDuringAutorotationMax(KeyPointValueNode):
    '''
    Maximum airspeed during autorotation. Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), phase=S('Autorotation')):
        self.create_kpvs_within_slices(airspeed.array, phase, max_value)


class AirspeedDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum airspeed during autorotation. Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, airspeed=P('Airspeed'), phase=S('Autorotation')):
        self.create_kpvs_within_slices(airspeed.array, phase, min_value)


########################################
# Altitude


class AltitudeDensityMax(KeyPointValueNode):
    '''
    Maximum altitude density. Helicopter only.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, alt_density=P('Altitude Density'), airborne=S('Airborne')):
        self.create_kpv_from_slices(
            alt_density.array,
            airborne.get_slices(),
            max_value
        )


class AltitudeRadioDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum altitude (radio) during autorotation. Helicopter only.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, alt_rad=P('Altitude Radio'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(alt_rad.array, autorotation, min_value)


class AltitudeDuringCruiseMin(KeyPointValueNode):
    '''
    Minimum altitude (AGL) recorded during cruise. Helicopter only.
    '''

    units = ut.FT
    can_operate = helicopter_only

    def derive(self, alt_agl=P('Altitude AGL'), cruise=S('Cruise')):
        self.create_kpvs_within_slices(alt_agl.array, cruise, min_value)


class AltitudeRadioMinBeforeNoseDownAttitudeAdoptionOffshore(KeyPointValueNode):
    '''
    ABO liftoff procedure demands that H175 helicopters lift into a hover
    (10-20ft RadAlt), turn into wind if necessary, descend to <= 3ft RadAlt
    and then apply takeoff power (later steps are disregarded).

    This KPV measures the minimum RadAlt during the descent to <= 3ft RadAlt.
    Helicopter only.
    '''
    units = ut.FT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        return ac_type == helicopter and family and family.value == 'H175' \
           and all_of(('Altitude Radio', 'Offshore', 'Liftoff', 'Hover',
                    'Nose Down Attitude Adoption',
                    'Altitude AAL For Flight Phases'), available)

    def derive(self, offshores=M('Offshore'), liftoffs=KTI('Liftoff'),
               hovers=S('Hover'), nose_downs=S('Nose Down Attitude Adoption'),
               rad_alt=P('Altitude Radio'),
               alt_aal=P('Altitude AAL For Flight Phases')):


        clumped_offshores = clump_multistate(offshores.array, 'Offshore')
        masked_alt_aal = mask_outside_slices(alt_aal.array, clumped_offshores +
                                             hovers.get_slices())

        for clump in clumped_offshores:

            liftoffs_in_clump = [l.index for l in liftoffs if \
                                 is_index_within_slice(l.index, clump)]

            nose_downs_in_clump = [n.slice.start for n in nose_downs if \
                                   is_index_within_slice(n.slice.start, clump)]

            if not liftoffs_in_clump or not nose_downs_in_clump:
                continue

            rad_alt_slices = []

            for idx, liftoff in enumerate(liftoffs_in_clump):
                try:
                    rad_alt_slices.append(slice(liftoffs_in_clump[idx],
                                                nose_downs_in_clump[idx]))
                except IndexError:
                    continue

            for _slice in rad_alt_slices:
                # Diffs of reversed altitude - as we are checking backwards
                # from the start of the nose down phase in order to find the
                # first positive diff
                diffs = np.ma.ediff1d(masked_alt_aal[_slice][::-1])

                try:
                    min_rad_alt_idx = _slice.stop - \
                                      np.ma.where(diffs > 0)[0][0] - 1
                except IndexError:
                    # Fallback to minimum of entire slice instead of local
                    # minimum in case of no positive diff values
                    min_rad_alt_idx = np.ma.argmin(
                                      mask_outside_slices(alt_aal.array,
                                                          [_slice]))

                self.create_kpv(min_rad_alt_idx,
                                rad_alt.array[min_rad_alt_idx])


class AltitudeRadioAtNoseDownAttitudeInitiation(KeyPointValueNode):
    '''
    Radio altitude at nose down attitude initiation.
    '''
    units = ut.FT
    align_frequency = 16

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        return ac_type == helicopter and family and family.value == 'H175'\
               and all_of(('Altitude Radio', 'Nose Down Attitude Adoption'),
                   available)

    def derive(self, rad_alt=P('Altitude Radio'),
               nose_downs=S('Nose Down Attitude Adoption')):

        for nose_down in nose_downs:
            self.create_kpv(nose_down.slice.start,
                            rad_alt.array[nose_down.slice.start])


##############################################################################
# Collective


class CollectiveFrom10To60PercentDuration(KeyPointValueNode):
    '''
    Collective from 10% to 60% duration. Helicopter only.
    '''

    can_operate = helicopter_only

    name = 'Collective From 10 To 60% Duration'
    units = ut.SECOND

    def derive(self, collective=P('Collective'), rtr=S('Rotors Turning')):
        start = 10
        end = 60
        target_ranges = np.ma.clump_unmasked(np.ma.masked_outside(collective.array, start - 1, end + 1))
        valid_sections = []
        for section in target_ranges:
            if (np.ma.ptp(collective.array[max(section.start-1, 0): section.stop+1]) > end - start) and \
               (collective.array[section.start] < collective.array[section.stop]) and \
               ((section.stop - section.start) < collective.frequency*10.0):
                valid_sections.append(section)
        self.create_kpvs_from_slice_durations(slices_and(valid_sections, rtr.get_slices()),
                                              collective.frequency)


##############################################################################
# Tail Rotor


class TailRotorPedalWhileTaxiingABSMax(KeyPointValueNode):
    '''
    Maximum Absolute tail rotor pedal during ground taxi. Helicopter only.
    '''

    name = 'Tail Rotor Pedal While Taxiing ABS Max'

    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       max_abs_value)


class TailRotorPedalWhileTaxiingMax(KeyPointValueNode):
    '''
    Maximum tail rotor pedal during ground taxi. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       max_value)


class TailRotorPedalWhileTaxiingMin(KeyPointValueNode):
    '''
    Minimum tail rotor pedal during ground taxi. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.PERCENT

    def derive(self, pedal=P('Tail Rotor Pedal'), taxiing=S('Taxiing')):
        self.create_kpvs_within_slices(pedal.array, taxiing.get_slices(),
                                       min_value)


##############################################################################
# Cyclic


class CyclicDuringTaxiMax(KeyPointValueNode):
    '''
    Maximum cyclic angle during taxi. Helicopter only.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Angle'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_value)


class CyclicLateralDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum lateral displacement of the cyclic from the neutral
    point during ground taxi phase. Helicopter only.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Lateral'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_abs_value)


class CyclicAftDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum rearward displacement of the cyclic from the neutral
    point during ground taxi phase. Helicopter only.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Fore-Aft'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        np.ma.masked_greater_equal(cyclic.array, 0)
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       max_value)


class CyclicForeDuringTaxiMax(KeyPointValueNode):
    '''
    Measures the maximum fore ward displacement of the cyclic from the neutral
    point during ground taxi phase. Helicopter only.
    '''

    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, cyclic=P('Cyclic Fore-Aft'), taxi=S('Taxiing'), rtr=S('Rotors Turning')):
        np.ma.masked_less_equal(cyclic.array, 0)
        self.create_kpvs_within_slices(cyclic.array, slices_and(taxi.get_slices(),
                                                                rtr.get_slices()),
                                       min_value)


##############################################################################
# Engine Transients


class EngTorqueExceeding100Percent(KeyPointValueNode):
    '''
    Measures the duration of Eng (*) Torque Avg exceeding 100. Helicopter only.
    '''

    units = ut.SECOND
    can_operate = helicopter_only

    def derive(self, avg_torque = P('Eng (*) Torque Avg')):
        threshold_slices = slices_above(avg_torque.array, 100)[1]
        threshold_slices = slices_remove_small_slices(threshold_slices, 2, avg_torque.hz)
        self.create_kpvs_from_slice_durations(threshold_slices, self.frequency)


class EngTorqueExceeding110Percent(KeyPointValueNode):
    '''
    Measures the duration of Eng (*) Torque Avg exceeding 110. Helicopter only.
    '''

    units = ut.SECOND
    can_operate = helicopter_only

    def derive(self, avg_torque = P('Eng (*) Torque Avg')):
        threshold_slices = slices_above(avg_torque.array, 110)[1]
        threshold_slices = slices_remove_small_slices(threshold_slices, 2, avg_torque.hz)
        self.create_kpvs_from_slice_durations(threshold_slices, self.frequency)


##############################################################################
# Engine N2


class EngN2DuringMaximumContinuousPowerMin(KeyPointValueNode):
    '''
    Minimum N2 recorded during maximum continuous power.

    Maximum continuous power applies whenever takeoff or go-around
    power settings are not in force. Helicopter only.
    '''

    name = 'Eng N2 During Maximum Continuous Power Min'
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self,
               eng_n2_min=P('Eng (*) N2 Min'),
               mcp=S('Maximum Continuous Power')):

        self.create_kpvs_within_slices(eng_n2_min.array, mcp, min_value)


##############################################################################
# Engine Torque


class EngTorqueWithOneEngineInoperativeMax(KeyPointValueNode):
    '''
    Maximum engine torque recorded with one engine inoperative.
    Helicoper only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self,
               eng_trq_max=P('Eng (*) Torque Max'),
               airborne=S('Airborne'),
               one_eng=M('One Engine Inoperative')):

        phases = slices_and(runs_of_ones(one_eng.array == 'OEI'), airborne.get_slices())
        self.create_kpvs_within_slices(eng_trq_max.array, phases, max_value)


class EngTorqueAbove90KtsMax(KeyPointValueNode):
    '''
    Maximum engine torque where the indicate airspeed is above 90 kts.

    Some Helicopters have Torque restictions above 90 kts to limit flight
    control loads at high speed thereby preserving dynamic component
    replacement times. Helicopter only.

    '''

    name = 'Eng Torque Above 90 Kts Max'

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, eng=P('Eng (*) Torque Max'), air_spd=P('Airspeed')):
        slices_no_gaps = slices_remove_small_gaps(air_spd.slices_above(90), time_limit=10, hz=air_spd.hz)

        self.create_kpvs_within_slices(
            eng.array,
            slices_no_gaps,
            max_value
            )


class EngTorqueAbove100KtsMax(KeyPointValueNode):
    '''
    Maximum engine torque where the indicate airspeed is above 100 kts.

    Some Helicopters have Torque restictions above 100 kts to limit flight
    control loads at high speed thereby preserving dynamic component
    replacement times (taken from S92 type certificate). Helicopter only.
    '''

    name = 'Eng Torque Above 100 Kts Max'

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, eng=P('Eng (*) Torque Max'), air_spd=P('Airspeed')):
        self.create_kpvs_within_slices(
            eng.array,
            air_spd.slices_above(100),
            max_value
        )


##############################################################################
# Gearbox Oil


class MGBOilTempMax(KeyPointValueNode):
    '''
    Find the Max temperature for the main gearbox oil. Helicopter only.
    '''
    units = ut.CELSIUS
    name = 'MGB Oil Temp Max'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Temp', 'MGB (Fwd) Oil Temp',
                          'MGB (Aft) Oil Temp'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Temp'), mgb_fwd=P('MGB (Fwd) Oil Temp'),
               mgb_aft=P('MGB (Aft) Oil Temp'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.max(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, max_value)


class MGBOilPressMax(KeyPointValueNode):
    '''
    Find the Maximum main gearbox oil pressure. Helicopter only.
    '''
    units = ut.PSI
    name = 'MGB Oil Press Max'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press', 'MGB (Fwd) Oil Press',
                          'MGB (Aft) Oil Press'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Press'), mgb_fwd=P('MGB (Fwd) Oil Press'),
               mgb_aft=P('MGB (Aft) Oil Press'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.max(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, max_value)


class MGBOilPressMin(KeyPointValueNode):
    '''
    Find the Minimum main gearbox oil pressure. Helicopter only.
    '''
    units = ut.PSI
    name = 'MGB Oil Press Min'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press', 'MGB (Fwd) Oil Press',
                          'MGB (Aft) Oil Press'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=P('MGB Oil Press'), mgb_fwd=P('MGB (Fwd) Oil Press'),
               mgb_aft=P('MGB (Aft) Oil Press'), airborne=S('Airborne')):
        gearboxes = vstack_params(mgb, mgb_fwd, mgb_aft)
        gearbox = np.ma.min(gearboxes, axis=0)
        self.create_kpvs_within_slices(gearbox, airborne, min_value)


class MGBOilPressLowDuration(KeyPointValueNode):
    '''
    Duration of the gearbox oil pressure low warning. Helicopter only.
    '''
    units = ut.SECOND
    name = 'MGB Oil Press Low Duration'

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        aircraft = ac_type == helicopter
        gearbox = any_of(('MGB Oil Press Low', 'MGB Oil Press Low (1)',
                          'MGB Oil Press Low (2)'), available)
        airborne = 'Airborne' in available
        return aircraft and gearbox and airborne

    def derive(self, mgb=M('MGB Oil Press Low'),
               mgb1=M('MGB Oil Press Low (1)'),
               mgb2=M('MGB Oil Press Low (2)'),
               airborne=S('Airborne')):
        hz = (mgb or mgb1 or mgb2).hz
        gearbox = vstack_params_where_state((mgb, 'Low Press'),
                                            (mgb1, 'Low Press'),
                                            (mgb2, 'Low Press'))
        self.create_kpvs_where(gearbox.any(axis=0), hz, phase=airborne)


class CGBOilTempMax(KeyPointValueNode):
    '''
    Find the Max temperature for the combining gearbox oil. Helicopter only.
    '''
    units = ut.CELSIUS
    name = 'CGB Oil Temp Max'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Temp'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, max_value)


class CGBOilPressMax(KeyPointValueNode):
    '''
    Find the Maximum combining gearbox oil pressure. Helicopter only.
    '''
    units = ut.PSI
    name = 'CGB Oil Press Max'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Press'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, max_value)


class CGBOilPressMin(KeyPointValueNode):
    '''
    Find the Minimum combining gearbox oil pressure. Helicopter only.
    '''
    units = ut.PSI
    name = 'CGB Oil Press Min'
    can_operate = helicopter_only

    def derive(self, cgb=P('CGB Oil Press'), airborne=S('Airborne')):
        self.create_kpvs_within_slices(cgb.array, airborne, min_value)


##############################################################################
# Heading


class HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxStandardApproach(KeyPointValueNode):
    '''
    Maximum heading variation (PTP) 1.5 to 1.0 NM from touchdown.
    Helicopter only.
    '''

    units = ut.DEGREE

    name = 'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Standard Approach'

    can_operate = helicopter_only

    def derive(self, heading=P('Heading Continuous'),
               dtts=KTI('Distance To Touchdown'),
               offshore_twn=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):

        for approach in approaches:
            if approach.type in ['APPROACH', 'LANDING', 'GO_AROUND', 'TOUCH_AND_GO', 'RIG']:
                for tdwn in offshore_twn:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        start_kti = dtts.get_previous(tdwn.index, name='1.5 NM To Touchdown')
                        stop_kti = dtts.get_previous(tdwn.index, name='1.0 NM To Touchdown')
                        if start_kti and stop_kti:
                            phase = slice(start_kti.index, stop_kti.index+1)
                            heading_delta = np.ma.ptp(heading.array[phase])
                            self.create_kpv(phase.stop-1, heading_delta)


class HeadingVariation1_5NMTo1_0NMFromOffshoreTouchdownMaxSpecialProcedure(KeyPointValueNode):
    '''
    Maximum heading variation (PTP) 1.5 to 1.0 NM from touchdown.
    Helicopter only.
    '''

    units = ut.DEGREE

    name = 'Heading Variation 1.5 NM To 1.0 NM From Offshore Touchdown Max Special Procedure'

    can_operate = helicopter_only

    def derive(self, heading=P('Heading Continuous'),
               dtts=KTI('Distance To Touchdown'),
               offshore_twn=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):

        for approach in approaches:
            if approach.type in ['SHUTTLING', 'AIRBORNE_RADAR']:
                for tdwn in offshore_twn:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        start_kti = dtts.get_previous(tdwn.index, name='1.5 NM To Touchdown')
                        stop_kti = dtts.get_previous(tdwn.index, name='1.0 NM To Touchdown')
                        if start_kti and stop_kti:
                            phase = slice(start_kti.index, stop_kti.index+1)
                            heading_delta = np.ma.ptp(heading.array[phase])
                            self.create_kpv(phase.stop-1, heading_delta)


class TrackVariation100To50Ft(KeyPointValueNode):
    '''
    Checking the variation in track angle during the latter stages of the descent.
    Uses Altitude AGL. Helicopter only.
    '''

    name = 'Track Variation 100 To 50 Ft'
    units = ut.DEGREE_S

    can_operate = helicopter_only

    def derive(self, track=P('Track'),
               alt_agl=P('Altitude AGL')):

        # The threshold applied here ensures that the altitude passes through this range and does not
        # just dip into the range, as might happen for a light aircraft or helicopter flying at 100ft.
        for band in alt_agl.slices_from_to(100, 50, threshold=1.0):
            dev = np.ma.ptp(track.array[band])
            self.create_kpv(band.stop, dev)


class HeadingDuringLanding(KeyPointValueNode):
    '''
    We take the median heading during the landing roll as this avoids problems
    with drift just before touchdown and heading changes when turning off the
    runway. The value is "assigned" to a time midway through the landing phase.

    This KPV is a helicopter variant to accommodate helicopter transitions,
    so that the landing runway can be identified where the aircraft is
    operating at a conventional airport. Helicopter only.
    '''

    units = ut.DEGREE

    def derive(self,
               hdg=P('Heading Continuous'),
               land_helos=S('Transition Flight To Hover')):
        for land_helo in land_helos:
            index = land_helo.slice.start
            self.create_kpv(index, float(round(hdg.array[index], 8)) % 360.0)

##############################################################################
# Groundspeed


class Groundspeed20FtToTouchdownMax(KeyPointValueNode):
    '''
    The highest groundspeed during the flare manoeuvre. Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self,
               air_spd=P('Groundspeed'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            air_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            max_value,
        )


class Groundspeed20SecToOffshoreTouchdownMax(KeyPointValueNode):
    '''
    Find the maximum groundspeed 20 seconds from the point of an offshore touchdown.
    Helicopter only.
    '''
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, groundspeed=P('Groundspeed'),
               touchdown=KTI('Offshore Touchdown'),
               secs_tdwn=KTI('Secs To Touchdown')):

        #idx_to_tdwn = \
            #[s.index for s in secs_tdwn if s.name == '20 Secs To Touchdown']
        #idx_at_tdwn = [t.index for t in touchdown]

        #if idx_to_tdwn and idx_at_tdwn:
            #_slice = [slice(a, b) for a, b in zip(idx_to_tdwn, idx_at_tdwn)]
            #self.create_kpvs_within_slices(groundspeed.array, _slice,
                                           #max_value)
        for tdwn in touchdown:
            secs_to_touchdown = secs_tdwn.get_previous(tdwn.index)
            if secs_to_touchdown:
                self.create_kpv(*max_value(groundspeed.array,
                                           _slice=slice(secs_to_touchdown.index, tdwn.index+1),
                                           start_edge=secs_to_touchdown.index,
                                           stop_edge=tdwn.index))


class Groundspeed0_8NMToOffshoreTouchdownSpecialProcedure(KeyPointValueNode):
    '''
    Groundspeed at 0.8 NM away from an offshore touchdown during a Shuttling Approach,
    ARDA or AROA. Helicopter only.
    '''

    name = 'Groundspeed 0.8 NM To Offshore Touchdown Special Procedure'

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, groundspeed=P('Groundspeed'),
               dtts=KTI('Distance To Touchdown'), touchdown=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):

        for approach in approaches:
            if approach.type in ['SHUTTLING','AIRBORNE_RADAR']:
                for tdwn in touchdown:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        dist_to_touchdown = dtts.get_previous(tdwn.index, name='0.8 NM To Touchdown')
                        if dist_to_touchdown:
                            self.create_kpvs_at_ktis(groundspeed.array, [dist_to_touchdown])


class Groundspeed0_8NMToOffshoreTouchdownStandardApproach(KeyPointValueNode):
    '''
    Groundspeed at 0.8 NM away from an offshore touchdown during a standard
    approach. Helicopter only.
    '''

    name = 'Groundspeed 0.8 NM To Offshore Touchdown Standard Approach'
    units = ut.KT
    can_operate = helicopter_only

    def derive(self, groundspeed=P('Groundspeed'),
               dtts=KTI('Distance To Touchdown'), touchdown=KTI('Offshore Touchdown'),
               approaches=App('Approach Information')):

        for approach in approaches:
            if approach.type in ['APPROACH', 'LANDING', 'GO_AROUND', 'TOUCH_AND_GO', 'RIG']:
                for tdwn in touchdown:
                    if is_index_within_slice(tdwn.index, slice(approach.slice.start, approach.slice.stop+10)):
                        dist_to_touchdown = dtts.get_previous(tdwn.index, name='0.8 NM To Touchdown')
                        if dist_to_touchdown:
                            self.create_kpvs_at_ktis(groundspeed.array, [dist_to_touchdown])


class GroundspeedBelow15FtFor20SecMax(KeyPointValueNode):
    '''
    Maximum groundspeed below 15ft AAL recorded for at least 20 seconds
    while airborne. Helicopter only.
    '''

    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), alt_aal=P('Altitude AAL For Flight Phases'), airborne=S('Airborne')):
        gspd_20_sec = second_window(gnd_spd.array, self.frequency, 20)
        height_bands = slices_and(airborne.get_slices(),
                                  slices_below(alt_aal.array, 15)[1])
        self.create_kpv_from_slices(gspd_20_sec, height_bands, max_value)


class GroundspeedWhileAirborneWithASEOff(KeyPointValueNode):
    '''
    Measures the maximum groundspeed during flight with stability systems
    disengaged. Helicopter only.
    '''

    name = 'Groundspeed While Airborne With ASE Off'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), ase=M('ASE Engaged'), airborne=S('Airborne')):
        sections = clump_multistate(ase.array, 'Engaged', airborne.get_slices(), False)
        self.create_kpvs_within_slices(gnd_spd.array, sections, max_value)


class GroundspeedWhileHoverTaxiingMax(KeyPointValueNode):
    '''
    Maximum groundspeed reached durng hover taxi. Helicopter only.
    '''

    name = 'Groundspeed While Hover Taxiing Max'
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), hover_taxi=S('Hover Taxi')):
        self.create_kpvs_within_slices(gnd_spd.array, hover_taxi.get_slices(), max_value)


class GroundspeedWithZeroAirspeedFor5SecMax(KeyPointValueNode):
    '''
    Maximum groundspeed recorded for at least 5 seconds with airspeed at 0.
    Helicopter only.
    '''

    align_frequency = 2
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, wind_spd=P('Wind Speed'), wind_dir=P('Wind Direction'),
               gnd_spd=P('Groundspeed'), heading=P('Heading'),
               airborne=S('Airborne')):

        rad_scale = np.radians(1.0)
        headwind = gnd_spd.array + wind_spd.array * np.ma.cos((wind_dir.array-heading.array)*rad_scale)
        if np.ma.count(headwind):
            zero_airspeed = slices_and(airborne.get_slices(),
                                    slices_below(headwind, 0)[1])
            zero_airspeed = slices_remove_small_slices(zero_airspeed, time_limit=5,
                                                      hz=self.frequency)
            self.create_kpvs_within_slices(gnd_spd.array, zero_airspeed, max_value)


class GroundspeedBelow100FtMax(KeyPointValueNode):
    '''
    Maximum groundspeed below 100ft AGL. Helicopter only.
    '''
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, gnd_spd=P('Groundspeed'), alt_agl=P('Altitude AGL For Flight Phases'),
               airborne=S('Airborne')):
        alt_slices = slices_and(airborne.get_slices(),
                                alt_agl.slices_below(100))
        self.create_kpvs_within_slices(gnd_spd.array,
                                       alt_slices,
                                       max_value)

##############################################################################
# Pitch


class PitchBelow1000FtMax(KeyPointValueNode):
    '''
    Maximum Pitch below 1000ft AGL in flight. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AGL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_below(1000), max_value)


class PitchBelow1000FtMin(KeyPointValueNode):
    '''
    Minimum Pitch below 1000ft AGL in flight. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt=P('Altitude AGL')):
        self.create_kpvs_within_slices(pitch.array,
                                       alt.slices_below(1000), min_value)


class PitchBelow5FtMax(KeyPointValueNode):
    '''
    Maximum Pitch below 5ft AGL in flight. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(), alt_agl.slices_below(5))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class Pitch5To10FtMax(KeyPointValueNode):
    '''
    Maximum Pitch ascending 5 to 10ft AGL in flight. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(),
                            alt_agl.slices_from_to(5, 10))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class Pitch10To5FtMax(KeyPointValueNode):
    '''
    Maximum Pitch descending 10 to 5ft AGL in flight. Helicopter only.
    '''
    can_operate = helicopter_only

    units = ut.DEGREE

    def derive(self, pitch=P('Pitch'), alt_agl=P('Altitude AGL'),
               airborne=S('Airborne')):
        slices = slices_and(airborne.get_slices(),
                            alt_agl.slices_from_to(10, 5))
        self.create_kpvs_within_slices(pitch.array, slices, max_value)


class Pitch500To100FtMax(KeyPointValueNode):
    '''
    Maximum pitch 500ft to 100ft AGL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 500)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION,
            freq=pitch.frequency)


class Pitch500To100FtMin(KeyPointValueNode):
    '''
    Minimum pitch 500ft to 100ft AGL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 500)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION,
            freq=pitch.frequency)


class Pitch100To20FtMax(KeyPointValueNode):
    '''
    Maximum pitch 100ft to 20ft AAL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 20, 100)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            max_value,
            min_duration=HOVER_MIN_DURATION, # TODO: check where this came from.
            freq=pitch.frequency)


class Pitch100To20FtMin(KeyPointValueNode):
    '''
    Minimum pitch 100ft to 20ft AGL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descent'),
               ac_type=A('Aircraft Type')):

        alt_band = np.ma.masked_outside(alt_agl.array, 20, 100)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            pitch.array,
            alt_app_sections,
            min_value,
            min_duration=HOVER_MIN_DURATION, # TODO: check where this came from.
            freq=pitch.frequency)


class Pitch50FtToTouchdownMin(KeyPointValueNode):
    '''
    Minimum pitch 50ft AGL to touchdown. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self,
               pitch=P('Pitch'),
               alt_agl=P('Altitude AGL'),
               touchdowns=KTI('Touchdown')):

        self.create_kpvs_within_slices(
            pitch.array,
            alt_agl.slices_to_kti(50, touchdowns),
            min_value,
        )


class PitchOnGroundMax(KeyPointValueNode):
    '''
    Pitch attitude maximum to check for sloping ground operation.
    Helicopter only.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'),
               grounded=S('Grounded'), on_deck=S('On Deck')):
        '''
        The collective parameter ensures this is not the attitude during liftoff.
        '''
        my_slices = slices_and_not(grounded.get_slices(), on_deck.get_slices())
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(my_slices, low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       max_value)


class PitchOnDeckMax(KeyPointValueNode):
    '''
    Pitch attitude maximum during operation on a moving deck. Helicopter only.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       max_value)


class PitchOnGroundMin(KeyPointValueNode):
    '''
    Minimum pitch on the ground (or deck). Helicopter only.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), grounded=S('Grounded'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        my_slices = slices_and_not(grounded.get_slices(), my_slices)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       min_value)


class PitchOnDeckMin(KeyPointValueNode):
    '''
    Minimum pitch while on deck. Helicopter only.
    '''
    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, pitch=P('Pitch'), coll=P('Collective'), on_deck=S('On Deck')):
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(pitch.array,
                                       my_slices,
                                       min_value)


##############################################################################
# Rate of Descent


class RateOfDescent100To20FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    100ft and 20ft AAL. Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent500To100FtMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between
    500ft and 100ft AAL. Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL'),
               descending=S('Descent')):

        alt_band = np.ma.masked_outside(alt_agl.array, 500, 100)
        # maximum RoD must be a big negative value; mask all positives
        alt_band[vrt_spd.array > 0] = np.ma.masked
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_app_sections,
            min_value,
            min_duration=5.0,
            freq=vrt_spd.frequency)


class RateOfDescent20FtToTouchdownMax(KeyPointValueNode):
    '''
    Measures the most negative vertical speed (rate of descent) between 20ft
    and touchdown. Uses altitude AGL. Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               touchdowns=KTI('Touchdown'),
               alt_agl=P('Altitude AGL')):
        '''
        The ground effect compressibility makes the normal pressure altitude
        based vertical speed meaningless, so we use the more complex inertial
        computation to give accurate measurements within ground effect.
        '''
        # maximum RoD must be a big negative value; mask all positives
        vrt_spd.array[vrt_spd.array > 0] = np.ma.masked
        self.create_kpvs_within_slices(
            vrt_spd.array,
            alt_agl.slices_to_kti(20, touchdowns),
            min_value,
        )


class RateOfDescentBelow500FtMax(KeyPointValueNode):
    '''
    Returns the highest single rate of descent for all periods below 500ft AGL.
    Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self,
               vrt_spd=P('Vertical Speed Inertial'),
               alt_agl=P('Altitude AGL For Flight Phases'),
               descending=S('Descending')):
        height_bands = slices_and(descending.get_slices(),
                                  slices_below(alt_agl.array, 500)[1])
        self.create_kpvs_within_slices(vrt_spd.array, height_bands, min_value,
            min_duration=HOVER_MIN_DURATION, freq=vrt_spd.frequency)


class RateOfDescentBelow30KtsWithPowerOnMax(KeyPointValueNode):
    '''
    Rate of descent below 30kts IAS with engine power on max. Helicopter only.
    '''

    units = ut.FPM

    can_operate = helicopter_only

    def derive(self, vrt_spd=P('Vertical Speed Inertial'), air_spd=P('Airspeed'), descending=S('Descending'),
               power=P('Eng (*) Torque Avg')):
        speed_bands = slices_and(descending.get_slices(),
                                  slices_below(air_spd.array, 30)[1])
        speed_bands = slices_and(speed_bands,
                                 slices_above(power.array, 20.0)[1])
        self.create_kpvs_within_slices(vrt_spd.array, speed_bands, min_value)


class VerticalSpeedAtAltitude(KeyPointValueNode):
    '''
    Approach vertical speed at 500 and 300 Ft. Helicopter only.
    '''
    NAME_FORMAT = 'Vertical Speed At %(altitude)d Ft'
    NAME_VALUES = {'altitude': [500, 300]}
    units = ut.FPM
    can_operate = helicopter_only

    def derive(self, vert_spd=P('Vertical Speed'), alt_agl=P('Altitude AGL'),
               approaches=S('Approach')):
        for approach in approaches:
            for altitude in self.NAME_VALUES['altitude']:
                index = index_at_value(alt_agl.array, altitude,
                                       approach.slice, 'nearest')
                if not index:
                    continue
                value = value_at_index(vert_spd.array, index)
                if value:
                    self.create_kpv(index, value, altitude=altitude)


##############################################################################
# Roll


class Roll100To20FtMax(KeyPointValueNode):
    '''
    Maximum recorded roll angle between 100ft and 20ft AAL. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases'), descending=S('Descent')):
        alt_band = np.ma.masked_outside(alt_agl.array, 100, 20)
        alt_app_sections = valid_slices_within_array(alt_band, descending)
        self.create_kpvs_within_slices(
            roll.array,
            alt_app_sections,
            max_abs_value,
            min_duration=HOVER_MIN_DURATION,
            freq=roll.frequency,
        )


class RollAbove300FtMax(KeyPointValueNode):
    '''
    Maximum Roll above 300ft AGL in flight. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        _, height_bands = slices_above(alt_agl.array, 300)
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollBelow300FtMax(KeyPointValueNode):
    '''
    Maximum Roll below 300ft AGL in flight. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases'),
               airborne=S('Airborne')):
        alt_slices = slices_and(airborne.get_slices(),
                                slices_below(alt_agl.array, 300)[1])
        self.create_kpvs_within_slices(roll.array, alt_slices, max_abs_value)


class RollWithAFCSDisengagedMax(KeyPointValueNode):
    '''
    Maximum roll whilst AFCS 1 and AFCS 2 are disengaged. Helicopter only.
    '''
    units = ut.DEGREE
    name = 'Roll With AFCS Disengaged Max'
    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), afcs1=M('AFCS (1) Engaged'),
               afcs2=M('AFCS (2) Engaged')):
        afcs = vstack_params_where_state((afcs1, 'Engaged'),
                                         (afcs2, 'Engaged')).any(axis=0)

        afcs_slices = np.ma.clump_unmasked(np.ma.masked_equal(afcs, 1))
        self.create_kpvs_within_slices(roll.array, afcs_slices, max_abs_value)


class RollAbove500FtMax(KeyPointValueNode):
    '''
    Maximum Roll above 500ft AGL in flight. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        height_bands = slices_above(alt_agl.array, 500)[1]
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollBelow500FtMax(KeyPointValueNode):
    '''
    Maximum Roll below 500ft AGL in flight. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), alt_agl=P('Altitude AGL For Flight Phases')):
        height_bands = slices_below(alt_agl.array, 500)[1]
        self.create_kpvs_within_slices(roll.array, height_bands, max_abs_value)


class RollOnGroundMax(KeyPointValueNode):
    '''
    Roll attitude on firm ground or a solid rig platform. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), coll=P('Collective'), grounded=S('Grounded'), on_deck=S('On Deck')):
        my_slices = slices_and_not(grounded.get_slices(), on_deck.get_slices())
        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(my_slices, low_coll)
        self.create_kpvs_within_slices(roll.array,
                                       my_slices,
                                       max_abs_value)


class RollOnDeckMax(KeyPointValueNode):
    '''
    Roll attitude on moving deck. Helicopter only.
    '''

    units = ut.DEGREE

    can_operate = helicopter_only

    def derive(self, roll=P('Roll'), coll=P('Collective'), on_deck=S('On Deck')):

        _, low_coll = slices_below(coll.array, 25.0)
        my_slices = slices_and(on_deck.get_slices(), low_coll)
        self.create_kpvs_within_slices(roll.array,
                                       my_slices,
                                       max_abs_value)


class RollRateMax(KeyPointValueNode):
    '''
    Maximum recorded roll rate. Helicopter only.
    '''

    units = ut.DEGREE_S

    can_operate = helicopter_only

    def derive(self, rr=P('Roll Rate'), airs=S('Airborne')):

        for air in airs:
            cycles = cycle_finder(rr.array[air.slice], min_step=5.0)
            for index in cycles[0][1:-1]:
                roll_rate = rr.array[index]
                if abs(roll_rate) > 5.0:
                    self.create_kpv(index+air.slice.start, roll_rate)


##############################################################################
# Rotor


class RotorSpeedDuringAutorotationAbove108KtsMin(KeyPointValueNode):
    '''
    Minimum recorded rotor speed during autorotation above 108kts IAS.
    Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), air_spd=P('Airspeed'), autorotation=S('Autorotation')):
        speed_bands = slices_and(autorotation.get_slices(),
                                  slices_above(air_spd.array, 108)[1])
        self.create_kpvs_within_slices(nr.array, speed_bands, min_value)


class RotorSpeedDuringAutorotationBelow108KtsMin(KeyPointValueNode):
    '''
    Minimum recorded rotor speed during autorotation below 108kts IAS.
    Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), air_spd=P('Airspeed'), autorotation=S('Autorotation')):
        speed_bands = slices_and(autorotation.get_slices(),
                                  slices_below(air_spd.array, 108)[1])
        self.create_kpvs_within_slices(nr.array, speed_bands, min_value)


class RotorSpeedDuringAutorotationMax(KeyPointValueNode):
    '''
    Maximum recorded rotor speed during autorotation. Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(nr.array, autorotation.get_slices(), max_value)


class RotorSpeedDuringAutorotationMin(KeyPointValueNode):
    '''
    Minimum rotor speed during autorotion. Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), autorotation=S('Autorotation')):
        self.create_kpvs_within_slices(nr.array, autorotation.get_slices(),
                                       min_value)


class RotorSpeedWhileAirborneMax(KeyPointValueNode):
    '''
    This excludes autorotation, so is maximum rotor speed with power applied.
    Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), airborne=S('Airborne'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(airborne.get_slices(),
                                                  autorotation.get_slices()),
                                    max_value)


class RotorSpeedWhileAirborneMin(KeyPointValueNode):
    '''
    This excludes autorotation, so is minimum rotor speed with power applied.
    Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), airborne=S('Airborne'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(airborne.get_slices(),
                                                   autorotation.get_slices()),
                                    min_value)


class RotorSpeedWithRotorBrakeAppliedMax(KeyPointValueNode):
    '''
    Maximum rotor speed recorded with rotor brake applied. Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), rotor_brake=P('Rotor Brake Engaged')):
        nr_array = np.ma.masked_less(nr.array, 1) # not interested if Rotor is not turning.
        slices = clump_multistate(rotor_brake.array, 'Engaged')
        # Synthetic minimum duration to ensure two samples needed to trigger.
        self.create_kpvs_within_slices(nr_array, slices, max_value, min_duration=1, freq=1)


class RotorsRunningDuration(KeyPointValueNode):
    '''
    Duration for which the rotors were running. Helicopter only.
    '''

    units = ut.SECOND

    can_operate = helicopter_only

    def derive(self, rotors=M('Rotors Running')):
        running = runs_of_ones(rotors.array == 'Running')
        if running:
            value = slices_duration(running, rotors.frequency)
            self.create_kpv(running[-1].stop, value)


class RotorSpeedDuringMaximumContinuousPowerMin(KeyPointValueNode):
    '''
    Minimum rotor speed during maximum continuous power phase.
    Helicopter only.
    '''

    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, nr=P('Nr'), mcp=S('Maximum Continuous Power'), autorotation=S('Autorotation')):
        self.create_kpv_from_slices(nr.array,
                                    slices_and_not(mcp.get_slices(),
                                                   autorotation.get_slices()),
                                    min_value)


class RotorSpeed36To49Duration(KeyPointValueNode):
    '''
    Duration in which rotor speed in running between 36% and 49%.
    Helicopter only.
    '''

    units = ut.SECOND

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)

    def derive(self, nr=P('Nr')):
        self.create_kpvs_from_slice_durations(
            slices_between(nr.array, 36, 49)[1], nr.frequency)


class RotorSpeed56To67Duration(KeyPointValueNode):
    '''
    Duration in which rotor speed in running between 56% and 67%.
    Helicopter only.
    '''

    units = ut.SECOND

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)

    def derive(self, nr=P('Nr')):
        self.create_kpvs_from_slice_durations(
            slices_between(nr.array, 56, 67)[1], nr.frequency)


class RotorSpeedAt6PercentCollectiveDuringEngStart(KeyPointValueNode):
    '''
    During the engines starting the collective the needs to be high, but as the
    rotor speed starts to increase the colective needs to be lowered. This KPV
    records the value Nr when the collective is fully lowered (<6% collective).
    Helicopter only.
    '''

    units = ut.PERCENT

    @classmethod
    # This KPV is specific to the S92 helicopter
    def can_operate(cls, available, ac_type=A('Aircraft Type'),
                    family=A('Family')):
        is_s92 = ac_type == helicopter and family and family.value == 'S92'
        return is_s92 and all_deps(cls, available)

    def derive(self, nr=P('Nr'), collective=P('Collective'),
               firsts=KTI('First Eng Fuel Flow Start')):
        for first in firsts:
            if value_at_index(collective.array, first.index) > 50:
                # ensure we are looking at rotor start and not rotors already
                # running at start of data
                index = index_at_value(collective.array, 6,
                                             slice(first.index, None))
                value = value_at_index(nr.array, index)
                self.create_kpv(index, value)
            else:
                continue

##############################################################################
# Wind


class WindSpeedInCriticalAzimuth(KeyPointValueNode):
    '''
    Maximum relative windspeed when wind blowing into tail rotor.
    The critical direction is helicopter type-specific. Helicopter only.
    '''

    align_frequency = 2
    units = ut.KT

    can_operate = helicopter_only

    def derive(self, wind_spd=P('Wind Speed'), wind_dir=P('Wind Direction'),
               tas=P('Airspeed True'), heading=P('Heading'),
               airborne=S('Airborne')):

        # Puma AS330 critical arc is the port quarter
        min_arc = 180
        max_arc = 270

        rad_scale = np.radians(1.0)
        headwind = tas.array + wind_spd.array * np.ma.cos((wind_dir.array-heading.array)*rad_scale)
        sidewind = wind_spd.array * np.ma.sin((wind_dir.array-heading.array)*rad_scale)

        app_dir = np.arctan2(sidewind, headwind)/rad_scale%360
        critical_dir = np.ma.masked_outside(app_dir, min_arc, max_arc)
        app_speed = np.ma.sqrt(sidewind*sidewind + headwind*headwind)
        critical_speed = np.ma.array(data=app_speed.data, mask=critical_dir.mask)

        self.create_kpvs_within_slices(critical_speed, airborne, max_value)


##############################################################################
# Temperature


class SATMin(KeyPointValueNode):
    '''
    Minimum recorded SAT. Helicopter only.
    '''

    name = 'SAT Min'
    units = ut.CELSIUS

    can_operate = helicopter_only

    def derive(self,
               sat=P('SAT'),
               family=A('Family'),
               rotors_turning=S('Rotors Turning')):

        if family and family.value == 'S92':
            self.create_kpvs_within_slices(sat.array, rotors_turning, min_value)
        else:
            self.create_kpv(*min_value(sat.array))


class SATRateOfChangeMax(KeyPointValueNode):
    '''
    Peak rate of increase of SAT - specific to offshore helicopter operations to detect
    transit though gas plumes. Helicopter only.
    '''

    name = 'SAT Rate Of Change Max'
    units = ut.CELSIUS

    can_operate = helicopter_only

    def derive(self, sat=P('SAT'), airborne=S('Airborne')):
        width = None if sat.frequency <= 0.25 else 4
        sat_roc = rate_of_change_array(sat.array, sat.frequency, width=width)
        self.create_kpv_from_slices(sat_roc, airborne, max_value)


##############################################################################
# Cruise Guide Indicator


class CruiseGuideIndicatorMax(KeyPointValueNode):
    '''
    Maximum CGI reading throughtout the whole record. Helicopter only.
    '''
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, cgi=P('Cruise Guide'), airborne=S('Airborne')):
        self.create_kpv_from_slices(cgi.array, airborne, max_abs_value)


##############################################################################
# Training Mode


class TrainingModeDuration(KeyPointValueNode):
    '''
    Specific to the S92 helicopter, FADEC training mode used. Helicopter only.
    '''

    units = ut.SECOND

    @classmethod
    def can_operate(cls, available):
        # S92A case
        if ('Training Mode' in available) and \
           not(any_of(('Eng (1) Training Mode', 'Eng (2) Training Mode'), available)):
            return True
        # H225 case
        elif all_of(('Eng (1) Training Mode', 'Eng (2) Training Mode'), available) and \
            ('Training Mode' not in available) :
            return True
        # No other cases operational yet.
        else:
            return False

    def derive(self, trg=P('Training Mode'),
               trg1=P('Eng (1) Training Mode'),
               trg2=P('Eng (2) Training Mode'),
               ):

        # S92A case
        if trg:
            trg_slices = runs_of_ones(trg.array)
            frequency = trg.frequency
        else:
            # H225 case
            trg_slices = slices_or(runs_of_ones(trg1.array), runs_of_ones(trg2.array))
            frequency = trg1.frequency

        self.create_kpvs_from_slice_durations(trg_slices,
                                              frequency,
                                              min_duration=2.0,
                                              mark='start')


##############################################################################
# Hover height


class HoverHeightDuringOnshoreTakeoffMax(KeyPointValueNode):
    '''
    Maximum hover height, to monitor for safe hover operation.
    Helicopter only.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, rad_alt=P('Altitude Radio'), offshore=M('Offshore'), hover=S('Hover'), toff=S('Takeoff')):
        phases = slices_and(runs_of_ones(offshore.array == 'Onshore'), hover.get_slices())
        for phase in phases:
            if toff.get(within_slice=phase, within_use='any'):
                self.create_kpvs_within_slices(rad_alt.array, [phase], max_value)


class HoverHeightDuringOffshoreTakeoffMax(KeyPointValueNode):
    '''
    Maximum hover height, to monitor for safe hover operation.
    Helicopter only.
    '''

    units = ut.FT

    can_operate = helicopter_only

    def derive(self, rad_alt=P('Altitude Radio'), offshore=M('Offshore'), hover=S('Hover'), toff=S('Takeoff')):
        phases = slices_and(runs_of_ones(offshore.array == 'Offshore'), hover.get_slices())
        for phase in phases:
            if toff.get(within_slice=phase, within_use='any'):
                self.create_kpvs_within_slices(rad_alt.array, [phase], max_value)
