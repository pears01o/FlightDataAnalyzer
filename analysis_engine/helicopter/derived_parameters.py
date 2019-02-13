# -*- coding: utf-8 -*-
from __future__ import print_function

import numpy as np

from flightdatautilities import units as ut

from analysis_engine.node import (
    A, M, P, KTI, DerivedParameterNode, helicopter, helicopter_only
)

from analysis_engine.library import (
    any_of,
    all_of,
    bearings_and_distances,
    blend_two_parameters,
    cycle_finder,
    hysteresis,
    integrate,
    moving_average,
    np_ma_masked_zeros_like,
    repair_mask,
    slices_and,
    slices_int,
    slices_find_small_slices,
    slices_remove_small_slices
)

from analysis_engine.settings import (
    ALTITUDE_AGL_SMOOTHING,
    ALTITUDE_AGL_TRANS_ALT
)


class ApproachRange(DerivedParameterNode):
    '''
    This is the range to the touchdown point for both ILS and visual
    approaches including go-arounds. The reference point is the ILS Localizer
    antenna where the runway is so equipped, or the end of the runway where
    no ILS is available.

    The array is masked where no data has been computed, and provides
    measurements in metres from the reference point where the aircraft is on
    an approach.

    A simpler function is provided for helicopter operations as they may
    not - in fact normally do not - have a runway to land on.
    '''

    units = ut.METER

    def derive(self,
               alt_aal=P('Altitude AAL'),
               lat=P('Latitude Smoothed'),
               lon=P('Longitude Smoothed'),
               tdwns=KTI('Touchdown')):
        app_range = np_ma_masked_zeros_like(alt_aal.array)

        #Helicopter compuation does not rely on runways!
        stop_delay = 10 # To make sure the helicopter has stopped moving

        for tdwn in tdwns:
            end = tdwn.index
            endpoint = {'latitude': lat.array[int(end)],
                        'longitude': lon.array[int(end)]}
            prev_tdwn = tdwns.get_previous(end)
            begin = prev_tdwn.index + stop_delay if prev_tdwn else 0
            this_leg = slices_int(begin, end+stop_delay)
            _, app_range[this_leg] = bearings_and_distances(lat.array[this_leg],
                                                            lon.array[this_leg],
                                                            endpoint)
        self.array = app_range


class AltitudeADH(DerivedParameterNode):
    '''
    Altitude Above Deck Height

    The rate of descent will differ between the radio and pressure calculations significantly for a period
    as the aircraft comes over the deck. We test for these large differences and substitute the pressure rate
    of descent for just those samples, then reconstruct the altitude trace by integrating the corrected differential.
    '''
    name = 'Altitude ADH'

    units = ut.FT

    def derive(self, rad=P('Altitude Radio'),
               hdot=P('Vertical Speed'),
               ):

        def seek_deck(rad, hdot, min_idx, rad_hz):

            def one_direction(rad, hdot, sence, rad_hz):
                # Stairway to Heaven is getting a bit old. Getting with the times?
                # Vertical Speed / 60 = Pressure alt V/S in feet per second
                b_diffs = hdot/60

                # Rate of change on radalt array = Rad alt V/S in feet per second
                r_diffs = np.ma.ediff1d(rad*rad_hz, to_begin=b_diffs[0])

                # Difference between ROC greater than 6fps will mean flying over
                # the deck; use pressure alt roc when that happens and radio alt
                # roc in all other cases
                diffs = np.ma.where(np.ma.abs(r_diffs-b_diffs)>6.0*rad_hz, b_diffs, r_diffs)

                height = integrate(diffs,
                                   frequency=rad_hz,
                                   direction=sence,
                                   repair=False)
                return height

            height_from_rig = np_ma_masked_zeros_like(rad)
            if len(rad[:min_idx]) > 0:
                height_from_rig[:min_idx] = one_direction(rad[:min_idx], hdot[:min_idx], "backwards", rad_hz)
            if len(rad[min_idx:]) > 0:
                height_from_rig[min_idx:] = one_direction(rad[min_idx:], hdot[min_idx:], "forwards", rad_hz)

            '''
            # And we are bound to want to know the rig height somewhere, so here's how to work that out.
            rig_height = rad[0]-height_from_rig[0]
            # I checked this and it seems pretty consistent.
            # See Library\Projects\Helicopter FDM\Algorithm Development\Rig height estimates from Bond initial test data.xlsx
            #lat=hdf['Latitude'].array[app_slice][-1]
            #lon=hdf['Longitude'].array[app_slice][-1]
            #print(lat, lon, rig_height)
            '''
            return height_from_rig

        # Prepare a masked array filled with zeros for the parameter (same length as radalt array)
        self.array = np_ma_masked_zeros_like(rad.array)
        rad_peak_idxs, rad_peak_vals = cycle_finder(rad.array, min_step=150.0)


        if len(rad_peak_idxs)<4:
            return

        slice_idxs = list(zip(rad_peak_idxs[:-2], rad_peak_idxs[1:-1],
                              rad_peak_idxs[2:], rad_peak_vals[1:]))
        for slice_idx in slice_idxs[1:-1]:
            this_deck_slice = slice(slice_idx[0]+1, slice_idx[2]-1)
            if slice_idx[3] > 5.0:
                # We didn't land in this period
                continue
            else:
                self.array[this_deck_slice] = seek_deck(rad.array[this_deck_slice],
                                                        hdot.array[this_deck_slice],
                                                        slice_idx[1]-slice_idx[0],
                                                        rad.frequency)
                '''
                import matplotlib.pyplot as plt
                plt.plot(rad.array[this_deck_slice])
                plt.plot(self.array[this_deck_slice])
                plt.show()
                plt.clf()
                '''

class AltitudeAGL(DerivedParameterNode):
    '''
    This simple alorithm adopts radio altitude where available and merges pressure
    altitude values a transition altitude, joining to the radio altitude segments
    by making a linear adjustment.

    Note that at high altitudes, the pressure altitude is still corrected, so flight
    levels will be inaccurate.
    '''
    name = 'Altitude AGL'
    units = ut.FT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        if ac_type == helicopter:
            return all_of(('Altitude Radio', 'Altitude STD Smoothed', 'Gear On Ground'), available) or \
                   ('Altitude Radio' not in available and 'Altitude AAL' in available)
        else:
            return False

    def derive(self, alt_rad=P('Altitude Radio'),
               alt_aal=P('Altitude AAL'),
               alt_baro=P('Altitude STD Smoothed'),
               gog=M('Gear On Ground')):

        # If we have no Altitude Radio we will have to fall back to Altitude AAL
        if not alt_rad:
            self.array = alt_aal.array
            return

        # When was the helicopter on the ground?
        gear_on_grounds = np.ma.clump_masked(np.ma.masked_equal(gog.array, 1))
        # Find and eliminate short spikes (15 seconds) as these are most likely errors.
        short_spikes = slices_find_small_slices(gear_on_grounds, time_limit=15, hz=gog.hz)
        for slice in short_spikes:
            gog.array[slice.start:slice.stop] = 0

        # Remove slices shorter than 15 seconds as these are most likely created in error.
        gear_on_grounds = slices_remove_small_slices(gear_on_grounds, time_limit=15, hz=gog.hz)
        # Compute the half period which we will need.
        hp = int(alt_rad.frequency*ALTITUDE_AGL_SMOOTHING)/2
        # We force altitude AGL to be zero when the gear shows 'Ground' state
        alt_rad_repaired = repair_mask(alt_rad.array, frequency=alt_rad.frequency, repair_duration=20.0, extrapolate=True)
        alt_agl = moving_average(np.maximum(alt_rad.array, 0.0) * (1 - gog.array.data), window=hp*2+1, weightings=None)

        # Refine the baro estimates
        length = len(alt_agl)-1
        baro_sections = np.ma.clump_masked(np.ma.masked_greater(alt_agl, ALTITUDE_AGL_TRANS_ALT))
        for baro_section in baro_sections:
            begin = max(baro_section.start - 1, 0)
            end = min(baro_section.stop + 1, length)
            start_diff = alt_baro.array[begin] - alt_agl[begin]
            stop_diff = alt_baro.array[end] - alt_agl[end]
            if start_diff is not np.ma.masked and stop_diff is not np.ma.masked:
                diff = np.linspace(start_diff, stop_diff, end-begin-2)
                alt_agl[begin+1:end-1] = alt_baro.array[begin+1:end-1]-diff
            elif start_diff is not np.ma.masked:
                alt_agl[begin+1:end-1] = alt_baro.array[begin+1:end-1] - start_diff
            elif stop_diff is not np.ma.masked:
                alt_agl[begin+1:end-1] = alt_baro.array[begin+1:end-1] - stop_diff
            else:
                pass
        low_sections = np.ma.clump_unmasked(np.ma.masked_greater(alt_agl, 5))
        for both in slices_and(low_sections, gear_on_grounds):
            alt_agl[both] = 0.0

        '''
        # Quick visual check of the altitude agl.
        import matplotlib.pyplot as plt
        plt.plot(alt_baro.array, 'y-')
        plt.plot(alt_rad.array, 'r-')
        plt.plot(alt_agl, 'b-')
        plt.show()
        '''

        self.array = alt_agl

class AltitudeAGLForFlightPhases(DerivedParameterNode):
    '''
    This parameter repairs short periods of masked data, making it suitable for
    detecting altitude bands on the climb and descent. The parameter should not
    be used to compute KPV values themselves, to avoid using interpolated
    values in an event.

    Hysteresis avoids repeated triggering of KPVs when operating at one of
    the nominal heights. For example, helicopter searches at 500ft.
    '''

    name = 'Altitude AGL For Flight Phases'
    units = ut.FT

    def derive(self, alt_agl=P('Altitude AGL')):

        repair_array = repair_mask(alt_agl.array, repair_duration=None)
        hyst_array = hysteresis(repair_array, 10.0)
        self.array = np.ma.where(alt_agl.array > 10.0, hyst_array, repair_array)

class AltitudeDensity(DerivedParameterNode):
    '''
    Only computed for helicopters, this includes compensation for temperature changes
    that cause the air density to vary from the ISA standard.
    '''

    units = ut.FT

    def derive(self, alt_std=P('Altitude STD'), sat=P('SAT'),
               isa_temp=P('SAT International Standard Atmosphere')):
        # TODO: libary function to convert to Alitude Density see Aero Calc.
        # pressure altitude + [120 x (OAT - ISA Temp)]
        # isa_temp = 15 - 1.98 / 1000 * std_array
        self.array = alt_std.array + (120 * (sat.array - isa_temp.array))


class Collective(DerivedParameterNode):
    '''
    '''

    align = False
    units = ut.PERCENT

    def derive(self,
               capt=P('Collective (1)'),
               fo=P('Collective (2)')):

        self.array, self.frequency, self.offset = blend_two_parameters(capt, fo)


class CyclicForeAft(DerivedParameterNode):
    '''
    '''
    align = False
    name = 'Cyclic Fore-Aft'
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and any_of(cls.get_dependency_names(), available)

    def derive(self,
               capt=P('Cyclic Fore-Aft (1)'),
               fo=P('Cyclic Fore-Aft (2)')):

        self.array, self.frequency, self.offset = blend_two_parameters(capt, fo)


class CyclicLateral(DerivedParameterNode):
    '''
    '''
    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and any_of(cls.get_dependency_names(), available)

    def derive(self,
               capt=P('Cyclic Lateral (1)'),
               fo=P('Cyclic Lateral (2)')):

        self.array, self.frequency, self.offset = blend_two_parameters(capt, fo)


class CyclicAngle(DerivedParameterNode):
    '''
    '''
    align = False
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self,
               cyclic_pitch=P('Cyclic Fore-Aft'),
               cyclic_roll=P('Cyclic Lateral')):

        self.array = np.ma.sqrt(cyclic_pitch.array ** 2 + cyclic_roll.array ** 2)


##############################################################################
# Gearbox Oil


class MGBOilTemp(DerivedParameterNode):
    '''
    This derived parameter blends together two main gearbox temperatures.
    '''
    name = 'MGB Oil Temp'
    align = False
    units = ut.CELSIUS

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return any_of(('MGB Oil Temp (1)', 'MGB Oil Temp (2)'), available) \
               and ac_type == helicopter

    def derive(self, t1=P('MGB Oil Temp (1)'), t2=P('MGB Oil Temp (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(t1, t2)


class MGBOilPress(DerivedParameterNode):
    '''
    This derived parameter blends together two main gearbox pressures.
    '''
    name = 'MGB Oil Press'
    align = False
    units = ut.PSI

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return any_of(('MGB Oil Press (1)', 'MGB Oil Press (2)'), available) \
               and ac_type == helicopter

    def derive(self, p1=P('MGB Oil Press (1)'), p2=P('MGB Oil Press (2)')):
        self.array, self.frequency, self.offset = blend_two_parameters(p1, p2)


class Nr(DerivedParameterNode):
    '''
    Combination of rotor speed signals from two sources where required.
    '''

    align = False
    units = ut.PERCENT

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type == helicopter and any_of(cls.get_dependency_names(), available)

    def derive(self, p1=P('Nr (1)'), p2=P('Nr (2)')):
        self.array, self.frequency, self.offset = \
            blend_two_parameters(p1, p2)


class TailRotorPedal(DerivedParameterNode):
    '''
    '''

    align = False
    units = ut.PERCENT

    def derive(self,
               capt=P('Tail Rotor Pedal (1)'),
               fo=P('Tail Rotor Pedal (2)')):

        self.array, self.frequency, self.offset = blend_two_parameters(capt, fo)


class TorqueAsymmetry(DerivedParameterNode):
    '''
    '''

    align_frequency = 1 # Forced alignment to allow fixed window period.
    align_offset = 0
    units = ut.PERCENT

    can_operate = helicopter_only

    def derive(self, torq_max=P('Eng (*) Torque Max'), torq_min=P('Eng (*) Torque Min')):
        diff = (torq_max.array - torq_min.array)
        window = 5 # 5 second window
        self.array = moving_average(diff, window=window)

