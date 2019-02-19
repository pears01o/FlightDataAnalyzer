# -*- coding: utf-8 -*-

from __future__ import print_function

import numpy as np

from analysis_engine.node import (
    A, M, P, S, helicopter, helicopter_only,
    MultistateDerivedParameterNode
)

from analysis_engine.library import (
    align,
    all_of,
    any_of,
    mask_inside_slices,
    merge_two_parameters,
    moving_average,
    nearest_neighbour_mask_repair,
    np_ma_zeros_like,
    offset_select,
    repair_mask,
    runs_of_ones,
    slices_and,
    slices_remove_small_slices,
    vstack_params_where_state
)

from analysis_engine.settings import (
    AUTOROTATION_SPLIT,
    ROTORS_TURNING
)


class AllEnginesOperative(MultistateDerivedParameterNode):
    '''
    Any Engine is running neither is OEI

    OEI: One Engine Inoperative
    AEO: All Engines Operative
    '''

    values_mapping = {
        0: '-',
        1: 'AEO',
    }

    can_operate = helicopter_only

    def derive(self,
               any_running=M('Eng (*) Any Running'),
               eng_oei=M('One Engine Inoperative'),
               autorotation=S('Autorotation')):
        aeo = np.ma.logical_not(eng_oei.array == 'OEI')
        for section in autorotation:
            aeo[section.slice] = False
        self.array = np.ma.logical_and(any_running.array == 'Running', aeo)


class ASEEngaged(MultistateDerivedParameterNode):
    '''
    Determines if *any* of the "ASE (*) Engaged" parameters are recording the
    state of Engaged.

    This is a discrete with only the Engaged state.
    '''

    name = 'ASE Engaged'
    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type and ac_type.value == 'helicopter' and \
               any_of(cls.get_dependency_names(), available)

    def derive(self,
               ase1=M('ASE (1) Engaged'),
               ase2=M('ASE (2) Engaged'),
               ase3=M('ASE (3) Engaged')):
        stacked = vstack_params_where_state(
            (ase1, 'Engaged'),
            (ase2, 'Engaged'),
            (ase3, 'Engaged'),
        )
        self.array = stacked.any(axis=0)
        self.offset = offset_select('mean', [ase1, ase2, ase3])


class ASEChannelsEngaged(MultistateDerivedParameterNode):
    '''
    '''
    name = 'ASE Channels Engaged'
    values_mapping = {0: '-', 1: 'Single', 2: 'Dual', 3: 'Triple'}

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return ac_type and ac_type.value == 'helicopter' and len(available) >= 2

    def derive(self,
               ase1=M('ASE (1) Engaged'),
               ase2=M('ASE (2) Engaged'),
               ase3=M('ASE (3) Engaged')):
        stacked = vstack_params_where_state(
            (ase1, 'Engaged'),
            (ase2, 'Engaged'),
            (ase3, 'Engaged'),
        )
        self.array = stacked.sum(axis=0)
        self.offset = offset_select('mean', [ase1, ase2, ase3])


class Eng1OneEngineInoperative(MultistateDerivedParameterNode):
    '''
    Look for at least 1% difference between Eng 2 N2 speed and the rotor speed to indicate
    Eng 1 can use OEI limits.

    OEI: One Engine Inoperative
    '''

    name = 'Eng (1) One Engine Inoperative'

    values_mapping = {
        0: '-',
        1: 'Active',
    }

    can_operate = helicopter_only

    def derive(self,
               eng_2_n2=P('Eng (2) N2'),
               nr=P('Nr'),
               autorotation=S('Autorotation')):

        nr_periods = np.ma.masked_less(nr.array, 80)
        nr_periods = mask_inside_slices(nr_periods, autorotation.get_slices())
        delta = nr_periods - eng_2_n2.array
        self.array = np.ma.where(delta > AUTOROTATION_SPLIT, 'Active', '-')


class Eng2OneEngineInoperative(MultistateDerivedParameterNode):
    '''
    Look for at least 1% difference between Eng 1 N2 speed and the rotor speed to indicate
    Eng 1 can use OEI limits.

    OEI: One Engine Inoperative
    '''

    name = 'Eng (2) One Engine Inoperative'

    values_mapping = {
        0: '-',
        1: 'Active',
    }

    can_operate = helicopter_only

    def derive(self,
               eng_1_n2=P('Eng (1) N2'),
               nr=P('Nr'),
               autorotation=S('Autorotation')):

        nr_periods = np.ma.masked_less(nr.array, 80)
        nr_periods = mask_inside_slices(nr_periods, autorotation.get_slices())
        delta = nr_periods - eng_1_n2.array
        self.array = np.ma.where(delta > AUTOROTATION_SPLIT, 'Active', '-')


class GearOnGround(MultistateDerivedParameterNode):
    '''
    Combination of left and right main gear signals.
    '''
    align = False
    values_mapping = {
        0: 'Air',
        1: 'Ground',
    }

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        gog_available = any_of(('Gear (L) On Ground', 'Gear (R) On Ground'), available)
        if gog_available:
            return True
        elif all_of(('Vertical Speed', 'Eng (*) Torque Avg'), available):
            return True
        else:
            return False

    def derive(self,
               gl=M('Gear (L) On Ground'),
               gr=M('Gear (R) On Ground'),
               vert_spd=P('Vertical Speed'),
               torque=P('Eng (*) Torque Avg'),
               ac_series=A('Series'),
               collective=P('Collective')):

        if gl and gr:
            delta = abs((gl.offset - gr.offset) * gl.frequency)
            if 0.75 < delta or delta < 0.25:
                # If the samples of the left and right gear are close together,
                # the best representation is to map them onto a single
                # parameter in which we accept that either wheel on the ground
                # equates to gear on ground.
                self.array = np.ma.logical_or(gl.array, gr.array)
                self.frequency = gl.frequency
                self.offset = gl.offset
                return
            else:
                # If the paramters are not co-located, then
                # merge_two_parameters creates the best combination possible.
                self.array, self.frequency, self.offset = merge_two_parameters(gl, gr)
                return
        elif gl or gr:
            gear = gl or gr
            self.array = gear.array
            self.frequency = gear.frequency
            self.offset = gear.offset
        elif vert_spd and torque:
            vert_spd_limit = 100.0
            torque_limit = 30.0
            if ac_series and ac_series.value == 'Columbia 234':
                vert_spd_limit = 125.0
                torque_limit = 22.0
                collective_limit = 15.0

                vert_spd_array = align(vert_spd, torque) if vert_spd.hz != torque.hz else vert_spd.array
                collective_array = align(collective, torque) if collective.hz != torque.hz else collective.array

                vert_spd_array = moving_average(vert_spd_array)
                torque_array = moving_average(torque.array)
                collective_array = moving_average(collective_array)

                roo_vs_array = runs_of_ones(abs(vert_spd_array) < vert_spd_limit, min_samples=1)
                roo_torque_array = runs_of_ones(torque_array < torque_limit, min_samples=1)
                roo_collective_array = runs_of_ones(collective_array < collective_limit, min_samples=1)

                vs_and_torque = slices_and(roo_vs_array, roo_torque_array)
                grounded = slices_and(vs_and_torque, roo_collective_array)

                array = np_ma_zeros_like(vert_spd_array)
                for _slice in slices_remove_small_slices(grounded, count=2):
                    array[_slice] = 1
                array.mask = vert_spd_array.mask | torque_array.mask
                array.mask = array.mask | collective_array.mask
                self.array = nearest_neighbour_mask_repair(array)
                self.frequency = torque.frequency
                self.offset = torque.offset

            else:
                vert_spd_array = align(vert_spd, torque) if vert_spd.hz != torque.hz else vert_spd.array
                # Introducted for S76 and Bell 212 which do not have Gear On Ground available

                vert_spd_array = moving_average(vert_spd_array)
                torque_array = moving_average(torque.array)

                grounded = slices_and(runs_of_ones(abs(vert_spd_array) < vert_spd_limit, min_samples=1),
                                      runs_of_ones(torque_array < torque_limit, min_samples=1))

                array = np_ma_zeros_like(vert_spd_array)
                for _slice in slices_remove_small_slices(grounded, count=2):
                    array[_slice] = 1
                array.mask = vert_spd_array.mask | torque_array.mask
                self.array = nearest_neighbour_mask_repair(array)
                self.frequency = torque.frequency
                self.offset = torque.offset

        else:
            # should not get here if can_operate is correct
            raise NotImplementedError()


class OneEngineInoperative(MultistateDerivedParameterNode):
    '''
    Any Engine is running either engine is OEI

    OEI: One Engine Inoperative
    '''

    values_mapping = {
        0: '-',
        1: 'OEI',
    }

    can_operate = helicopter_only

    def derive(self,
               eng_1_oei=M('Eng (1) One Engine Inoperative'),
               eng_2_oei=M('Eng (2) One Engine Inoperative'),
               autorotation=S('Autorotation')):

        oei = vstack_params_where_state((eng_1_oei, 'Active'),
                                        (eng_2_oei, 'Active')).any(axis=0)
        for section in autorotation:
            oei[section.slice] = False
        self.array = oei


class RotorBrakeEngaged(MultistateDerivedParameterNode):
    ''' Discrete parameter describing when any rotor brake is engaged. '''

    values_mapping = {0: '-', 1: 'Engaged'}

    @classmethod
    def can_operate(cls, available, ac_type=A('Aircraft Type')):
        return any_of(cls.get_dependency_names(), available) and \
               ac_type == helicopter

    def derive(self,
               brk1=M('Rotor Brake (1) Engaged'),
               brk2=M('Rotor Brake (2) Engaged')):

        stacked = vstack_params_where_state(
            (brk1, 'Engaged'),
            (brk2, 'Engaged'),
        )
        self.array = stacked.any(axis=0)
        self.array.mask = stacked.mask.any(axis=0)


class RotorsRunning(MultistateDerivedParameterNode):
    '''

    '''

    values_mapping = {
        0: 'Not Running',
        1: 'Running',
    }

    can_operate = helicopter_only

    def derive(self, nr=P('Nr')):
        self.array = np.ma.where(repair_mask(nr.array) > ROTORS_TURNING, 'Running', 'Not Running')
