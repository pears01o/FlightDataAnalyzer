from __future__ import print_function, unicode_literals

import argparse
import csv
import itertools
import logging
import numpy as np
import os
import simplekml
import six

from copy import copy

from hdfaccess.file import hdf_file
from flightdatautilities import units as ut
from flightdatautilities.print_table import indent

from analysis_engine.library import (
    bearing_and_distance, 
    latitudes_and_longitudes, 
    repair_mask, 
    value_at_index,
)
from analysis_engine.node import derived_param_from_hdf, Parameter

import matplotlib

try:
    import wx
except ImportError:
    matplotlib.use('Agg')
else:
    matplotlib.use('WXAgg')

import matplotlib.pyplot as plt

'''
Note: if you are having problems with blocking plots try
    import matplotlib.pyplot as plt
    plt.ion()

This turns interactive mode on. If this works there is an interactive setting
in matplotlibrc to turn this on globally
'''

logger = logging.getLogger(name=__name__)

# KPV / KTI names not to display as markers
SKIP_KPVS = []
KEEP_KPVS = ['ILS Frequency During Approach']
SKIP_KTIS = ['Transmit']
KEEP_KTIS = ['Takeoff Start','Liftoff', 'Touchdown', 
             'Localizer Established Start', 'Localizer Established End', 
             'Glideslope Established Start', 'Glideslope Established End',
             'IAN Final Approach Established Start', 'IAN Final Approach Established End',
             'IAN Glidepath Established Start', 'IAN Glidepath Established End',
             ]

class TypedWriter(object):
    """
    A CSV writer which will write rows to CSV file "f",
    which uses "fieldformats" to format fields.
    
    ref: http://stackoverflow.com/questions/2982642/specifying-formatting-for-csv-writer-in-python
    """

    def __init__(self, f, fieldnames, fieldformats, skip_header=False, **kwds):
        self.writer = csv.DictWriter(f, fieldnames, **kwds)
        if not skip_header:
            self.writer.writeheader()

        self.formats = fieldformats

    def _format(self, row):
        return dict((k, self.formats.get(k, '%s') % v if v or v == 0.0 else v) 
                    for k, v in six.iteritems(row))
    
    def writerow(self, row):
        self.writer.writerow(self._format(row))

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
            
    def rowlist(self, rows):
        "Return a list of formatted rows as ordered by fieldnames"
        res = []
        for row in rows:
            res.append(self.writer._dict_to_list(self._format(row)))
        return res
            
            
def add_track(kml, track_name, lat, lon, colour, alt_param=None, alt_mode=None,
              visible=True):
    '''
    alt_mode such as simplekml.constants.AltitudeMode.clamptoground
    '''
    track_config = {'visibility': 1 if visible else 0}
    if alt_param:
        track_config['altitudemode'] = alt_mode
        track_config['extrude'] = 1
    
    tracks = []
    track_coords = []
    count = 0
    ##scope_lon = np.ma.flatnotmasked_edges(lon.array)
    ##scope_lat = np.ma.flatnotmasked_edges(lat.array)
    ##begin = max(scope_lon[0], scope_lat[0])+1
    ##end = min(scope_lon[1], scope_lat[1])-1
    for i in range(0, len(lon.array)):
        _lon = value_at_index(lon.array, i)
        _lat = value_at_index(lat.array, i)
        if alt_param:
            _alt = value_at_index(alt_param.array, i)
            coords = (_lon, _lat, _alt)
        else:
            coords = (_lon, _lat)
            
        if not all(coords) or any(np.isnan(coords)):
            continue
        track_coords.append(coords)
        count += 1
        # Split up tracks because Google Earth cannot extrude long LineStrings.
        if count >= 4000:
            tracks.append(track_coords)
            track_coords = []
            count = 0
    if track_coords:
        tracks.append(track_coords)
    
    for index, track_coords in enumerate(tracks, start=1):
        track_config = track_config.copy()
        track_config['name'] = '%s (%d)' % (track_name, index)
        track_config['coords'] = track_coords
        line = kml.newlinestring(**track_config)
        line.style.linestyle.color = colour
        line.style.polystyle.color = '66%s' % colour[2:] # set opacity of area fill to 40%
    return


def draw_centreline(kml, rwy):
    start_lat = rwy['start']['latitude']
    start_lon = rwy['start']['longitude']
    end_lat = rwy['end']['latitude']
    end_lon = rwy['end']['longitude']
    brg, dist = bearing_and_distance(end_lat, end_lon, start_lat, start_lon)
    brgs = np.ma.array([brg])
    dists = np.ma.array([30000])
    lat_30k, lon_30k = latitudes_and_longitudes(brgs, dists, rwy['start'])
    try:
        angle = np.deg2rad(rwy['glideslope']['angle'])
    except:
        angle = np.deg2rad(3.0)
    end_height = ut.convert(30000 * np.tan(angle), ut.METER, ut.FT)
    track_config = {'name': 'ILS'}
    track_coords = []
    track_coords.append((end_lon,end_lat))
    track_coords.append((lon_30k.data[0],lat_30k.data[0], end_height))
    track_config['coords'] = track_coords
    kml.newlinestring(**track_config)
    return


def track_to_kml(hdf_path, kti_list, kpv_list, approach_list,
                 plot_altitude=None, dest_path=None):
    '''
    Plot results of process_flight onto a KML track.
    
    :param flight_attrs: List of Flight Attributes
    :type flight_attrs: list
    :param plot_altitude: Name of Altitude parameter to use in KML
    :type plot_altitude: String
    '''
    one_hz = Parameter()
    kml = simplekml.Kml()
    with hdf_file(hdf_path) as hdf:
        # Latitude param, Longitude param, track name, colour
        coord_params = (
            {'lat': 'Latitude Smoothed',
             'lon': 'Longitude Smoothed',
             'track': 'Smoothed',
             'colour': 'ff7fff7f'},
            {'lat': 'Latitude Prepared',
             'lon': 'Longitude Prepared',
             'track': 'Prepared',
             'colour': 'ffA11EB3'},
            {'lat': 'Latitude',
             'lon': 'Longitude',
             'track': 'Recorded',
             'colour': 'ff0000ff'},
            {'lat': 'Latitude (Coarse)',
             'lon': 'Longitude (Coarse)',
             'track': 'Coarse',
             'colour': 'ff0000ff'},
        )
        altitude_absolute_params = ('Altitude Visualization With Ground Offset',
                                    'Altitude QNH', 'Altitude STD',
                                    'Altitude AAL')
        altitude_relative_params = ('Altitude Radio',)
        # Check latitude and longitude pair exist.
        if not any(c['lat'] in hdf and c['lon'] in hdf for c in coord_params):
            logger.error("Cannot write track as coordinate paarmeters not in hdf")
            return False
        # Choose best altitude parameter if not specified.
        if not plot_altitude:
            altitude_params = itertools.chain(altitude_absolute_params,
                                              altitude_relative_params)
            try:
                plot_altitude = next(p for p in altitude_params if p in hdf)
            except StopIteration:
                logger.warning("Disabling altitude on KML plot as it is "
                               "unavailable.")
        # Get altitude param from hdf.
        if plot_altitude and plot_altitude in hdf.keys():
            alt = derived_param_from_hdf(hdf[plot_altitude]).get_aligned(one_hz)
            alt.array = repair_mask(alt.array, frequency=alt.frequency, repair_duration=None)
            alt.array = ut.convert(alt.array, ut.FT, ut.METER)
        else:
            alt = None
        
        if plot_altitude in altitude_absolute_params:
            altitude_mode = simplekml.constants.AltitudeMode.absolute
        elif plot_altitude in altitude_relative_params:
            altitude_mode = simplekml.constants.AltitudeMode.relativetoground
        else:
            altitude_mode = simplekml.constants.AltitudeMode.clamptoground
        
        ## Get best latitude and longitude parameters.
        best_lat = None
        best_lon = None
        
        for coord_config in coord_params:
            lat_name = coord_config['lat']
            lon_name = coord_config['lon']
            if not lat_name in hdf or not lon_name in hdf:
                continue
            lat = hdf[lat_name]
            lon = hdf[lon_name]
            best = not best_lat or not best_lon
            add_track(kml, coord_config['track'], lat, lon,
                      coord_config['colour'], alt_param=alt,
                      alt_mode=altitude_mode,
                      visible=best)
            add_track(kml, coord_config['track'] + ' On Ground', lat, lon,
                      coord_config['colour'], visible=best)
            if best:
                best_lat = derived_param_from_hdf(lat).get_aligned(one_hz)
                best_lon = derived_param_from_hdf(lon).get_aligned(one_hz)
    
    # Add KTIs.
    for kti in kti_list:
        kti_point_values = {'name': kti.name}
        
        if not KEEP_KTIS and kti.name in SKIP_KTIS:
            continue
        elif len(KEEP_KTIS)>0 and (kti.name not in KEEP_KTIS):
            continue
        
        altitude = alt.at(kti.index) if alt else None
        kti_point_values['altitudemode'] = altitude_mode
        if altitude:
            kti_point_values['coords'] = ((kti.longitude, kti.latitude, altitude),)
        else:
            kti_point_values['coords'] = ((kti.longitude, kti.latitude),)
        kml.newpoint(**kti_point_values)
    
    # Add KPVs.
    for kpv in kpv_list:

        # Trap kpvs with invalid latitude or longitude data (normally happens
        # at the start of the data where accelerometer offsets are declared,
        # and this avoids casting kpvs into the Atlantic.
        kpv_lat = best_lat.at(kpv.index)
        kpv_lon = best_lon.at(kpv.index)
        if kpv_lat is None or kpv_lon is None or \
           (kpv_lat == 0.0 and kpv_lon == 0.0):
            continue

        if not KEEP_KPVS and kpv.name in SKIP_KPVS:
            continue
        elif len(KEEP_KPVS)>0 and (kpv.name not in KEEP_KPVS):
            continue
        
        style = simplekml.Style()
        style.iconstyle.color = simplekml.Color.red
        kpv_point_values = {'name': '%s (%.3f)' % (kpv.name, kpv.value)}
        altitude = alt.at(kpv.index) if alt else None
        kpv_point_values['altitudemode'] = altitude_mode
        if altitude:
            kpv_point_values['coords'] = ((kpv_lon, kpv_lat, altitude),)
        else:
            kpv_point_values['coords'] = ((kpv_lon, kpv_lat),)
        
        pnt = kml.newpoint(**kpv_point_values)
        pnt.style = style
    
    # Add approach centre lines.
    for app in approach_list:
        try:
            draw_centreline(kml, app.runway)
        except:
            pass

    if not dest_path:
        dest_path = hdf_path + ".kml"
    kml.save(dest_path)
    return dest_path


fig = plt.figure() 
def plot_parameter(array, new_subplot=False, show=True, label='', marker=None):
    """
    For quickly plotting a single parameter to see its shape.
    
    :param array: Numpy array
    :type array: np.array
    :param marker: Optional marker format character.
    :type marker: str
    :param show: Whether to display the figure (and block)
    :type show: Boolean
    """
    if array is None:
        print("Cannot plot as array is None!")
        return
    n = len(fig.axes)
    if n:
        if new_subplot:
            # resize
            for i in range(n):
                fig.axes[i].change_geometry(n+1, 1, i+1)
            # add the new subplot
            ax = fig.add_subplot(n+1, 1, n+1, sharex=fig.axes[0])
        else:
            # re-use last one
            ax = fig.axes[-1]
    else:
        ax = fig.add_subplot(111)

    # Plot the data
    label = label or "Length: %d | Min: %.2f | Max: %.2f" % (
        len(array), array.min(), array.max())
    ax.plot(array, marker=marker, label=label)
    if show:
        plt.show()
    

def plot_essential(hdf_path):
    """
    Plot the essential parameters for flight analysis.
    
    Assumes hdf_path file contains the parameter series:
    Frame Counter, Airspeed, Altitude STD, Head True
    
    show() is to be called elsewhere (from matplotlib.pyplot import show)
    
    :param hdf_path: Path to HDF file.
    :type hdf_path: string
    """
    fig = plt.figure() ##figsize=(10,8))
    plt.title(os.path.basename(hdf_path))
    
    with hdf_file(hdf_path) as hdf:
        ax1 = fig.add_subplot(4,1,1)
        #ax1.set_title('Frame Counter')
        ax1.plot(hdf['Frame Counter'].array, 'k--')
        ax2 = fig.add_subplot(4,1,2)
        ax2.plot(hdf['Airspeed'].array, 'r-')
        ax3 = fig.add_subplot(4,1,3,sharex=ax2)
        ax3.plot(hdf['Altitude STD'].array, 'g-')
        ax4 = fig.add_subplot(4,1,4,sharex=ax2)
        ax4.plot(hdf['Heading'].array, 'b-')    


def plot_flight(hdf_path, kti_list, kpv_list, phase_list, aircraft_info):
    """
    """
    fig = plt.figure() ##figsize=(10,8))
    plt.title(os.path.basename(hdf_path))
    
    with hdf_file(hdf_path) as hdf:
        #---------- Axis 1 ----------
        ax1 = fig.add_subplot(4,1,1)
        alt_data = hdf['Altitude STD'].array
        alt = hdf.get('Altitude AAL For Flight Phases',
                      hdf['Altitude STD']).array
        frame = hdf['Time'].array
        #frame = hdf.get('Frame Counter',hdf['Altitude STD']).array
        
        sections = []
        sections.append(alt_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Landing', 'Airborne', 'Grounded'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice])
            sections.append(alt[phase.slice])
            if phase.name == 'Takeoff':
                sections.append('r-')
            elif phase.name == 'Landing':
                sections.append('g-')
            elif phase.name == 'Airborne':
                sections.append('b-')
            elif phase.name == 'Grounded':
                sections.append('k-')
        ax1.plot(*sections)
        
        #---------- Axis 2 ----------
        ax2 = fig.add_subplot(4,1,2)
        #ax2 = fig.add_subplot(4,1,2,sharex=ax1)
        vert_spd = hdf.get('Vertical Speed For Flight Phases', hdf['Altitude STD']).array
        vert_spd_data = hdf.get('Vertical Speed', hdf['Altitude STD']).array
        sections = []
        sections.append(vert_spd_data)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Takeoff', 'Level Flight', 'Descending'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice]-frame[0])
            sections.append(vert_spd[phase.slice])
            if phase.name == 'Takeoff':
                sections.append('g-')
            elif phase.name == 'Level Flight':
                sections.append('b-')
            elif phase.name == 'Descending':
                sections.append('c-')
        ax2.plot(*sections)
        
        #---------- Axis 3 ----------
        ax3 = fig.add_subplot(4,1,3)
        #ax3 = fig.add_subplot(4,1,3,sharex=ax1)
        airspeed = hdf.get('Airspeed',hdf['Altitude STD']).array
        sections = []
        sections.append(airspeed)
        sections.append('k-')
        for phase in filter(lambda p: p.name in (
            'Fast'), phase_list):
            # Declare the x-axis parameter first...
            sections.append(frame[phase.slice]-frame[0])
            sections.append(airspeed[phase.slice])
            if phase.name == 'Fast':
                sections.append('r-')
        
        ax3.plot(*sections)
        
        #---------- Axis 4 ----------
        if 'Heading Continuous' in hdf:
            ax4 = fig.add_subplot(4,1,4,sharex=ax1)
            ax4.plot(hdf['Heading Continuous'].array, 'b-')  
    
    for kpv in kpv_list:
        label = '%s %s' % (kpv.name, kpv.value)
        ax1.annotate(label, xy=(kpv.index, alt[kpv.index]),
                     xytext=(-5, 100), 
                     textcoords='offset points',
                     #arrowprops=dict(arrowstyle="->"),
                     rotation='vertical'
                     )
    '''
    for kti in kti_list:
        label = '%s' % (kti.name)
        ax1.annotate(label, xy=(kti.index, alt[kti.index]),
                     xytext=(-5, 100), 
                     textcoords='offset points',
                     #arrowprops=dict(arrowstyle="->"),
                     rotation='vertical'
                     )
    '''
    plt.show()
    return


def _index_or_slice(x):
    try:
        return float(x.index)
    except (TypeError, AttributeError):
        return x.slice.start


def csv_flight_details(hdf_path, kti_list, kpv_list, phase_list,
                       dest_path=None, append_to_file=True):
    """
    Currently writes to csv and prints to a table.
    
    Phase types have a 'duration' column
    
    :param dest_path: Outputs CSV to dest_path (removing if exists). If None,
      collates results by appending to a single file: 'combined_test_output.csv'
    """
    rows = []
    params = ['Airspeed', 'Altitude AAL']
    attrs = ['value', 'datetime', 'latitude', 'longitude'] 
    header = ['path', 'type', 'index', 'duration', 'name'] + attrs + params
    if not dest_path:
        header.append('Path')
    formats = {'index': '%.3f',
               'value': '%.3f',
               'duration': '%.2f',
               'latitude': '%.4f',
               'longitude': '%.4f',
               'Airspeed': '%d kts',
               'Altitude AAL': '%d ft',
               }
    for value in kti_list:
        vals = value._asdict()  # recordtype
        vals['path'] = hdf_path
        vals['type'] = 'Key Time Instance'
        rows.append( vals )

    for value in kpv_list:
        vals = value._asdict()  # recordtype
        vals['path'] = hdf_path
        vals['type'] = 'Key Point Value'
        rows.append( vals )

    for value in phase_list:
        vals = value._asdict()  # namedtuple
        vals['name'] = value.name + ' [START]'
        vals['path'] = hdf_path
        vals['type'] = 'Phase'
        vals['index'] = value.start_edge
        vals['duration'] = value.stop_edge - value.start_edge  # (secs)
        rows.append(vals)
        # create another at the stop of the phase
        end = copy(vals)
        end['name'] = value.name + ' [END]'
        end['index'] = value.stop_edge
        rows.append(end)
    
    # Append values of useful parameters at this time
    with hdf_file(hdf_path) as hdf:
        for param in params:
            # Create DerivedParameterNode to utilise the .at() method
            if param not in hdf:
                continue
            p = hdf[param]
            dp = Parameter(name=p.name, array=p.array, 
                           frequency=p.frequency, offset=p.offset)
            for row in rows:
                row[param] = dp.at(row['index'])

    # sort rows
    rows = sorted(rows, key=lambda x: x['index'])

    skip_header = False

    # print to CSV
    if not dest_path:
        dest_path = 'combined_test_output.csv'
    elif os.path.isfile(dest_path):
        if not append_to_file:
            logger.info("Deleting existing copy of: %s", dest_path)
            os.remove(dest_path)
        else:
            # If we are appending and the file exista, we don't want to output
            # the header again
            skip_header = True

    with open(dest_path, 'a') as dest:
        writer = TypedWriter(dest, fieldnames=header, fieldformats=formats,
                             skip_header=skip_header, extrasaction='ignore')
        writer.writerows(rows)
        # print to Debug I/O
        logger.debug(indent([header] + writer.rowlist(rows), hasHeader=True, wrapfunc=lambda x:str(x)))
    return rows




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot a flight.")
    parser.add_argument('file', type=str,
                        help='Path of file to process.')
    parser.add_argument('-tail', dest='tail_number', type=str, default='G-ABCD',
                        help='Aircraft Tail Number for processing.')
    parser.add_argument('-frame', dest='frame', type=str, default=None,
                        help='Data frame name.')
    args = parser.parse_args()    
       
    plot_flight(args.file, [], [], [], {
        'Tail Number': args.tail_number,
        'Precise Positioning': True,
        'Frame': args.frame})
