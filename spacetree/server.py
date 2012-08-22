#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################

'''
Simple HTTP server for the spacetree and parameter list utilities.
'''

################################################################################
# Imports (#1)


import argparse
import httplib2
import logging
import os
import simplejson
import sys
import urllib
import webbrowser

from BaseHTTPServer import BaseHTTPRequestHandler, HTTPServer
from cgi import FieldStorage
from datetime import datetime
from jinja2 import Environment, PackageLoader
from tempfile import mkstemp
from urlparse import urlparse


################################################################################
# Logging Configuration


logging.getLogger('analysis_engine').addHandler(logging.NullHandler())


################################################################################
# Imports (#2)


from hdfaccess.file import hdf_file

from analysis_engine import settings
from analysis_engine.dependency_graph import (
    graph_adjacencies,
    graph_nodes,
    process_order,
)
from analysis_engine.node import NodeManager
from analysis_engine.process_flight import get_derived_nodes

from browser import register_additional_browsers


################################################################################
# Constants


DEFAULT_HOST = 'localhost'
DEFAULT_PORT = 8080

# TODO: Change to 'production' when updated to include Parameter API.
ENVIRONMENT = 'test'
SUFFIX = '' if ENVIRONMENT == 'production' else '-%s' % ENVIRONMENT
BASE_URL = 'https://polaris%s.flightdataservices.com' % SUFFIX

FILE_EXT_MODE_MAP = {
    # Images:
    'gif': 'rb',
    'ico': 'rb',
    'jpg': 'rb',
    'png': 'rb',
}
FILE_EXT_TYPE_MAP = {
    # Styles:
    'css': 'text/css',
    # Scripts:
    'js': 'text/javascript',
    'json': 'application/json',
    # Images:
    'gif': 'image/gif',
    'ico': 'image/x-icon',
    'jpg': 'image/jpeg',
    'png': 'image/png',
}


################################################################################
# Helpers


def parse_arguments():
    '''
    '''

    def port_range(string):
        '''
        Range type used by argument parser for port values.
        '''
        try:
            value = int(string)
            if not 0 < value <= 65535:
                raise ValueError('Port number out-of-range.')
        except:
            msg = '%r is not a valid port number' % string
            raise argparse.ArgumentTypeError(msg)
        return value

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description='Flight Data Parameter Tree Viewer',
    )

    parser.add_argument('-n', '--no-browser',
        action='store_false',
        dest='browser',
        help='Don\'t launch a browser when starting the server.',
    )

    parser.add_argument('-p', '--port',
        default=DEFAULT_PORT,
        type=port_range,
        help='Port on which to run the server (default: %(default)d)',
    )

    return parser.parse_args()


def lookup_path(relative_path):
    '''
    Convert a relative path to the asset path. Accounts for being frozen.
    '''
    file_path = relative_path.lstrip('/').replace('/', os.sep)
    if getattr(sys, 'frozen', False):
        # http://www.pyinstaller.org/export/v1.5.1/project/doc/Manual.html?format=raw#accessing-data-files
        if '_MEIPASS2' in os.environ:
            # --onefile distribution
            return os.path.join(os.environ['_MEIPASS2'], file_path)
        else:
            # --onedir distribution
            return os.path.join(os.path.dirname(sys.executable), file_path)
    else:
        return file_path


################################################################################
# Handlers


class SpacetreeRequestHandler(BaseHTTPRequestHandler):
    '''
    '''

    _template_pkg = ('spacetree', lookup_path('templates'))
    _template_env = Environment(loader=PackageLoader(*_template_pkg))

    ####################################
    # Response Helper Methods

    def _respond(self, body, status=200, content_type='text/html'):
        '''
        Respond with body setting status and content-type.
        '''
        self.send_response(status)
        self.send_header('Content-Type', content_type)
        self.end_headers()
        self.wfile.write(body)

    def _respond_with_template(self, template_path, context={}, **kwargs):
        '''
        Respond with a rendered template.
        '''
        template = self._template_env.get_template(template_path)
        self._respond(template.render(**context), **kwargs)

    def _respond_with_error(self, status, message):
        '''
        Respond with error status code and message.
        '''
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.send_error(status, message)

    def _respond_with_static(self, path):
        '''
        '''
        # Lookup the file and content type:
        file_path = lookup_path(path)
        ext = os.path.splitext(file_path)[-1][1:]  # Remove the leading dot.
        mode_ = FILE_EXT_MODE_MAP.get(ext, 'r')
        type_ = FILE_EXT_TYPE_MAP.get(ext, 'text/html')

        # Attempt to serve resource from the current directory:
        self._respond(open(file_path, mode_).read(), 200, type_)

    ####################################
    # HTTP Method Handlers

    def do_POST(self):
        '''
        '''
        if self.path.endswith('/spacetree'):
            self._spacetree()
            return

    def do_GET(self):
        '''
        '''
        parsed_url = urlparse(self.path)
        path = parsed_url.path
        if path == '/':
            self._index()  # File upload page with aircraft information.
            return
        elif path.endswith('/parameters'):
            self._parameters()
            return
        elif path.endswith('/spacetree'):
            self._index()  # Redirect to index if no HDF file in POST.
            return
        elif path.endswith('/favicon.ico'):
            self._respond_with_static('_assets/img/' + path)
            return
        elif path.startswith('/_assets'):
            try:
                self._respond_with_static(path)
                return
            except IOError:
                pass
        self._respond_with_error(404, 'Page Not Found %s' % self.path)

    ####################################
    # Page Response Methods

    def _index(self, error=None):
        '''
        :param error: Optional error to display with the form.
        :type error: str
        '''
        self._respond_with_template('index.html', {
            'error': error,
        })

    def _spacetree(self):
        '''
        '''
        form = FieldStorage(
            self.rfile,
            headers=self.headers,
            environ={'REQUEST_METHOD': 'POST'},
        )

        # Handle uploading of an HDF file:
        file_upload = form['hdf_file']
        if not file_upload.filename:
            self._index(error='Please select a file to upload.')
            return
        # Create a temporary file for the upload:
        file_desc, file_path = mkstemp()
        file_obj = os.fdopen(file_desc, 'w')
        file_obj.write(file_upload.file.read())
        file_obj.close()
        try:
            with hdf_file(file_path) as hdf_file_obj:
                lfl_params = hdf_file_obj.keys()
        except IOError:
            self._index(error='Please select a valid HDF file.')
            return

        # Fetch parameters to display in a grid:
        param_names = self._generate_json(lfl_params)
        polaris_query, params, missing_params = self._fetch_params(param_names)

        # Render the spacetree:
        self._respond_with_template('spacetree.html', {
            'missing_params': missing_params,
            'params': sorted(params.items()),
            'polaris_query': polaris_query,
            'server': BASE_URL,
        })

    ####################################
    # Generate Data From HDF

    def _generate_json(self, lfl_params):
        '''
        '''
        # Ensure file is a valid HDF file before continuing:
        derived_nodes = get_derived_nodes(settings.NODE_MODULES)
        required_params = derived_nodes.keys()

        # TODO: Update ac_info with keys from provided fields:
        ac_info = {
            'Family': u'B737 NG',
            'Frame': u'737-3C',
            'Identifier': u'15',
            'Main Gear To Lowest Point Of Tail': None,
            'Main Gear To Radio Altimeter Antenna': None,
            'Manufacturer Serial Number': u'39009',
            'Manufacturer': u'Boeing',
            'Model': u'B737-8JP',
            'Precise Positioning': True,
            'Series': u'B737-800',
            'Tail Number': 'G-ABCD',
        }

        # TODO: Option to populate an AFR:
        achieved_flight_record = {}

        # Generate the dependency tree:
        node_mgr = NodeManager(datetime.now(),
                               lfl_params,
                               required_params,
                               derived_nodes,
                               ac_info,
                               achieved_flight_record)
        _graph = graph_nodes(node_mgr)
        gr_all, gr_st, order = process_order(_graph, node_mgr)
        param_names = gr_st.nodes()

        # FIXME: Requires use of lookup_path()?
        # Save the dependency tree to tree.json:
        with open('_assets/ajax/tree.json', 'w') as f:
            simplejson.dump(graph_adjacencies(gr_st), f, indent=4)

        # FIXME: Requires use of lookup_path()?
        # Save the list of nodes to node_list.json:
        with open('_assets/ajax/node_list.json', 'w') as f:
            simplejson.dump(param_names, f, indent=4)

        return param_names

    ####################################
    # Fetch Parameters via REST API

    def _fetch_params(self, param_names):
        '''
        Fetch params from server.
        '''
        http = httplib2.Http(disable_ssl_certificate_validation=True)
        body = urllib.urlencode({'parameters': simplejson.dumps(param_names)})
        try:
            print >>sys.stderr, 'Fetching parameters from %s' % BASE_URL
            response, content = http.request(BASE_URL + '/api/parameter', 'POST', body)
        except Exception as err:
            print >>sys.stderr, 'Exception raised during API query:', str(err)
            polaris_query = False
            params = {param_name: {
                'database': None,
                'limits': None
            } for param_name in param_names}
        else:
            polaris_query = True
            params = simplejson.loads(content)['data']
        # FIXME: Requires use of lookup_path()?
        with open('data/mandatory_parameters.txt', 'r') as f:
            mandatory_params = f.read().splitlines()
        for param_name, param_info in params.iteritems():
            param_info['mandatory'] = param_name in mandatory_params
        missing_params = set(mandatory_params) - set(param_names)
        return polaris_query, params, sorted(missing_params)


################################################################################
# Program


if __name__ == '__main__':

    opt = parse_arguments()

    url = 'http://%s:%d/' % (DEFAULT_HOST, opt.port)

    server = HTTPServer((DEFAULT_HOST, opt.port), SpacetreeRequestHandler)
    print >>sys.stderr, 'Spacetree server is running at %s' % url
    print >>sys.stderr, 'Quit the server with CONTROL-C.'

    if opt.browser:
        print >>sys.stderr, 'Registering additional web browsers...'
        register_additional_browsers()
        print >>sys.stderr, 'Launching viewer in a web browser...'
        webbrowser.open_new_tab(url)
    else:
        print >>sys.stderr, 'Browse to the above location to use the viewer...'

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print >>sys.stderr, '\nShutting down server...'
        server.socket.close()


################################################################################
# vim:et:ft=python:nowrap:sts=4:sw=4:ts=4
