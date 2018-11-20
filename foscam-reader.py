#!/usr/bin/env python3.6
# Support for interacting with a Foscam IP camera.
# See https://www.foscam.es/descarga/ipcam_cgi_sdk.pdf
# for documentation on the different endpoints.

import argparse
import math
import os
import requests


def videostream_fps_to_code(fps):
    """Returns the supported fps and code at or just below fps.

    If the frame rate isn't directly supported, the next lower
    rate is returned, on the assumption that the goal is to
    not overwhelm the receiver of the stream. The exception is
    when fps is too low, in which case the lowest
    supported rate code is returned.
    """
    if fps is None:
        return 0

    fps_to_code = [
    (30., 0),  # This is a guess at the full rate.
    (20., 1),
    (15., 3),
    (10., 6),
    (5., 11),
    (4., 12),
    (3., 13),
    (2., 14),
    (1., 15),
    (1.0/2, 17),
    (1.0/3, 19),
    (1.0/4, 21),
    (1.0/5, 23),
    ]

    for supported_fps, fps_code in fps_to_code:
        if fps >= supported_fps:
            return supported_fps, fps_code
    return fps_to_code[-1]


class Foscam(object):
    def __init__(self, host, user=None, password=None, common_headers=None, basic_auth=None):
        self.host = host
        self.base_url = f'http://{host}'
        self.common_params = {}
        # Foscam wants the user and password in the query params, not just before the host name.
        if isinstance(user, str):
            self.common_params['user'] = user
        if isinstance(password, str):
            self.common_params['password'] = password
        self.common_headers = {}
        if common_headers:
            self.common_headers.update(common_headers)
        if basic_auth:
            self.common_headers['Authorization'] = 'Basic ' + basic_auth
        # self.basic_auth = basic_auth
        # self.headers = {'Authorization': 'Basic ' + basic_auth}

    def get_endpoint(self, endpoint, params=None, headers=None, **kwargs):
        if params:
            tmp = self.common_params.copy()
            tmp.update(params)
            params = tmp
        else:
            params = self.common_params

        if headers:
            tmp = self.common_headers.copy()
            tmp.update(headers)
            headers = tmp
        else:
            headers = self.common_headers

        url = self.base_url
        if endpoint:
            if endpoint.startswith('/'):
                url = url + endpoint
            else:
                url = url + '/' + endpoint

        print(f'Getting {url}')
        response = requests.get(url, params=params, headers=headers, **kwargs)
        print(f'Requested {response.request.url}')
        print('REQUEST headers=')
        print(response.request.headers)
        print()
        print(f'Got {url}; response headers=')
        print(response.headers)


    def get_snapshot(self):
        response = self.get_endpoint('/snapshot.cgi')
        return response

    def open_videostream(self, frames_per_second=1, hi_res=True):
        """Read jpeg stream from videostream.cgi."""
        fps, code = videostream_fps_to_code(frames_per_second)
        if not math.isclose(fps, frames_per_second):
            print(f'Requesting {fps} f/s, instead of {frames_per_second} f/s')
        params = dict(
            rate=code,
            resolution=(32 if hi_res else 8)
        )
        response = self.get_endpoint('/videostream.cgi', params=params, stream=True)
        return response


if __name__ == '__main__':
    default_host=os.environ.get('FOSCAM_HOST', '')
    # print(f'default_host={default_host}')
    # print(f'os.environ={os.environ}')

    parser = argparse.ArgumentParser(
        description='Test access to Foscam.')
    parser.add_argument(
        '--host', required=(not default_host),
        default=default_host,
        help="Internet host name of the Foscam IP Camera. Defaults to $FOCSCAM_HOST")
    parser.add_argument('--basic_auth',
        default=os.environ.get('FOSCAM_BASIC_AUTH', ''),
             help='Basic Auth token.')
    parser.add_argument('--user',
        default=os.environ.get('FOSCAM_USER', ''),
             help='Foscam user name.')
    parser.add_argument('--password',
        default=os.environ.get('FOSCAM_PASSWORD', ''),
             help='Foscam password.')
    args = parser.parse_args()

    def arg_error(msg):
        print(msg, file=sys.stderr)
        parser.print_help()
        sys.exit(1)

    if not args.user:
        args.user = None
        args.password = None
    elif not args.password:
        args.password = ''

    print(f'args={args}')

    foscam = Foscam(args.host, user=args.user,
        password=args.password,
        basic_auth=args.basic_auth)

    response = foscam.open_videostream()

    for line in response.iter_lines(chunk_size=64):
        

