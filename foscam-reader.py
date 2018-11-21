#!/usr/bin/env python3.6
# Support for interacting with a Foscam IP camera.
# See https://www.foscam.es/descarga/ipcam_cgi_sdk.pdf
# for documentation on the different endpoints.

import argparse
import codecs
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
        return response

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


#class SimpleMultipartReaderResponse(object):

class SimpleMultipartReader(object):
    def __init__(self, response):
        self.response = response
        self.chunk = b''
        self.cursor = 0

    def available(self):
        return len(self.chunk) - self.cursor

    def find_delimiter(self, delimiter):
        return self.chunk.find(delimiter, self.cursor) - self.cursor

    def _append_next_chunk(self, chunk_size=None):
        # Want just the next chunk for now. May later want to iterate over many.
        loops = 0
        for new_chunk in self.response.iter_content(chunk_size=chunk_size, decode_unicode=False):
            # new_chunk is a sequence of bytes.
            if len(new_chunk) == 0:
                # DANGER of looping indefinitely!
                loops += 1
                if loops > 10:
                    raise ValueError('Too many empty new_chunks!')
                continue
            # If less is available in chunk than is already consumed
            # AND is to be appended, then don't keep appending to self.chunk.
            if self.chunk:
                if self.cursor > (self.available() + len(new_chunk)):
                    self.chunk = self.chunk[self.cursor:] + new_chunk
                    self.cursor = 0
                else:
                    self.chunk += new_chunk
            else:
                self.chunk = new_chunk
                self.cursor = 0
            return True
        # Unable to get another chunk from the underlying response stream.
        return False

    def next_line(self, chunk_size=1024, decode_unicode=False, max_valid_length=1024):
        while True:
            index = self.find_delimiter(b'\r\n')
            if index >= 0:
                # Found it.
                break
            if self.available() >= max_valid_length:
                raise ValueError(
                    f'There is no line terminator in the next {self.available()} bytes')
            # Need more data.
            if not self._append_next_chunk(chunk_size=chunk_size):
                # Unable to get more!
                raise ValueError(
                    'There is no line terminator before the end of the stream')

        if index > max_valid_length:
            raise ValueError(
                f'The next line terminator is too far away ({index} > {max_valid_length})')

        line_bytes = self.chunk[self.cursor:self.cursor+index]
        self.cursor += index + 2

        if decode_unicode:
            return codecs.decode(line_bytes, encoding=self.response.encoding, errors='replace')
        return line_bytes

    def _find_delimiter(self, delimiter):
        return self.chunk.find(delimiter, self.cursor)
        if index < 0:
            # Not found.
            return False


        if index - self.cursor > max_valid_length:
            raise ValueError(
                f'There is no line terminator in the next {max_valid_length} characters')



        index = self.chunk.find('\r\n', self.cursor)
        if index < 0:
            # Not found.
            return False

        if index - self.cursor > max_valid_length:
            raise ValueError(
                f'There is no line terminator in the next {max_valid_length} characters')






    def read_next_part(self, chunk_size=1024):
        headers = {}
        body = ''

        # Start by reading lines terminated by \r\n.




        while self._append_next_chunk(chunk_size=chunk_size):



        in

        while True:
            chunk = None
            if self.previous_chunk:
                chunk = self.previous_chunk
                self.previous_chunk = None
            else:
                # Want just the next chunk
                for chunk in self.response.iter_content():
                    break
            if 











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
        print(f'{line!r}')


