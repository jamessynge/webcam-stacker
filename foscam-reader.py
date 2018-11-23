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


class MultipartReader(object):
    def __init__(self, response):
        content_type = response.headers.get('Content-Type', '')
        prefix = 'multipart/x-mixed-replace;boundary='
        if not content_type.startswith(prefix):
            raise ValueError(f'Wrong Content-Type: {content_type}')
        self.response = response
        self.boundary_text = '--' + content_type[len(prefix):]
        self.next_boundary_bytes = codecs.encode(
            '\r\n--' + content_type[len(prefix):], encoding='ascii')
        self.chunk = b''
        self.cursor = 0

    def available(self):
        return len(self.chunk) - self.cursor

    def find_delimiter(self, delimiter, search_from=0):
        search_from += self.cursor
        return self.chunk.find(delimiter, search_from) - self.cursor

    def _append_next_chunk(self, chunk_size=1024):
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

    def _append_until(self, stop_cb, chunk_size=1024):
        empty_loops = 0
        for new_chunk in self.response.iter_content(chunk_size=chunk_size, decode_unicode=False):
            # new_chunk is a sequence of bytes.
            if len(new_chunk) == 0:
                # DANGER of looping indefinitely!
                empty_loops += 1
                if empty_loops > 10:
                    ValueError('Read too many empty chunks in a row!')
                continue
            empty_loops = 0
            if self.available() > 0:
                # If less is available in chunk than is already consumed
                # AND is to be appended, then don't keep appending to self.chunk.
                if self.cursor > (self.available() + len(new_chunk)):
                    self.chunk = self.chunk[self.cursor:] + new_chunk
                    self.cursor = 0
                else:
                    self.chunk += new_chunk
            else:
                self.chunk = new_chunk
                self.cursor = 0
            if stop_cb():
                return True
        # Unable to get another chunk from the underlying response stream.
        return False

    def next_line(self, chunk_size=1024, encoding='ascii', errors='replace', max_valid_length=1024):
        search_from = 0
        while True:
            index = self.find_delimiter(b'\r\n', search_from=search_from)
            if index >= 0:
                # Found it.
                break
            # Resume searching where we left off, rather than at the beginning.
            search_from = max(0, self.available() - 2)
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

        line = self.chunk[self.cursor:self.cursor+index]
        print(f'raw line {line!r}')
        self.cursor += index + 2

        if encoding:
            line = codecs.decode(line, encoding=encoding, errors='replace')
        print(f'next_line -> {line!r}')
        return line

    def read_next_part(self, chunk_size=1024):
        headers = requests.structures.CaseInsensitiveDict()
        body = b''

        # Read the boundary, skipping over blank lines first.
        while True:
            line = self.next_line()
            if line:
                break
        assert line == self.boundary_text, f'{line} != {self.boundary_text}'

        # Now read non-blank lines as HTTP headers.
        while True:
            line = self.next_line()
            if not line:
                break

            # Not handling multi-line header values (i.e. where the
            # value continues on to the next line).
            print(f'line to split {line!r}')
            name, value = line.split(':', maxsplit=1)
            name = name.strip()
            value = value.strip()
            headers[name] = value


        print('Located these part headers:')
        print(headers)

        if 'Content-Length' in headers:
            content_length = int(headers['Content-Length'])
            if content_length > (10 * 1024 * 1024):
                raise ValueError(f'Content-Length ({content_length}) is too large')
            need = content_length + len(self.next_boundary_bytes)
            if not self._append_until(lambda: self.available() >= content_length):
                raise ValueError(
                    f'Unable to read body of Content-Length {content_length}')
            next_boundary = self.find_delimiter(self.next_boundary_bytes)
            if next_boundary < 0:
                # Not found (yet), so not a problem to return content_length bytes.
                pass
            elif next_boundary < content_length:
                # Not enough content!
                raise ValueError(f'Only found {next_boundary} bytes, expected {content_length - next_boundary} more bytes.')
            body = self.chunk[self.cursor:self.cursor+content_length]
            self.cursor += content_length
        else:
            raise Exception('Not Yet Implemented')

        return headers, body


        # while self._append_next_chunk(chunk_size=chunk_size):



        # in

        # while True:
        #     chunk = None
        #     if self.previous_chunk:
        #         chunk = self.previous_chunk
        #         self.previous_chunk = None
        #     else:
        #         # Want just the next chunk
        #         for chunk in self.response.iter_content():
        #             break
        #     if 











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
    mpr = MultipartReader(response)
    while True:
        headers, body = mpr.read_next_part()

    # for line in response.iter_lines(chunk_size=64):
    #     print(f'{line!r}')


