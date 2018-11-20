#!/usr/bin/env python3.6
# Support for interacting with a Foscam IP camera.
# See https://www.foscam.es/descarga/ipcam_cgi_sdk.pdf
# for documentation on the different endpoints.

import requests

class Foscam(object):
    def __init__(self, host, user=None, password=None, common_headers=None, basic_auth=None):
        self.host = host
        self.base_url = 'http://%s'
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
        return requests.get(url, params=params, headers=headers)

    def get_snapshot(self):
        response = self.get_endpoint('/snapshot.cgi')
        print('Got /snapshot.cgi; response headers=')
        print(response.headers)
        pass

    def stream_video(self, cb):
        # Read jpeg stream from the /videostream.cgi endpoint of a Foscam IP Camera.
        pass

