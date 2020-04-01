#!/usr/bin/env python3
"""
Download EPW file given lat and lon or zipcode if in US

T.Wang
"""

import argparse
import csv
from contextlib import closing
import os
from requests import get
from requests.exceptions import RequestException


class getEPW(object):
    """Download the closest EPW file from the given Lat and Lon."""
    _file_path_ = os.path.dirname(os.path.realpath(__file__))
    epw_url = "epw_url.csv"
    zip2latlon = "zip_latlon.txt"
    epw_url_path = os.path.join(_file_path_, 'data', epw_url)
    assert os.path.isfile(epw_url_path), 'File not found: {}'.format(epw_url_path)
    zip2latlon_path = os.path.join(_file_path_, 'data', zip2latlon)
    assert os.path.isfile(zip2latlon_path),\
            'File not found: {}'.format(zip2latlon_path)

    def __init__(self, lat, lon):
        self.lat = float(lat)
        self.lon = float(lon)
        distances = []
        urls = []
        with open(self.epw_url_path, 'r') as rdr:
            csvreader = csv.DictReader(rdr, delimiter=',')
            for row in csvreader:
                distances.append((float(row['Latitude']) - self.lat)**2
                                + (float(row['Longitude']) - self.lon)**2)
                urls.append(row['URL'])
        min_idx = distances.index(min(distances))
        url = urls[min_idx]
        epw_fname = os.path.basename(url)
        try:
            with closing(get(url, allow_redirects=False, stream=True)) as resp:
                raw = resp.content
        except RequestException as e:
            raise e
        with open(epw_fname, 'wb') as wtr:
            wtr.write(raw)
        self.fname = epw_fname

    @classmethod
    def from_zip(cls, zipcode):
        zipcode = str(zipcode)
        with open(cls.zip2latlon_path, 'r') as rdr:
            csvreader = csv.DictReader(rdr, delimiter='\t')
            for row in csvreader:
                if row['GEOID'] == zipcode:
                    lat = row['INTPTLAT']
                    lon = row['INTPTLONG']
                    break
            else:
                raise 'zipcode not found in US'
        return cls(lat, lon)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', help='Latitude')
    parser.add_argument('-o', help='Longitude')
    parser.add_argument('-z', help='Zipcode (US only)')
    args = parser.parse_args()
    if args.z is None:
        assert None not in [args.a, args.o], "Need Lat and Long"
        getEPW(args.a, args.o)
    else:
        getEPW.from_zip(args.z)
