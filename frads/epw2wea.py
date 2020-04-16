#!/usr/bin/env python
"""
Modified version of epw2wea.

Input arguments:
1. epwfile path: a positional mandatory argument specifying the file path of the epw file
2. weafile path: a positional mandatory argument following the epwfile path argument
defining the file path/name of the output weafile
3. --daylight_hour_only (-s): optional flag that can be placed anywhere to ask for
outputing only daylight hours in the wea file.
Daylight hours are calculated based on a simplified/translated version of the Radiance
gensky & sun.c module -- daylight hours are defined as when solar altitude is > 0
4/5 start and end hour (-shr/-ehr) range to include
To generate a regular Radiance sky matrix, run: gendaymtx {.wea} > {.smx}
To generate a single sky vector for each entry in the wea file, use included wea2sky.py
Created by Taoning Wang

"""

import argparse
from math import pi, sin, cos, asin


class epw2wea(object):
    """."""

    def __init__(self, *, epw, dh, sh, eh):
        """."""
        self.epw = epw
        #self.wea = wea
        self.read_epw()  # read-in epw/tmy data

        if sh is not None:
            self.sh = sh
            self.s_hour()

        if eh is not None:
            self.eh = eh
            self.e_hour()

        if dh:
            self.daylight()  # filter out non-daylight hours if asked

        self.wea = self.header + self.string
        self.dt_string = []
        for line in self.string.splitlines():
            entry = line.split()
            mo = int(entry[0])
            da = int(entry[1])
            hr = int(float(entry[2]))
            self.dt_string.append(f"{mo:02d}{da:02d}_{hr:02d}30")


    def solar_angle(self, month, day, hour):
        """Simplified translation from the Radiance sun.c and gensky.c code.

        This function test if the solar altitude is greater than zero
        """
        mo_da = [0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334]

        julian_date = mo_da[month - 1] + day

        solar_decline = 0.4093 * sin((2 * pi / 368) * (julian_date - 81))

        solar_time = hour + (0.170 * sin((4 * pi / 373) * (julian_date - 80))
                             - 0.129 * sin((2 * pi / 355) * (julian_date - 8))
                             + 12 * (self.s_meridian - self.longitude_r) / pi)

        altitude = asin(sin(self.latitude_r) * sin(solar_decline)
                        - cos(self.latitude_r) * cos(solar_decline)
                        * cos(solar_time * (pi / 12)))

        return altitude > 0

    def daylight(self):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line
                      if self.solar_angle(int(li.split()[0]),
                                          int(li.split()[1]),
                                          float(li.split()[2]))]
        self.string = "\n".join(new_string)

    def s_hour(self):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line if float(
            li.split()[2]) >= self.sh]
        self.string = "\n".join(new_string)

    def e_hour(self):
        """."""
        string_line = self.string.splitlines()
        new_string = [li for li in string_line if float(
            li.split()[2]) <= self.eh]
        self.string = "\n".join(new_string)

    def read_epw(self):
        """."""
        with open(self.epw, 'r') as epw:
            raw = epw.readlines()  # read-in epw content
        epw_header = raw[0].split(',')
        content = raw[8:]
        string = ""
        for li in content:
            line = li.split(',')
            month = int(line[1])
            day = int(line[2])
            hour = int(line[3]) - 1
            hours = hour + 0.5
            dir_norm = float(line[14])
            dif_hor = float(line[15])
            string += "%d %d %2.3f %.1f %.1f\n" \
                % (month, day, hours, dir_norm, dif_hor)
        self.string = string
        city = epw_header[1]
        country = epw_header[3]
        self.latitude = float(epw_header[6])
        self.latitude_r = self.latitude * pi / 180
        self.longitude = -1 * float(epw_header[7])
        self.longitude_r = self.longitude * pi / 180
        self.tz = int(float(epw_header[8])) * (-15)
        self.s_meridian = self.tz * pi / 180
        elevation = epw_header[9].rstrip()
        self.header = "place {}_{}\n".format(city, country)
        self.header += "latitude {}\n".format(self.latitude)
        self.header += "longitude {}\n".format(self.longitude)
        self.header += "time_zone {}\n".format(self.tz)
        self.header += "site_elevation {}\n".format(elevation)
        self.header += "weather_data_file_units 1\n"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modified version of epw2wea with added\
     capability of including only the daylight hours")
    parser.add_argument('-s', '--daylight_hour_only', action="store_true",
                        help='output only for daylight hours', required=False)
    parser.add_argument('-shr', '--start_hour', type=float,
                        help='start hour (float)')
    parser.add_argument('-ehr', '--end_hour', type=float,
                        help='end hour (float)')
    parser.add_argument('epwfile', type=str, help="epw file path")
    parser.add_argument('weafile', type=str, help="output wea file path")
    args = parser.parse_args()

    epw_fpath = args.epwfile
    wea_fpath = args.weafile
    s_hour = args.start_hour
    e_hour = args.end_hour

    if args.daylight_hour_only:
        DayHour = True
        print("Writing only daylight hours ...")
    else:
        DayHour = False

    wea = epw2wea(epw_fpath, wea_fpath, dh=DayHour, sh=s_hour, eh=e_hour)
    print(wea.wea)
