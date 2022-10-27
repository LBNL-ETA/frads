# gencolorsky
---
```
gencolorsky <year> <month> <day> <hour> <minute> [options]
```
Gencolorsky uses [libRadtran](http://www.libradtran.org/) to compute spatially- and spectrally-resolved sky radiation data based on an Earth’s spherical atmosphere radiative transfer model that includes Rayleigh scattering by air molecules, molecular absorption, aerosol, water, and ice clouds. One of the main command-line programs in libRadtran, uvspec, is invoked to compute the sky radiance at every r° (default 3°) in both the azimuth and altitude directions.  Within uvspec, the DISORT radiative transfer solver is used.  Extraterrestrial solar source data (280-800 nm), at 1nm interval, are used to generate spectrally-resolved sky radiation data at each sample point, which by default is computed at a 10 nm interval from 360 nm to 800 nm. By default, this sky spectral data is converted to Radiance RGB using CIE XYZ tristimulus with either the 2° or 10° standard observer, using user defined color space (default: Radiance RGB). If -i setting is used, the resulting three channels become photopic (CIE-Y), melanopic equivalent daylight il/luminance (EDI), and solar ir/radiance, respectively. The output is a folder called cs_{datetime}_{lat}_{lon} in the current working directory, containing a sky.rad file along with associated color data. The options to gencolorsky are can be found using `-h` flag on the command line.

## Options

`-a/--latitude`
:	Location latitude; positive is northern hemisphere (required)

`-o/--longitude`
:	Location longitude; positive is western (required)

`-m/--tzone`:

:	Location standard meridian (required)

`-u/--altitude`:

:	Location altitude in km, default = 0

`-i/--pmt`:

:	Compute for photopic, melanopic, and solar for each of the three channels, instead of RGB.

`-r/--anglestep`:

:	Angular resolution at which the sky is sampled, default = 3°

`-s/--observer`:

:	Standard observer, 2° or 10°

`-c/--colorspace`:

:	Colorspace from which the sky is derived, choices are {radiance, sharp, adobe, rimm, 709, p3, 2020}, default=Radiance

`-e/--atm`:

:	Atmospheric composition file. Default to use AFGL data that came with libRadtran, automatically chosen depending on location and time of year.

`-l/--aerosol`:

:	Standard aerosol profile. This option overwrites the aerosol optical depth setting,defined below. The profile choices are: `{Continental_clean | Continental_average | Continental_polluted | Urban | Maritime_clean | Maritime_polluted | Maritime_tropical | Desert | Antarctic }`.

`-b/--cloudcover`:

:	Cloud cover, [0, 1], 1 is complete cover. Cloud cover data can be sourced from TMY data.

`-d/--aod`:

Aerosol optical depth, which can be sourced from TMY data.

`-g/--cloudprofile`:

:	Cloud profile file path. Space separated file, three columns: height (km), liquid water content (LWC) (g/m3), effective radius (R_eff) (um). Default: water clouds exist from 2-5 km in altitude with a LWC of 2.5 g/m3 and R_eff of 100 um

`-t/--total`:

:	Compute GHI, DNI, and DHI, instead of full sky description. Handy for quick comparison against measurements.

`-v/--verbose`:

:	Verbosity -v=Debug, -vv=Info, -vvv=Warning, -vvvv=Error, -vvvvv=Critical, default=Warning

## Examples

Compute direct normal, diffuse horizontal, and global horizontal irradiance in solar spectrum for 2022-06-21 10:00 in Berkeley CA USA, with clear sky and continental_average aerosol profile:

```
gencolorsky 2022 6 21 10 0 -a 37.7 -o 122.2 -m 120 -l continental_average -t
	DNI: 864.40 W/m2; DHI: 124.47 W/m2; GHI: 862.96 W/m2
```

Compute RGB sky for 2022-12-21 16:30 in Berkeley, CA, USA, with .2 cloud cover and .3 aerosol optical depth:

```
gencolorsky 2022 12 21 16 30 -a 37.7 -o 122.2 -m 120 -b .2 -d .3
```

Compute photopic, melanopic, and solar for each of the three-channel for the same sky:

```
gencolorsky 2022 12 21 16 30 -a 37.7 -o 122.2 -m 120 -b .2 -d .3 -i
```
