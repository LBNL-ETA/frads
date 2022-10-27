# mrad
---
```
mrad [-h] [-v] <command> [options]
```
You can complete an entire matrix-based simulation using the `mrad` program. In general there are two steps:

1. Configure your model in the form of a configuration file (.cfg)
2. Feed the configuration file to `mrad` for a completly automated simulation. 

A configuration file can be generated using the `init` subcommand (see blow). It also can be programmtically generated or even manually typed in. Details of the configuration files are described below.


## Commands

 - [mrad init](mrad_init.md)
 - [mrad run](mrad_run.md)

### Verbosity setting

Verbosity, the amount of information printed to your terminal, can be adjusted by using the `-v` option. `-v` will display the most detailed information and `-vvvv` is effectively the silent mode. By default, only warning information is displayed. Instead of display information onto the terminal, all logging information can also be redirected as standard error into a file.

```
-v = debug
-vv = info
-vvv = warning (default)
-vvvv = critical
```

### Display help message

Information regarding how to run mrad and its sub-command can be display on your terminal by giving `-h/--help` options.

```
$ mrad -h
```
or
```
$ mrad <command> -h
```


## Configuration

A configuration stores input data and file paths needed to carry out an entire simulation workflow.
A configuration file (.cfg) can be typed in manually, programatically generated, or generated using the init command from the project root directory. The easiest way to generate template config file is to use `mrad init` command. To start with, the `init` command needs minimum a weather file path or a epw file path. 

A configuration file consists of four sections: 
`SimControl`, `Site`, `Model`, and `RaySenders`.

### SimControl

These are the options availble under this section:

`vmx_basis`:
:	view matrix basis.

`vmx_opt`:
:	view matrix simulation options.

`dmx_opt`:
:	view matrix basis.

`fmx_basis`:
:	Facade matrix basis, basis used for generate non-coplanar matrices

`smx_basis`:
:	Sky matrix basis, which defines how fine to discretize the sky. Usually r1 or r4 for Tregenza or Reinhart Sky with 4 subdivisions.

`dsmx_opt`:
:	two-phase method matrix option.

`cdsmx_opt`:
:	Direct-sun coefficient matrices options.

`ray_count`:
:	Number of rays per sample.


`separate_direct`:
:	Whether to do a separate direct sun calculation. Turn this on to use five-phase method.


`nprocess`:
:	Number of processors to use. This only works on Linux and MacOS.

`overwrite`:
:	Whether to overwrite existing a matrices files.

### Site

`wea_path`:
:	wea file path

`epw_path`:
:	epw file path

`start_hour`:
:	Filter out hours in the weather files before this hour.

`end_hour`:
:	Filter out hours in the weather files after this hour.

`daylight_hours_only`:
:	Filter out non-daylight ours based on frads intern solar angle calculation.

`orientation`:
:	Set the model orientation.


### Model

`material`:
:	Material file paths. These files only contain material definitions

`scene`:
:	Object file paths. These files can include window files. If so, we will do two-phase simulation.

`windows`:
:	Window file paths. Each file contains a window group.

`window_xml`:

`window_cfs`:


### RaySenders

`grid_surface`:
:	surface geometry file (usually .rad) containing the surface polygon from which the grid will be based on.

`grid_spacing`:
:	Grid spacing in the model unit.

`grid_height`: 
:	Grid height from the grid surface in the model unit.

`view`:
:	A view for rendering. This can be view describtion string or a view file defined as `-vf view_file.vf`.


### Default configuration

The default configuration setting (`mrad_default.cfg`) can be found inside the data directory.
This default setting is loaded with both `init` and `run` command.
So any options that is not set with the user defined configuration file will be set 
according to the default configuration.

The default configurations are:

```
[DEFAULT]
vmx_basis = kf
vmx_opt = -ab 6 -ad 4096 -lw 0.0001
fmx_basis = kf
smx_basis = r4
dmx_opt = -ab 2 -ad 128 -c 5000
dsmx_opt = -ab 8 -ad 4096 -lw 0.0001 -lr 8
cdsmx_opt = -ab 1 -dj 0 -st 0
cdsmx_basis = r6
ray_count = 1
nprocess = 1
separate_direct = False
overwrite = False
start_hour = 0
end_hour = 0
daylight_hours_only = True
overwrite = True
```
