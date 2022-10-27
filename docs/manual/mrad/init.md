# mrad init
---
```
mrad init {-a wea_path | -e epw_path} [-n model_name]
	[-o object [object ...]] [-m material [material ...]]
	[-w window [window ...]] [-x xmls [xmls ...]]
	[-g surface_path spacing height]
```
The `init` command can be used to initiate a mrad configuration file. `init` command provide a series of option to select files and define the model setup. Either a `.wea` or `.epw` file path need to be provided for `init` to run (described below). You will also need to define where are objects and material files. Although `init` doesn't require specifying grid of view files, you will need to define at least one of these, by manually typing in the configuration file, in order for `mrad` to carry out the actual simulation.

## Options

`-a/--wea_path` 
:	WEA file path, which can be generated using `epw2wea` program, 
or generated using script. Some weather file site also provide 
wea file, such as [Climate.OneBuilding.org](https://climate.onebuilding.org). 
This option and `-e/--epw_path` option are mutually exclusive, 
and one of which is required by the `init` command.

`-e/--epw_path`
:	EnergyPlusWeather (EPW) file path. 
Can found at [EnergyPlus website](https://energyplus.net/weather) and 
[Climate.OneBuilding.org](https://climate.onebuilding.org). 
This option and `-a/--wea_path` option are mutually exclusive, 
and one of which is required by the `init` command.

`-n/--name`
:	Give your model a name. This name will be part of the file name of the results files.
If you don't give one, the name of the configuration file will be used.

`-o/--object`
:	Object file path[s], usually with a .rad extension. 
You can use wildcard file matching, such as `-o Objects/*.rad`, which will load 
in all files that has a `.rad` file extension within the `Objects` directory.

`-m/--material`
:	material file path[s], usually with a .mat extension. 
You can use wildcard file matching, such as `-o Objects/*.mat`, which will load in 
all files that has a `.mat` file extension within the `Objects` directory. 
This implies that you will need to separate out your material definitions 
from the rest of your geometry definitions. In other words, 
these files should only contain material definitions.

`-w/--window`
:	Window file path[s], usually with a .rad extension. You can use wildcard, such 
as `-o Objects/window*.rad`, which will load in all files that starts with the 
word `window` and has a `.rad` file extension within the `Objects` directory. 
This implies that if you intent to have special treatments for different windows, 
e.g., swap different Window BSDF to compare performance, you will need to 
separate out the windows geometry definition from the rest of the model. 
If you intent to have different window groups, they need to also be grouped 
by files.

`-x/--xmls`
:	BSDf file path[s], usually with a .xml extension. You can use wildcard, such as `-o Resources/*.xml`, which will load in all files that has a `.xml` file extension within the `Objects` directory. You will need to define `.xml` files with a one-to-one mapping to the window files described above. This implies that the window geometry within a window file will have the same BSDF treatment as defined with this option.

`-g/--grid`
:	This option takes in three values: grid surface file path, grid spacing, and grid height. 
Grid surface file path is the file that contains a surface from which the grid will be 
constructed from. In most cases, this is the floor. Grid spacing and height are in the 
same unit that the model is using. 
A example us of this option can be: `-g Objects/floor.rad 2 2.5`, which will define 
a 2x2 grid based on the `floor.rad` and 2.5 unit distance away from `floor.rad`.


## Example

Here is an `init` example specifying an .epw file, material, objects and window files.

```
mrad init -e ./Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.epw -m Objects/material.mat 
	-o Objects/wall.rad Objects/ceiling.rad Objects/floor.rad 
	-w Objects/upper_glass.rad Objects/lower_glass.rad
```

A `default.cfg` file is generated as shown below. 
Notice that SimControl and RaySender sectons are empty.
You can configure [SimControl](#SimControl) as needed. 
Note that you will need to specify something in the RaySender
section for simulation to run. 

```
[SimControl]

[Site]
epw_path = Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.epw

[Model]
name = default
scene = Objects/ceiling.rad
	Objects/walls.rad
	Objects/floor.rad
material = Objects/materials.mat
window_paths = Objects/upper_glass.rad
	Objects/lower_glass.rad

[RaySender]
```
