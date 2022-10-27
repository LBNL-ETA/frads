# gen grid
```
gen grid [-h] [-v] <surface> <spacing> <height> [options]
```
The `grid` command can be used to generate an equal-distance sensor grid.
There are three required inputs to this command: `surface`, `spacing`, `height`, 
and a grid of sensor will be generate on the surface normal side, pointing
in the same direction as the surface normal. 

## Options

`-op`
:	Where the generate the sensor on the opposite side of the polygon.

## Example
To generate a grid of sensor based on the polygon defined in the `floor.rad`
file, with a spacing of 2 units and 3 units away from the surface.
```
gen grid floor.rad 2 3
	1 1.7 -2 0 0 -1
	1 3.7 -2 0 0 -1
	1 5.7 -2 0 0 -1
	1 7.7 -2 0 0 -1
	1 9.7 -2 0 0 -1
	1 11.7 -2 0 0 -1
	1 13.7 -2 0 0 -1
	1 15.7 -2 0 0 -1
	...
```
