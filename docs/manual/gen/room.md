# gen room
---
```
gen room [-h] [-v] <width> <depth> <floor-floor> <floor-ceiling> [options]
```
This command generates a side-lit rectangular room model with `width`, `depth`, 
`floor-floor`, and `floor-ceiling`. 

## Options

`-w`
:	Window[s], starting x, z, width, height.

`-n`
:	Model name

`-t`

:	Facade thickness.

`-r/--rotate`
:	Rotate the room counter-clockwise

## Example
Generate a room that's 12 unit wide, 14 unit deep and 3 unit height.
```
gen room 12 14 3
```
Genearte the same room, adding two windows, each 4 unit wide and 1.5 unit in height.
Viewing from outside, the lower left corner of the first window is 1 unit away from the left wall and
1 unit away from the floor.
```
gen room 12 14 3 -w 1 1 4 1.5 -w 5 1 4 1.5
```

