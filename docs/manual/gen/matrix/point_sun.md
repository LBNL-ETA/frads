# gen matrix point-sun
---
```
gen matrix point-sun <point_file> <model_files...> [options]
```
`gen matrix point-sun` generate a point to sun matrix. 
This matrix is usually the direct-sun part of the 
five-phase simulation. It takes a point file as the first
input, followed by the rest of the model files.

## Options
`-b/--basis`
:	Sun basis (default: r6)

## Example

Generate a point to sun matrix (e.g., direct-sun matrix)
```
gen matrix point-sun grid.pts obj/materials.mat obj/room.rad
```
