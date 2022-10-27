# gen matrix view-sun
---
```
gen matrix view-sun [-h] [-v] <view_file> <model_files...> [options]
```
`gen matrix view-sun` generate a view to sun matrix. This matrix is 
usually the direct-sun part of the five-phase simulation.

## Options

`-b/--basis`
:	Sun basis (default: r6)

`-w/--window`
:	Window file path[s]. (default: None)

`-s/--smx_path`
:	Sky matrix file path. (default: None)

`-r/--resolu`
:	Image resultion (default: 512x512)

## Example

Generate a view to sun matrix (e.g., image-based direct-sun matrix), with 
sun-culling based on window normals and oakland annual sky matrix.
```
gen matrix view-sun view1.vf obj/materials.mat obj/room.rad \
	-w window.rad -s oakland.smx
```

