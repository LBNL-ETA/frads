# gen matrix ncp
---
```
gen matrix ncp <window_file> <ncp_file> <model_files...> [options]
```
`gen matrix ncp` genertate a matrix (or BSDF) that describeds
the flux transport behavior of a non-coplanar shading system. 
The sender in this command is usually a window surface and 
the receiver is automatically generated that encompasses the 
non-coplanar shading system. The resulting matrix can be wrapped 
into a xml file and be used as a regular BSDF file.

## Options

`-b/--basis`
:	Window and receiving basis. (default: ['kf', 'kf'])

`-w/--wrap`
:	Generating a final xml file? (default: False)

## Example

Generate a non-coplanar shading matrix (e.g. a drop-arm awning system), and
wrap the matrix into a .xml file.
```
$ gen matrix ncp window.rad awning.rad obj/materials.mat obj/room.rad -w
```
