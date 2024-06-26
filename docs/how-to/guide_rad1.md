# How to set up a simple rtrace workflow.

Here we go through the process of setting up a simple Radiance
model and the workflow of computing irradiance values.


## Prepare a model

If you already have a Radiance model setup, you can skip this step
and follow along using your own.

If you don't have a model already we can use `gen room` to get ourself
a simple Radiance model.

Let's generate a open-office sized side-lit room
with four same-sized windows. The room will be 12 meters wide, 14 meters
deep, a floor to floor height of 4 meters, and a ceiling height of 3 meters. Each window is 2.5 meters in width
and 1.8 meters in height and has a sill height of 1 meter. Windows are 0.4 meters
apart from each other. Finally, we want our facade to have a thickness
of 0.1 meters. We'll call this model 'aroom'. The `gen room` command is:

```python 
! gen room 12 14 4 3 \
	-w 0.4 1 2.5 1.8 \
	-w 3.3 1 2.5 1.8 \
	-w 6.2 1 2.5 1.8 \
	-w 9.1 1 2.5 1.8 \
	-t 0.1 -n aroom # (1)
```

1. `gen room` is a command line function. To run shell commands from inside a IPython syntax (e.g. Jupyter Notebook), start the code with an exclamation mark (!).

Afterwards, we will have a `Objects` folder in our current working
directory with all of our Radiance model inside.

```
|____Objects
| |____window_03_aroom.rad
| |____wall_aroom.rad
| |____window_02_aroom.rad
| |____ceiling_aroom.rad
| |____window_01_aroom.rad
| |____materials_aroom.mat
| |____window_00_aroom.rad
| |____floor_aroom.rad
```
We can quickly visualize our model using `objview` and
make sure it's what we'd expect in terms of layout and geometry.
```
objview Objects/*aroom.mat Objects/*aroom.rad
```
And we can see that it is what we'd expect.

![image](../assets/model.png){: style="height:343px;width:321px"}


## Generate an octree file

Now that we have model, we can start to run some actual simulation.
Each code block below can be copy and paste into a something like
a Jupyter Lab for an interactive workflow.

First lets import all the necessary modules.
```py
import datetime
import pyradiance as pr
import frads as fr
```

Next lets gather our model files. Notice that we have our
material files first in the list.

```py
fpaths = ["Objects/materials_aroom.mat",
          "Objects/ceiling_aroom.rad",
          "Objects/wall_aroom.rad",
          "Objects/floor_aroom.rad",
          "Objects/window_00_aroom.rad",
          "Objects/window_01_aroom.rad",
          "Objects/window_02_aroom.rad",
          "Objects/window_03_aroom.rad",
]
```

Now that we know where all the paths are, we can call `oconv`
to get ourself a `octree` for ray tracing. We'd like to save
our octree as a `aroom.oct` file.


```python
room_octree = "aroom.oct"
with open(room_octree, 'wb') as f:
    f.write(pr.oconv(*fpaths))
```

Notices that we have a `aroom.oct`, which only contains the geometry.
We need to define our light source, usually some kind of sky model,
for rays to trace to. In this example, we will use Perez all-weather
sky model. There are also standard CIE skies as alternatives.
To do so, we can use `gen_perez_sky` function to get our sky
description and generate a new octree with it.

First to get our sky description, we make up a clear sky on 12-21 12:00
with direct normal irradiance of 800 W/m2 and diffuse horizontal irradiance of 100 W/m2.
We also need to define our location in terms of latitude, longitude, time-zone, and
elevation.

```py
date_time = datetime.datetime(2024, 12, 21, 12, 0)
sky_descr = fr.gen_perez_sky(date_time, latitude=37, longitude=122, timezone=120, dirnorm=800, diffhor=100)
```
Once we have our sky description, we can combine it with our `aroom.oct` octree
to make a new octree file. Let's call the octree with our sky specific information,
'aroom_37_122_1221_1200.oct'.
```
room_sky_octree = f'aroom_37_122_1221_1200.oct'
with open(room_sky_octree, "wb") as f:
    f.write(pr.oconv(stdin=sky_descr, octree=room_octree))
```


## Get rays

We need send rays to the octree file we just created.
In Radiance, rays are made of two vectors, one for the starting
position and one for the direction the ray is heading.
Essentially, we need six values to define our two vectors in
cartesian corrdiantes. A ray positioned at x=0, y=0, z=0,
pointing upwards is thus:
```
0 0 0 0 0 1
```
For this example, we're gonna simulate workplane illuminance.
These are essentially virtual sensor positioned at table height
pointing upwards, measuring how much light arrives at your table.
To get a grid of such sensors, we can use `gen_grid` utility
function, which need a `polygon`, `spacing`, and `height` as
arguments. Spacing and height define the grid spacing and the distance
from the polygon from which the grid is based-one.
Since we are generating a grid of workplane sensors,
we can use our floor as the polygon. To get our floor polygon,
we can simply load in our `floor_aroom.rad` file and parse
the polygon using the `parse_polygon` function from the `parsers` module.
The code block demonstrates how we generate a grid of sensors with
1 meter spacing and 0.75 meters away from the floor:

```py
floor_primitives = fr.unpack_primitives("Objects/floor_aroom.rad")
# Since we only have one primitive in this file,
# we'll take the first one to parse.
floor_polygon = fr.parse_polygon(floor_primitives[0])
grid = fr.gen_grid(floor_polygon, 1, 0.75)
```

## Let's trace

Finally, after all these preparation, we are ready to trace some rays.
Let's first trace a single ray, and use the one of the grid sensors
we had just created.

```py
aray = " ".join(map(str, grid[0]))
option = ["-I+", "-ab", "1", "-ad", "64", "-aa", "0", "-lw", "0.01"]
result = pr.rtrace(aray.encode(), room_sky_octree, params=option)
```
if we print the `result`, we will see the following:
```
>>> print(result)
#?RADIANCE
...

3.23E+01 1.23E+02 7.80E+01
```

Next, let's trace all of our grid sensors. Since our grid of sensors are a list
of lists of floats, we need to process them a little bit before rtrace can take them.

```
rays = "\n".join([" ".join(map(str, row)) for row in grid])
results = pr.rtrace(rays.encode(), room_sky_octree, params=option, header=False)
```
And our results, now with the header, is the following:
```
```
After weighting these RGB values to obtain illuminance values, it'd be nice if we
can visualize it somehow. One of the most common approach to visualize illuminance
values over a grid of sensors is to map these values to a color scale.  There are many
ways to achieve. Here we're gonna stay in the Python world and use the popular
[matplotlib](https://matplotlib.org) library to visualize our results.
favorite

