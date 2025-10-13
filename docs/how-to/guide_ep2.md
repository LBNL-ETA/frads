# How to create a glazing system?

This guide will show you how to create a glazing system (complex fenestration system) using the `create_glazing_system()` function. The glazing system can be added to the EnergyPlus model's `construction_complex_fenestration_state` object. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more details.

The `GlazingSystem` class contains information about the glazing system's solar absorptance, solar and visible transmittance and reflectance, and etc. The solar and photopic results are calcuated using [pyWincalc](https://github.com/LBNL-ETA/pyWinCalc).

Call `create_glazing_system()` to create a glazing system. The function takes in the name of the glazing system, a list of `LayerInput` objects, and an optional list of `Gap` objects. The function returns a `GlazingSystem` instance.

## LayerInput Object

The `LayerInput` class is used to define each layer in a glazing system. It allows you to specify:

* `input_source`: Path to the glazing/shading product file (JSON or XML format), or bytes data
* `flipped`: Boolean to flip the layer orientation (default: False)
* `slat_angle_deg`: Slat angle in degrees for venetian blinds (default: 0.0)
* `openings`: OpeningDefinitions object to define the top, bottom, left,right, and front openings of the layer (default: top:0, bottom:0, left:0, right:0, and front:0.05)

The glazing and shading product files can be downloaded from the [IGSDB](https://igsdb.lbl.gov/) website. The downloaded glazing product files are in JSON format and the shading product files are in XML format. The product files contain information about the product's transmittance, reflectance, and etc.

!!! note
    The list of `LayerInput` objects should be in order from exterior to interior.

!!! note
    The glazing system created by using `create_glazing_system()` has a default air gap at 0.0127 m thickness if no gaps are specified.

## Import the required classes and functions

```python
import frads as fr
```

## Example 1 Two layers of glazing products with default gap

**Double clear glazing system**

The glazing system consists of the following:

* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of clear glass

```python
gs = fr.create_glazing_system(
    name="double_clear",
    layer_inputs=[
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("igsdb_product_364.json"), # clear glass
    ],
)
```

## Example 2 Two layers of glazing products with custom gap

The `gaps` argument takes in a list of `Gap` objects. Each `Gap` object consists of a list of `Gas` objects and a float defining the gap thickness. The `Gas` object consists of the gas type and the gas fraction. The gas fraction is a float between 0 and 1. The sum of all gas fractions should be 1.

**Electrochromic glazing system**

The glazing system consists of the following:

* 1 layer of electrochromic glass
* 1 gap (10% air and 90% argon) at 0.0127 m thickness
* 1 layer of clear glass

```python
gs = fr.create_glazing_system(
    name="ec",
    layer_inputs=[
        fr.LayerInput("igsdb_product_7405.json"), # electrochromic glass
        fr.LayerInput("igsdb_product_364.json"), # clear glass
    ], # (1)
    gaps=[
        fr.Gap(
            gas=[fr.Gas("air", 0.1), fr.Gas("argon", 0.9)],
            thickness_m=0.0127
        )
    ],
)
```

1.  The list of `LayerInput` objects should be in order from exterior to interior.

## Example 3 Glazing with fabric shade

**Double glazing with exterior fabric shade**

The glazing system consists of the following:

* 1 layer of fabric shade (exterior)
* Gap: default air gap at 0.0127 m thickness
* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of clear glass

```python
gs = fr.create_glazing_system(
    name="double_clear_fabric",
    layer_inputs=[
        fr.LayerInput("fabric_shade.xml"), # fabric shade
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("igsdb_product_364.json"), # clear glass
    ],
)
```

## Example 4 Glazing with venetian blinds

**Double glazing with interior venetian blinds at 45-degree slat angle**

The glazing system consists of the following:

* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of venetian blinds at 45-degree slat angle

```python
gs = fr.create_glazing_system(
    name="double_clear_blinds",
    layer_inputs=[
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("venetian_blinds.json", slat_angle_deg=45), # venetian blinds
    ],
)
```

!!! note
    The `slat_angle_deg` parameter allows you to specify the slat angle for venetian blinds. The angle is measured from the horizontal plane at 0 degree. Positive slate angle means the slats are downward towards outside.

## Example 5 Fabric shade with openings

**Double glazing with fabric shade that has openings on all sides**

The glazing system consists of the following:

* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of clear glass
* Gap: default air gap at 0.0127 m thickness
* 1 layer of fabric shade with openings

```python
# Define openings for the shade layer
openings = fr.OpeningDefinitions(
    left_m=0.01,      # 10 mm opening on left side
    right_m=0.005,    # 5 mm opening on right side
    top_m=0.0025,     # 2.5 mm opening on top
    bottom_m=0.005,   # 5 mm opening on bottom
    front_multiplier=0.05  # 5% opening on front surface
)

gs = fr.create_glazing_system(
    name="double_clear_shade_openings",
    layer_inputs=[
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("igsdb_product_364.json"), # clear glass
        fr.LayerInput("fabric_shade.xml", openings=openings), # fabric with openings
    ],
)
```

!!! note
    The `OpeningDefinitions` object allows you to specify openings around the perimeter of shading layers. The opening dimensions are in meters. The `front_multiplier` specifies the fraction of the front surface that is open (0.0 to 1.0).


