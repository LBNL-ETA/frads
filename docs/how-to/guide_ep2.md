# How to create a glazing system?

This guide will show you how to create a glazing system (complex fenestration system) using the `GlazingSystem` class. The glazing system can be added to the EnergyPlus model's `construction_complex_fenestration_state` object. See [How to run a simple EnergyPlus simulation?](guide_ep1.md) for more details.

The `GlazingSystem` class contains information about the glazing system's solar absorptance, solar and visible transmittance and reflectance, and etc. The solar and photopic results are calcuated using [pyWincalc](https://github.com/LBNL-ETA/pyWinCalc).

Call `create_glazing_system()` to create a glazing system. The function takes in the name of the glazing system, a list of glazing/shading product files, and an optional list of gap layers. The function returns a `GlazingSystem` instance.

The glazing and shading product files can be downloaded from the [IGSDB](https://igsdb.lbl.gov/) website. The downloaded glazing product files are in JSON format and the shading product files are in XML format. The product files contain information about the product's transmittance, reflectance, and etc.

!!! note
    The list of glazing/shading product files should be in order from exterior to interior.

!!! note
    The glazing system created by using `create_glazing_system()` has a default air gap at 0.0127 m thickness.

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
    layers=[
        Path("igsdb_product_364.json"), # clear glass
        Path("igsdb_product_364.json"), # clear glass
    ],
)
```

## Example 2 Two layers of glazing products with custom gap

The `gaps` argument takes in a list of `Gap` objects. Each `Gap` object consists of a list of `Gas` objects and a float defining the gap thickness. The `Gas` object consists of the gas type and the gas fraction. The gas fraction is a float between 0 and 1. The sum of all gas fractions should be 1.

**Electrochromatic glazing system**

The glazing system consists of the following:

* 1 layer of electrochromic glass
* 1 gap (10% air and 90% argon) at 0.0127 m thickness
* 1 layer of clear glass

```python
gs = fr.create_glazing_system(
    name="ec",
    layers=[
        "igsdb_product_7405.json", # electrochromic glass
        "igsdb_product_364.json", # clear glass
    ], # (1)
    gaps=[
        fr.Gap(
            [fr.Gas("air", 0.1), fr.Gas("argon", 0.9)], 0.0127
        )
    ],
)
```

1.  The list of glazing/shading product files should be in order from exterior to interior.


