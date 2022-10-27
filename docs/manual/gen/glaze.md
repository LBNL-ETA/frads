# gen glaze
---
```
gen glaze [-h] [-v] <command> [options]
```
`gen glaze` command can be used to generate a Radiance material definition 
from either a [IGSDB](https://igsdb.lbl.gov) JSON file or an LBNL Optics file. `glaze` can be used to
generate a single or a double-pane glazing system.

## Options

`-x/--optics`
:	LBNL Optics file path[s]. Either optics files or IGSDB files (see below) is required.

`-d/--igsdb`
:	IGSDB JSON file path[s]. Either optics files (see above) or IGSDB files are required.

Optional arguments:

`-c/--cspace`:

Color space (color primaries) (default: radiance)

`-s/--observer`:

CIE Observer 2° or 10° (default: 2)

## Example

Generate a single-pane system using a IGSDB JSON file:
```
gen glaze -d igsdb_product_23000.json
	void BRTDfunc VNE-63_on_Pure_Mid_Iron
	10
		sr_clear_r sr_clear_g sr_clear_b 
		st_clear_r st_clear_g st_clear_b 
		0 0 0 glaze1.cal
	0
	19
		0 0 0 
		0 0 0 
		0 0 0 -1 
		0.094 0.07 0.03 
		0.101 0.043 0.021 
		1.102 0.593 0.246
```
To generate a double-pane glazing system using two JSON files downloaded 
from [IGSDB](https://igsdb.lbl.gov), using the default color primaries and
CIE observer,
```
gen glaze -d 82378.json 43203.json
	void BRTDfunc VNE-63_on_Pure_Mid_Iron+VE-2M_on_Pure_Mid_Iron
	10 
		if(Rdot,cr(fr(0.062),ft(1.310),fr(0.101)),cr(fr(0.094),ft(1.102),fr(0.070))) 
		if(Rdot,cr(fr(0.035),ft(0.662),fr(0.043)),cr(fr(0.070),ft(0.593),fr(0.041))) 
		if(Rdot,cr(fr(0.016),ft(0.277),fr(0.021)),cr(fr(0.030),ft(0.246),fr(0.017))) 
		ft(1.310)*ft(1.102) 
		ft(0.662)*ft(0.593) 
		ft(0.277)*ft(0.246) 
		0 0 0 glaze2.cal
	0
	9
		0 0 0
		0 0 0
		0 0 0

```
