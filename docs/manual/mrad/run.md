# mrad run
---

```
mrad run [cfg_path]
```

Once we have a viable configuration file, we can use the `run` command to start the simulation.
Once started, `mrad` will generate a `Matrices` and a `Results` directory at the current working
directory. All of the essential matrix files will be stored inside the `Matrices` folders.
Intermediate matrix files will be stored inside the system `temp` directory and removed after
the simulation. All results will be saved to the `Results` directory.

The output with `-vv` verbosity setting is similar to the following:
```
...
21-04-20 22:10:44 - frads.methods - INFO - Converting EPW to a .wea file
21-04-20 22:10:44 - frads.methods - Generating sku/sun matrix using command
21-04-20 22:10:44 - frads.methods - gendaymtx -of -m 4 /Users/taoning/Resources/USA_CA_Oakland.Intl.AP.724930_TMY3.wea
21-04-20 22:10:46 - frads.cli - INFO - Using two-phase method
21-04-20 22:10:46 - frads.methods - INFO - Computing for 2-phase sensor point matrices...
21-04-20 22:10:46 - frads.methods - INFO - Computing for image-based 2-phase matrices...
21-04-20 22:10:46 - frads.methods - INFO - Computing for 2-phase sensor grid results.
21-04-20 22:10:50 - frads.methods - INFO - Computing for 2-phase image-based results
```
