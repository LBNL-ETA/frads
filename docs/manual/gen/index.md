# Gen
```
gen [-h] [-v] <command> [options]
```
The `gen` command-line program consists of a host of commands that do the actual work.
As the name suggests, `gen` generates things. Specifically, it can be
used to generate a sensor grid, glazing description, a side-lit room model, and various
types of matrices. Verbosity, the amount of information printed to your console, 
can be adjusted by using the `-v` option, where `-v=debug; -vv=info; -vvv=warning; -vvvv=critical`. 
By default, only warning information is displayed. Instead 
of display information onto the terminal, all logging information 
can also be redirected as standard error into a file.  Information regarding how to run mrad and its sub-command can be display on your terminal by giving `-h/--help` options.

```
gen -h
```
or
```
gen <command> -h
```
or
```
gen <command> <sub-command> -h
```


## Commands
- [glaze](glaze.md)
- [grid](grid.md)
- [matrix](matrix.md)
- [room](room.md)
