
# Installation

Before you can use frads, you need to install it.

## Install Python

Being a Python based library, you'll need to install Python first.
Python version **3.8** or newer is required for frads.

Get the latest version of Python at https://www.python.org/downloads/ or with your operating system’s package manager.

You can verify that Python is installed by typing python from your cmd/powershell/terminal; you should see something like:

	$ python
	Python 3.X.X
	[GCC 4.x] on linux
	Type "help", "copyright", "credits" or "license" for more information.
	>>>

After you have Python installed, you should have `pip` command available in your shell environment as well. You can then use `pip` to install `frads`:

## Install pyenergyplus
frads relies on pyenergyplus for running and interacting with EnergyPlus in Python.
pyenergyplus (unofficial) can be installed by running

    $ pip install git+https://github.com/taoning/pyenergyplus_wheels


## Install frads

After you have pyenergyplus installed, you can then use `pip` to install `frads`:

	$ python -m pip install frads

## Verifying

To verify that `frads` can be seen by Python, type `python` from your shell. Then at the Python prompt, try to import `frads`

	>>> import frads
	>>> print(frads.__version__)
	1.0.0
