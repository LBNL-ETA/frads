frads library
======================
.. currentmodule:: frads

The frads library consists of Python modules needed by the command-line programs to complete the task; e.g., mtxmethod for computing sDA, ASE, and DGP. You can get the library directly from PyPI::

    pip install frads

.. _radmtx:

radmtx
------
   radmtx

.. _mtxmethod:

mtxmethod
----------------
   mtxmethod

.. _radutil:

radutil
--------
   radutil

.. _radgeom:

radgeom
---------
   radgeom

ep2rad
------
One can use the library by importing needed functions to facilitate a specific workflow::

   import epjson2rad
   radobj = epjson2rad.epJSON2Rad(epsj_path)
