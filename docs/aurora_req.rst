Requirements
============


Python requirements
-------------------

Aurora uses the latest Python-3 distribution and requires a modern Fortran compiler, available on most Unix systems. Additionally, the following packages are automatically installed (from PyPI) when installing Aurora:

  numpy scipy matplotlib xarray

Aurora ships the small helper modules that it uses to interface with tokamak modeling tools inside the `aurora` package itself, so users do not need to install any separate compatibility package.





Julia requirements
------------------

To run the Julia version of the code, Julia must be installed; see::

  https://julialang.org/downloads/

Everything else should be automatically handled by the Aurora installation (see :ref:`Installation`).
