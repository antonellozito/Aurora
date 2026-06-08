import netCDF4
import numpy as np


class NCFile(dict):
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.variables = self
        self.load()

    def load(self):
        self.clear()

        with netCDF4.Dataset(self.filename) as dataset:
            self["__format__"] = dataset.file_format
            self["__dimensions__"] = {
                name: len(dim) for name, dim in dataset.dimensions.items()
            }

            globals_dict = {}
            for attr in dataset.ncattrs():
                globals_dict[str(attr)] = dataset.getncattr(attr)
            if globals_dict:
                self["_globals"] = globals_dict

            for name, variable in dataset.variables.items():
                data = variable[...]
                if isinstance(data, np.ma.MaskedArray):
                    data = data.filled(np.nan)
                if getattr(data, "dtype", None) is not None and data.dtype.kind in {"S", "U"}:
                    try:
                        data = netCDF4.chartostring(data)
                    except Exception:
                        pass
                if np.ndim(data) == 0 and hasattr(data, "item"):
                    data = data.item()

                entry = {
                    "data": data,
                    "__dimensions__": tuple(map(str, variable.dimensions)),
                    "__dtype__": variable.dtype,
                    "__varid__": variable._varid,
                }
                for attr in variable.ncattrs():
                    entry[str(attr)] = variable.getncattr(attr)
                self[str(name)] = entry
