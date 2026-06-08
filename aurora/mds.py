import numpy as np


class MDSValue:
    def __init__(self, server, treename, shot, TDI, quiet=False):
        self.server = server
        self.treename = treename
        self.shot = shot
        self.TDI = TDI
        self.quiet = quiet
        self._connection = None

    def _get_connection(self):
        if self._connection is not None:
            return self._connection

        try:
            import MDSplus
        except ImportError as exc:
            raise ImportError("MDSplus is required for MDSValue") from exc

        connection = MDSplus.Connection(self.server)
        connection.openTree(self.treename, int(self.shot))
        self._connection = connection
        return connection

    def _get(self, expr):
        value = self._get_connection().get(expr)
        try:
            data = value.data()
        except Exception:
            data = value

        if isinstance(data, np.ndarray) and data.ndim == 0:
            return data.item()
        return data

    def data(self):
        return self._get(self.TDI)

    def dim_of(self, axis=0):
        return self._get(f"dim_of({self.TDI},{axis})")

    def units(self):
        return self._get(f"units_of({self.TDI})")

    def units_dim_of(self, axis=0):
        return self._get(f"units_of(dim_of({self.TDI},{axis}))")

    def check(self):
        try:
            return self.data() is not None
        except Exception:
            return False
