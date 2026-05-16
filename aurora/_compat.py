import numpy as np
import scipy.integrate as _scipy_integrate
import scipy.interpolate as _scipy_interpolate


def _install_errstate_compat():
    _orig_errstate = np.errstate

    try:
        _ctx = _orig_errstate(all="ignore")
        with _ctx:
            pass
        with _ctx:
            pass
        return
    except TypeError:
        pass

    class _ReusableErrState:
        def __init__(self, **kwargs):
            self._kwargs = kwargs
            self._ctx = None

        def __enter__(self):
            self._ctx = _orig_errstate(**self._kwargs)
            return self._ctx.__enter__()

        def __exit__(self, exc_type, exc, tb):
            try:
                return self._ctx.__exit__(exc_type, exc, tb)
            finally:
                self._ctx = None

        def __call__(self, func):
            def _wrapped(*args, **kwargs):
                with self:
                    return func(*args, **kwargs)

            _wrapped.__name__ = getattr(func, "__name__", "_wrapped")
            _wrapped.__doc__ = getattr(func, "__doc__", None)
            _wrapped.__module__ = getattr(func, "__module__", None)
            return _wrapped

    def _errstate_factory(**kwargs):
        return _ReusableErrState(**kwargs)

    np.errstate = _errstate_factory


def _install_interp2d_compat():
    try:
        _scipy_interpolate.interp2d([0.0, 1.0], [0.0, 1.0], np.zeros((2, 2)))
        return
    except NotImplementedError:
        pass

    class _Interp2DCompat:
        def __init__(
            self,
            x,
            y,
            z,
            kind="linear",
            copy=True,
            bounds_error=False,
            fill_value=None,
        ):
            self.x = np.asarray(x, dtype=float)
            self.y = np.asarray(y, dtype=float)
            self.z = np.asarray(z, dtype=float)

            if self.z.shape != (len(self.y), len(self.x)):
                raise ValueError("interp2d compatibility expects z.shape == (len(y), len(x))")

            self.kind = kind
            self.bounds_error = bounds_error
            self.fill_value = np.nan if fill_value is None else fill_value

            if kind in ("linear", "nearest"):
                self._interp = _scipy_interpolate.RegularGridInterpolator(
                    (self.y, self.x),
                    self.z,
                    method=kind,
                    bounds_error=bounds_error,
                    fill_value=self.fill_value,
                )
                self._use_spline = False
            elif kind in ("cubic", "quintic"):
                order = {"cubic": 3, "quintic": 5}[kind]
                order = max(1, min(order, len(self.x) - 1, len(self.y) - 1))
                self._interp = _scipy_interpolate.RectBivariateSpline(
                    self.y, self.x, self.z, kx=order, ky=order
                )
                self._use_spline = True
            else:
                raise NotImplementedError(
                    f"interp2d compatibility does not support kind={kind!r}"
                )

        def __call__(self, x_new, y_new):
            x_new = np.atleast_1d(np.asarray(x_new, dtype=float))
            y_new = np.atleast_1d(np.asarray(y_new, dtype=float))

            if self._use_spline:
                out = self._interp(y_new, x_new)
                outside = (
                    (x_new[None, :] < self.x.min())
                    | (x_new[None, :] > self.x.max())
                    | (y_new[:, None] < self.y.min())
                    | (y_new[:, None] > self.y.max())
                )
                if self.bounds_error and np.any(outside):
                    raise ValueError("One of the requested xi is out of bounds in interp2d")
                if np.any(outside):
                    out = np.asarray(out)
                    out[outside] = self.fill_value
            else:
                xx, yy = np.meshgrid(x_new, y_new)
                pts = np.column_stack((yy.ravel(), xx.ravel()))
                out = self._interp(pts).reshape(len(y_new), len(x_new))

            if out.shape == (1, 1):
                return np.array([out[0, 0]])
            return out

    _scipy_interpolate.interp2d = _Interp2DCompat


def install_runtime_compat():
    if not hasattr(np, "NaN"):
        np.NaN = np.nan

    if not hasattr(np, "RankWarning"):
        np.RankWarning = np.exceptions.RankWarning

    if not hasattr(np, "trapz"):
        np.trapz = np.trapezoid

    _install_interp2d_compat()
    _install_errstate_compat()

    if not hasattr(_scipy_integrate, "cumtrapz"):
        _scipy_integrate.cumtrapz = _scipy_integrate.cumulative_trapezoid
