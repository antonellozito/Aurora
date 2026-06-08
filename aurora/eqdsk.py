import copy
import re
from pathlib import Path

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.path import Path as MplPath
from scipy import interpolate, integrate

from .mds import MDSValue


_GEQDSK_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+\.\d*|\.\d+|\d+)(?:[EeDd][+-]?\d+)?"
)


def _polygon_area_centroid(r, z):
    r = np.asarray(r)
    z = np.asarray(z)
    if r[0] != r[-1] or z[0] != z[-1]:
        r = np.append(r, r[0])
        z = np.append(z, z[0])

    cross = r[:-1] * z[1:] - r[1:] * z[:-1]
    area = 0.5 * np.sum(cross)
    if np.isclose(area, 0.0):
        return 0.0, float(np.mean(r)), float(np.mean(z))

    factor = 1.0 / (6.0 * area)
    r_centroid = factor * np.sum((r[:-1] + r[1:]) * cross)
    z_centroid = factor * np.sum((z[:-1] + z[1:]) * cross)
    return abs(area), float(r_centroid), float(z_centroid)


def _midplane_intersection(r, z, z0):
    r = np.asarray(r)
    z = np.asarray(z)
    if r[0] != r[-1] or z[0] != z[-1]:
        r = np.append(r, r[0])
        z = np.append(z, z[0])

    intersections = []
    for r0, r1, z_start, z_end in zip(r[:-1], r[1:], z[:-1], z[1:]):
        dz0 = z_start - z0
        dz1 = z_end - z0
        if dz0 == 0:
            intersections.append(r0)
        if dz0 * dz1 < 0 and z_end != z_start:
            frac = (z0 - z_start) / (z_end - z_start)
            intersections.append(r0 + frac * (r1 - r0))
    if intersections:
        return float(np.max(intersections))
    return float(np.max(r))


class FluxSurfaceData(dict):
    def __init__(self, geqdsk, levels, flux_list, geo, midplane):
        super().__init__()
        self._geqdsk = geqdsk
        self["levels"] = np.asarray(levels)
        self["flux"] = flux_list
        self["geo"] = geo
        self["midplane"] = midplane
        self["R0"] = float(geqdsk["RMAXIS"])
        self["Z0"] = float(geqdsk["ZMAXIS"])
        self["BCENTR"] = float(geqdsk["BCENTR"])
        self["CURRENT"] = float(geqdsk["CURRENT"])

        aux = geqdsk["AuxQuantities"]
        self._bp_interp = interpolate.RegularGridInterpolator(
            (aux["Z"], aux["R"]),
            aux["Bp"],
            bounds_error=False,
            fill_value=np.nan,
        )

    def surfAvg(self, function):
        values = []
        for surface in self["flux"]:
            r = np.asarray(surface["R"])
            z = np.asarray(surface["Z"])

            if r.size <= 1:
                sampled = np.asarray(function(r, z))
                values.append(float(np.ravel(sampled)[0]))
                continue

            if r[0] != r[-1] or z[0] != z[-1]:
                r = np.append(r, r[0])
                z = np.append(z, z[0])

            dl = np.hypot(np.diff(r), np.diff(z))
            rc = 0.5 * (r[:-1] + r[1:])
            zc = 0.5 * (z[:-1] + z[1:])
            bp = self._bp_interp(np.column_stack([zc, rc]))
            bp = np.where(np.isfinite(bp) & (np.abs(bp) > 1e-12), np.abs(bp), 1.0)

            sampled = np.asarray(function(rc, zc))
            if sampled.shape != rc.shape:
                sampled = np.broadcast_to(sampled, rc.shape)

            weight = dl / bp
            values.append(float(np.sum(sampled * weight) / np.sum(weight)))

        return np.asarray(values)

    def volume_integral(self, values):
        values = np.asarray(values, dtype=float)
        vol = np.asarray(self["geo"]["vol"], dtype=float)
        if values.shape != vol.shape:
            raise ValueError("Input to volume_integral must match the flux-surface grid")
        return integrate.cumulative_trapezoid(values, vol, initial=0.0)


class GEQDSK(dict):
    def __init__(self, filename="", **_kwargs):
        super().__init__()
        self.filename = filename if isinstance(filename, str) else ""

        if isinstance(filename, dict):
            self.update(copy.deepcopy(filename))
            if "AuxQuantities" not in self and "PSIRZ" in self:
                self._postprocess()
        elif filename and Path(filename).exists():
            self.load(filename)

    def load(self, filename=None):
        if filename is not None:
            self.filename = filename

        with open(self.filename, "r", encoding="utf-8", errors="ignore") as stream:
            lines = stream.read().splitlines()

        def splitter(value, step=16):
            return [value[step * idx : step * (idx + 1)] for idx in range(len(value) // step)]

        def merge(block):
            if not block:
                return ""
            if len(block[0]) > 80:
                return ("".join(block)).replace(" ", "")
            return "".join(block)

        header = lines[0]
        self["CASE"] = header[:48].rstrip()
        try:
            tmp = [item for item in header[48:].split(" ") if item]
            _idum, self["NW"], self["NH"] = map(int, tmp[:3])
        except ValueError:
            self["NW"] = int(header[52:56])
            self["NH"] = int(header[56:60])

        offset = 1
        (
            self["RDIM"],
            self["ZDIM"],
            self["RCENTR"],
            self["RLEFT"],
            self["ZMID"],
            self["RMAXIS"],
            self["ZMAXIS"],
            self["SIMAG"],
            self["SIBRY"],
            self["BCENTR"],
            self["CURRENT"],
            self["SIMAG"],
            _xdum,
            self["RMAXIS"],
            _xdum,
            self["ZMAXIS"],
            _xdum,
            self["SIBRY"],
            _xdum,
            _xdum,
        ) = map(float, splitter(merge(lines[offset : offset + 4])))
        offset += 4

        nw = self["NW"]
        nh = self["NH"]
        nl_nw = int(np.ceil(nw / 5.0))

        self["FPOL"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        offset += nl_nw
        self["PRES"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        offset += nl_nw
        self["FFPRIM"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        offset += nl_nw
        self["PPRIME"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        offset += nl_nw

        nl_nwnh = int(np.ceil(nw * nh / 5.0))
        try:
            psirz = np.fromiter(splitter(merge(lines[offset : offset + nl_nwnh])), dtype=np.float64)
            self["PSIRZ"] = psirz[: nh * nw].reshape(nh, nw)
            offset += nl_nwnh
        except ValueError:
            nl_nwnh = nh * nl_nw
            psirz = np.fromiter(splitter(merge(lines[offset : offset + nl_nwnh])), dtype=np.float64)
            self["PSIRZ"] = psirz[: nh * nw].reshape(nh, nw)
            offset += nl_nwnh

        self["QPSI"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        offset += nl_nw

        if len(lines) > offset:
            self["NBBBS"], self["LIMITR"] = map(int, [item for item in lines[offset].split(" ") if item][:2])
            offset += 1

            nl_nbbbs = int(np.ceil(self["NBBBS"] * 2 / 5.0))
            boundary = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nbbbs])))))
            self["RBBBS"] = boundary[0::2][: self["NBBBS"]]
            self["ZBBBS"] = boundary[1::2][: self["NBBBS"]]
            offset += max(nl_nbbbs, 1)

            try:
                nl_limitr = int(np.ceil(self["LIMITR"] * 2 / 5.0))
                limiter = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_limitr])))))
                self["RLIM"] = limiter[0::2][: self["LIMITR"]]
                self["ZLIM"] = limiter[1::2][: self["LIMITR"]]
                offset += nl_limitr
            except ValueError:
                self["LIMITR"] = 5
                dd = self["RDIM"] / 10.0
                r_box = np.linspace(0, self["RDIM"], 2) + self["RLEFT"]
                z_box = np.linspace(0, self["ZDIM"], 2) - self["ZDIM"] / 2.0 + self["ZMID"]
                self["RLIM"] = np.array(
                    [
                        max(r_box[0], np.min(self["RBBBS"]) - dd),
                        min(r_box[1], np.max(self["RBBBS"]) + dd),
                        min(r_box[1], np.max(self["RBBBS"]) + dd),
                        max(r_box[0], np.min(self["RBBBS"]) - dd),
                        max(r_box[0], np.min(self["RBBBS"]) - dd),
                    ]
                )
                self["ZLIM"] = np.array(
                    [
                        max(z_box[0], np.min(self["ZBBBS"]) - dd),
                        max(z_box[0], np.min(self["ZBBBS"]) - dd),
                        min(z_box[1], np.max(self["ZBBBS"]) + dd),
                        min(z_box[1], np.max(self["ZBBBS"]) + dd),
                        max(z_box[0], np.min(self["ZBBBS"]) - dd),
                    ]
                )
        else:
            self["NBBBS"] = 0
            self["LIMITR"] = 0
            self["RBBBS"] = np.array([])
            self["ZBBBS"] = np.array([])
            self["RLIM"] = np.array([])
            self["ZLIM"] = np.array([])

        try:
            self["KVTOR"], self["RVTOR"], self["NMASS"] = map(
                float, [item for item in lines[offset].split(" ") if item]
            )
            offset += 1
            if self["KVTOR"] > 0:
                offset += nl_nw
                offset += nl_nw
            if self["NMASS"] > 0:
                offset += nl_nw
            self["RHOVN"] = np.asarray(list(map(float, splitter(merge(lines[offset : offset + nl_nw])))))
        except Exception:
            pass

        if "RHOVN" not in self or not len(self["RHOVN"]) or not np.sum(self["RHOVN"]):
            self.add_rhovn()

        self._postprocess()
        return self

    def add_rhovn(self):
        psin = np.linspace(0.0, 1.0, len(self["QPSI"]))
        qpsi = np.asarray(self["QPSI"], dtype=float)
        tor_flux = integrate.cumulative_trapezoid(qpsi, psin, initial=0.0)
        norm = tor_flux[-1] - tor_flux[0]
        if np.isclose(norm, 0.0):
            self["RHOVN"] = np.sqrt(psin)
        else:
            self["RHOVN"] = np.sqrt(np.clip((tor_flux - tor_flux[0]) / norm, 0.0, None))
        return self["RHOVN"]

    def _postprocess(self):
        self.add_rhovn()
        self._build_aux_quantities()
        self._build_flux_surfaces()

    def _build_aux_quantities(self):
        r = np.linspace(self["RLEFT"], self["RLEFT"] + self["RDIM"], self["NW"])
        z = np.linspace(
            self["ZMID"] - self["ZDIM"] / 2.0,
            self["ZMID"] + self["ZDIM"] / 2.0,
            self["NH"],
        )
        rr, _zz = np.meshgrid(r, z)

        psi = np.asarray(self["PSIRZ"], dtype=float)
        denom = self["SIBRY"] - self["SIMAG"]
        if np.isclose(denom, 0.0):
            denom = 1.0
        psin = (psi - self["SIMAG"]) / denom
        psin_axis = interpolate.RegularGridInterpolator(
            (z, r), psin, bounds_error=False, fill_value=None
        )([[self["ZMAXIS"], self["RMAXIS"]]])[0]
        psin = (psin - psin_axis) / max(1.0 - psin_axis, 1e-12)
        rhop_rz = np.sqrt(np.clip(psin, 0.0, None))

        dpsi_dz, dpsi_dr = np.gradient(psi, z[1] - z[0], r[1] - r[0])
        br = dpsi_dz / rr
        bz = -dpsi_dr / rr
        bp = np.hypot(br, bz)

        fp_interp = interpolate.interp1d(
            np.linspace(0.0, 1.0, len(self["FPOL"])),
            np.asarray(self["FPOL"], dtype=float),
            bounds_error=False,
            fill_value=(self["FPOL"][0], self["FPOL"][-1]),
        )
        fpol_rz = fp_interp(np.clip(psin, 0.0, 1.0))
        bt = fpol_rz / rr

        self["AuxQuantities"] = {
            "R": r,
            "Z": z,
            "PSIRZ": psi,
            "PSIRZ_NORM": psin,
            "RHOpRZ": rhop_rz,
            "RHOp": np.sqrt(np.linspace(0.0, 1.0, len(self["QPSI"]))),
            "Br": br,
            "Bz": bz,
            "Bp": bp,
            "Bt": bt,
        }

    def _build_flux_surfaces(self):
        aux = self["AuxQuantities"]
        r = aux["R"]
        z = aux["Z"]
        psin_grid = aux["PSIRZ_NORM"]
        levels = np.linspace(0.0, 1.0, len(self["QPSI"]))

        fig = Figure()
        FigureCanvasAgg(fig)
        ax = fig.add_subplot(111)
        contour = ax.contour(r, z, psin_grid, levels=levels[1:-1])

        traced = {}
        for level, segs in zip(contour.levels, contour.allsegs):
            candidates = []
            for seg in segs:
                if len(seg) < 8:
                    continue
                path = MplPath(seg)
                contains_axis = path.contains_point((self["RMAXIS"], self["ZMAXIS"]))
                area, r_centroid, z_centroid = _polygon_area_centroid(seg[:, 0], seg[:, 1])
                if contains_axis:
                    candidates.append((area, seg, r_centroid, z_centroid))
            if not candidates:
                for seg in segs:
                    if len(seg) < 8:
                        continue
                    area, r_centroid, z_centroid = _polygon_area_centroid(seg[:, 0], seg[:, 1])
                    candidates.append((area, seg, r_centroid, z_centroid))
            if candidates:
                traced[level] = max(candidates, key=lambda item: item[0])[1]

        flux_list = [{"R": np.array([self["RMAXIS"]]), "Z": np.array([self["ZMAXIS"]])}]
        geo_psin = [0.0]
        geo_vol = [0.0]
        geo_rhon = [0.0]
        geo_rvol = [0.0]
        midplane_r = [float(self["RMAXIS"])]

        for level in levels[1:]:
            nearest = None
            if traced:
                nearest = traced.get(level)
                if nearest is None:
                    nearest_key = min(traced, key=lambda key: abs(key - level))
                    nearest = traced[nearest_key]
            if nearest is None:
                flux_list.append({"R": np.array([self["RMAXIS"]]), "Z": np.array([self["ZMAXIS"]])})
                geo_psin.append(float(level))
                geo_vol.append(geo_vol[-1])
                geo_rhon.append(geo_rhon[-1])
                geo_rvol.append(geo_rvol[-1])
                midplane_r.append(midplane_r[-1])
                continue

            rs = np.asarray(nearest[:, 0], dtype=float)
            zs = np.asarray(nearest[:, 1], dtype=float)
            area, r_centroid, _z_centroid = _polygon_area_centroid(rs, zs)
            volume = 2.0 * np.pi * area * r_centroid
            rmid = _midplane_intersection(rs, zs, self["ZMAXIS"])

            flux_list.append(
                {
                    "R": rs,
                    "Z": zs,
                    "psi": self["SIMAG"] + level * (self["SIBRY"] - self["SIMAG"]),
                    "q": float(
                        np.interp(level, np.linspace(0.0, 1.0, len(self["QPSI"])), self["QPSI"])
                    ),
                }
            )
            geo_psin.append(float(level))
            geo_vol.append(float(volume))
            midplane_r.append(float(rmid))

        if geo_vol[-1] > 0.0:
            edge_r = max(midplane_r[-1] - self["RMAXIS"], 1e-12)
            geo_rhon = [max((value - self["RMAXIS"]) / edge_r, 0.0) for value in midplane_r]
            geo_rvol = [
                np.sqrt(max(value, 0.0) / (2.0 * np.pi**2 * self["RMAXIS"]))
                for value in geo_vol
            ]
        else:
            geo_rhon = np.zeros(len(midplane_r)).tolist()
            geo_rvol = np.zeros(len(geo_vol)).tolist()

        self["fluxSurfaces"] = FluxSurfaceData(
            self,
            levels=np.asarray(geo_psin),
            flux_list=flux_list,
            geo={
                "psin": np.asarray(geo_psin),
                "vol": np.asarray(geo_vol),
                "rhon": np.asarray(geo_rhon),
                "rvol": np.asarray(geo_rvol),
            },
            midplane={
                "R": np.asarray(midplane_r),
                "Z": np.full(len(midplane_r), float(self["ZMAXIS"])),
            },
        )

    def from_aug_sfutils(self, shot=None, time=None, eq_shotfile="EQI", ed=1):
        if shot is None:
            raise ValueError("Must specify shot")
        if time is None:
            raise ValueError("Must specify time")

        try:
            import aug_sfutils as sf
        except ImportError as exc:
            raise ImportError("aug_sfutils does not seem to be installed") from exc

        eqm = sf.EQU(shot, diag=eq_shotfile, ed=ed)
        geq = sf.to_geqdsk(eqm, t_in=time)
        geq["PSIRZ"] = np.asarray(geq["PSIRZ"]).T
        geq["LIMITR"] = len(geq["RLIM"])
        geq["NBBBS"] = len(geq["ZBBBS"])

        self.clear()
        self.update(geq)
        self.filename = self.filename or ""
        self._postprocess()
        return self

    def from_mdsplus(
        self,
        device=None,
        shot=None,
        time=None,
        exact=False,
        SNAPfile="EFIT01",
        time_diff_warning_threshold=10,
        fail_if_out_of_range=True,
        show_missing_data_warnings=None,
        quiet=False,
    ):
        del show_missing_data_warnings
        del quiet

        if device is None:
            raise ValueError("Must specify device")
        if shot is None:
            raise ValueError("Must specify shot")
        if time is None:
            raise ValueError("Must specify time")

        device_key = str(device).upper().replace("-", "")
        device_formats = {
            "DIIID": "A",
            "NSTX": "B",
            "NSTXU": "B",
            "CMOD": "C",
            "EAST": "B",
            "EAST_US": "B",
            "KSTAR": "A",
            "ST40": "B",
        }
        fmt = device_formats.get(device_key, "A")
        field = "TOP.EFIT.RESULTS" if device_key == "CMOD" else "TOP.RESULTS"
        tree_format = {
            "A": rf"\{SNAPfile}::TOP.RESULTS.{{letter}}EQDSK.{{signal}}",
            "B": rf"\{SNAPfile}::TOP.RESULTS.{{letter}}EQDSK.{{signal}}",
            "C": rf"\{SNAPfile}::{field}.{{letter}}_EQDSK.{{signal}}",
        }[fmt]
        transpose = fmt == "C"

        psirz_signal = tree_format.format(letter="G", signal="PSIRZ")
        psirz_obj = MDSValue(server=device, treename=SNAPfile, shot=shot, TDI=psirz_signal)
        time_vec = np.asarray(psirz_obj.dim_of(2 if transpose else 0), dtype=float)
        units = psirz_obj.units_dim_of(2 if transpose else 0)
        if isinstance(units, bytes):
            units = units.decode()
        if isinstance(units, str) and units.strip().lower() in {"s", "sec", "seconds"}:
            time_vec = time_vec * 1e3

        ind = int(np.argmin(np.abs(time_vec - time)))
        diff = float(abs(time_vec[ind] - time))
        if exact and diff > 0:
            raise ValueError("Could not find the requested exact EFIT time slice")
        if diff > time_diff_warning_threshold and fail_if_out_of_range:
            raise ValueError("Closest EFIT time slice is outside the allowed threshold")

        signal_map = {
            "RZERO": "RCENTR",
            "MW": "NW",
            "MH": "NH",
            "XDIM": "RDIM",
            "ZDIM": "ZDIM",
            "RMAXIS": "RMAXIS",
            "ZMAXIS": "ZMAXIS",
            "SSIMAG": "SIMAG",
            "SSIBRY": "SIBRY",
            "BCENTR": "BCENTR",
            "CPASMA": "CURRENT",
            "FPOL": "FPOL",
            "PRES": "PRES",
            "FFPRIM": "FFPRIM",
            "PPRIME": "PPRIME",
            "PSIRZ": "PSIRZ",
            "QPSI": "QPSI",
            "NBBBS": "NBBBS",
            "LIMITR": "LIMITR",
            "RBBBS": "RBBBS",
            "ZBBBS": "ZBBBS",
            "RLIM": "RLIM",
            "ZLIM": "ZLIM",
            "RHOVN": "RHOVN",
            "CASE": "CASE",
            "ECASE": "CASE",
            "R": "_R",
            "Z": "_Z",
            "ZMID": "ZMID",
        }

        gathered = {}
        for source, target in signal_map.items():
            letter = "G"
            tdi = tree_format.format(letter=letter, signal=source)
            try:
                data = MDSValue(server=device, treename=SNAPfile, shot=shot, TDI=tdi).data()
            except Exception:
                continue

            arr = np.asarray(data)
            if transpose and arr.ndim == 3:
                arr = arr.T

            if arr.ndim >= 1 and arr.shape[-1] == len(time_vec):
                arr = np.take(arr, ind, axis=-1)
            elif arr.ndim >= 1 and arr.shape[0] == len(time_vec):
                arr = np.take(arr, ind, axis=0)

            if np.ndim(arr) == 0 and hasattr(arr, "item"):
                arr = arr.item()

            gathered[target] = arr

        self.clear()
        self.update(gathered)
        if "_R" in self:
            self["RLEFT"] = float(np.asarray(self["_R"])[0])
            self["RDIM"] = float(np.asarray(self["_R"])[-1] - np.asarray(self["_R"])[0])
        if "_Z" in self:
            z_arr = np.asarray(self["_Z"])
            self["ZMID"] = float(0.5 * (z_arr[0] + z_arr[-1]))
            self["ZDIM"] = float(z_arr[-1] - z_arr[0])
        self.pop("_R", None)
        self.pop("_Z", None)

        self["NW"] = int(self["NW"])
        self["NH"] = int(self["NH"])
        self["NBBBS"] = int(np.ravel(self["NBBBS"])[0])
        self["LIMITR"] = int(np.ravel(self["LIMITR"])[0])
        self["PSIRZ"] = np.asarray(self["PSIRZ"])
        if self["PSIRZ"].shape != (self["NH"], self["NW"]):
            self["PSIRZ"] = np.asarray(self["PSIRZ"]).reshape(self["NH"], self["NW"])
        self["FPOL"] = np.asarray(self["FPOL"])
        self["PRES"] = np.asarray(self["PRES"])
        self["FFPRIM"] = np.asarray(self["FFPRIM"])
        self["PPRIME"] = np.asarray(self["PPRIME"])
        self["QPSI"] = np.asarray(self["QPSI"])
        self["RBBBS"] = np.asarray(self["RBBBS"]).reshape(-1)[: self["NBBBS"]]
        self["ZBBBS"] = np.asarray(self["ZBBBS"]).reshape(-1)[: self["NBBBS"]]
        self["RLIM"] = np.asarray(self.get("RLIM", [])).reshape(-1)[: self["LIMITR"]]
        self["ZLIM"] = np.asarray(self.get("ZLIM", [])).reshape(-1)[: self["LIMITR"]]
        self._postprocess()
        return self
