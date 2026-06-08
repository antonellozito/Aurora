import ast
import copy
import re

import numpy as np


def tolist(data, empty_lists=None):
    if isinstance(data, str):
        data = [data]

    if empty_lists:
        if not np.iterable(data):
            if data in empty_lists:
                return []
        else:
            data = [item for item in data if item not in empty_lists]

    if isinstance(data, np.ndarray):
        return np.atleast_1d(data).tolist()
    if isinstance(data, dict):
        return [data]

    try:
        return list(data)
    except TypeError:
        return [data]


_ROMAN_NUMERAL_MAP = (
    ("M", 1000),
    ("CM", 900),
    ("D", 500),
    ("CD", 400),
    ("C", 100),
    ("XC", 90),
    ("L", 50),
    ("XL", 40),
    ("X", 10),
    ("IX", 9),
    ("V", 5),
    ("IV", 4),
    ("I", 1),
)


def toRoman(n):
    if not (0 < n < 5000):
        raise ValueError("number out of range (must be 1..4999)")
    if int(n) != n:
        raise ValueError("decimals can not be converted")

    result = []
    value = int(n)
    for numeral, integer in _ROMAN_NUMERAL_MAP:
        while value >= integer:
            result.append(numeral)
            value -= integer
    return "".join(result)


def _element_row(z, symbol, name, a, abundance=1.0):
    return {
        "symbol_A": f"{symbol}({a})",
        "symbol": symbol,
        "name": name,
        "Z": int(z),
        "mass": float(a),
        "A": int(a),
        "abundance": float(abundance),
    }


_ELEMENT_ROWS = [
    _element_row(1, "H", "hydrogen", 1, 1.0),
    _element_row(1, "D", "deuterium", 2, 1e-12),
    _element_row(1, "T", "tritium", 3, 1e-24),
    _element_row(2, "He", "helium", 4),
    _element_row(3, "Li", "lithium", 7),
    _element_row(4, "Be", "beryllium", 9),
    _element_row(5, "B", "boron", 11),
    _element_row(6, "C", "carbon", 12),
    _element_row(7, "N", "nitrogen", 14),
    _element_row(8, "O", "oxygen", 16),
    _element_row(9, "F", "fluorine", 19),
    _element_row(10, "Ne", "neon", 20),
    _element_row(11, "Na", "sodium", 23),
    _element_row(12, "Mg", "magnesium", 24),
    _element_row(13, "Al", "aluminum", 27),
    _element_row(14, "Si", "silicon", 28),
    _element_row(15, "P", "phosphorus", 31),
    _element_row(16, "S", "sulfur", 32),
    _element_row(17, "Cl", "chlorine", 35),
    _element_row(18, "Ar", "argon", 40),
    _element_row(19, "K", "potassium", 39),
    _element_row(20, "Ca", "calcium", 40),
    _element_row(21, "Sc", "scandium", 45),
    _element_row(22, "Ti", "titanium", 48),
    _element_row(23, "V", "vanadium", 51),
    _element_row(24, "Cr", "chromium", 52),
    _element_row(25, "Mn", "manganese", 55),
    _element_row(26, "Fe", "iron", 56),
    _element_row(27, "Co", "cobalt", 59),
    _element_row(28, "Ni", "nickel", 59),
    _element_row(29, "Cu", "copper", 64),
    _element_row(30, "Zn", "zinc", 65),
    _element_row(31, "Ga", "gallium", 70),
    _element_row(32, "Ge", "germanium", 73),
    _element_row(33, "As", "arsenic", 75),
    _element_row(34, "Se", "selenium", 79),
    _element_row(35, "Br", "bromine", 80),
    _element_row(36, "Kr", "krypton", 84),
    _element_row(37, "Rb", "rubidium", 85),
    _element_row(38, "Sr", "strontium", 88),
    _element_row(39, "Y", "yttrium", 89),
    _element_row(40, "Zr", "zirconium", 91),
    _element_row(41, "Nb", "niobium", 93),
    _element_row(42, "Mo", "molybdenum", 96),
    _element_row(43, "Tc", "technetium", 98),
    _element_row(44, "Ru", "ruthenium", 101),
    _element_row(45, "Rh", "rhodium", 103),
    _element_row(46, "Pd", "palladium", 106),
    _element_row(47, "Ag", "silver", 108),
    _element_row(48, "Cd", "cadmium", 112),
    _element_row(49, "In", "indium", 115),
    _element_row(50, "Sn", "tin", 119),
    _element_row(51, "Sb", "antimony", 122),
    _element_row(52, "Te", "tellurium", 128),
    _element_row(53, "I", "iodine", 127),
    _element_row(54, "Xe", "xenon", 131),
    _element_row(55, "Cs", "cesium", 133),
    _element_row(56, "Ba", "barium", 137),
    _element_row(57, "La", "lanthanum", 139),
    _element_row(58, "Ce", "cerium", 140),
    _element_row(59, "Pr", "praseodymium", 141),
    _element_row(60, "Nd", "neodymium", 144),
    _element_row(61, "Pm", "promethium", 145),
    _element_row(62, "Sm", "samarium", 150),
    _element_row(63, "Eu", "europium", 152),
    _element_row(64, "Gd", "gadolinium", 157),
    _element_row(65, "Tb", "terbium", 159),
    _element_row(66, "Dy", "dysprosium", 163),
    _element_row(67, "Ho", "holmium", 165),
    _element_row(68, "Er", "erbium", 167),
    _element_row(69, "Tm", "thulium", 169),
    _element_row(70, "Yb", "ytterbium", 173),
    _element_row(71, "Lu", "lutetium", 175),
    _element_row(72, "Hf", "hafnium", 178),
    _element_row(73, "Ta", "tantalum", 181),
    _element_row(74, "W", "tungsten", 184),
    _element_row(75, "Re", "rhenium", 186),
    _element_row(76, "Os", "osmium", 190),
    _element_row(77, "Ir", "iridium", 192),
    _element_row(78, "Pt", "platinum", 195),
    _element_row(79, "Au", "gold", 197),
    _element_row(80, "Hg", "mercury", 201),
    _element_row(81, "Tl", "thallium", 204),
    _element_row(82, "Pb", "lead", 207),
    _element_row(83, "Bi", "bismuth", 209),
    _element_row(84, "Po", "polonium", 209),
    _element_row(85, "At", "astatine", 210),
    _element_row(86, "Rn", "radon", 222),
    _element_row(87, "Fr", "francium", 223),
    _element_row(88, "Ra", "radium", 226),
    _element_row(89, "Ac", "actinium", 227),
    _element_row(90, "Th", "thorium", 232),
    _element_row(91, "Pa", "protactinium", 231),
    _element_row(92, "U", "uranium", 238),
    _element_row(93, "Np", "neptunium", 237),
    _element_row(94, "Pu", "plutonium", 244),
    _element_row(95, "Am", "americium", 243),
    _element_row(96, "Cm", "curium", 247),
    _element_row(97, "Bk", "berkelium", 247),
    _element_row(98, "Cf", "californium", 251),
    _element_row(99, "Es", "einsteinium", 252),
    _element_row(100, "Fm", "fermium", 257),
    _element_row(101, "Md", "mendelevium", 258),
    _element_row(102, "No", "nobelium", 259),
    _element_row(103, "Lr", "lawrencium", 266),
    _element_row(104, "Rf", "rutherfordium", 267),
    _element_row(105, "Db", "dubnium", 268),
    _element_row(106, "Sg", "seaborgium", 269),
    _element_row(107, "Bh", "bohrium", 270),
    _element_row(108, "Hs", "hassium", 269),
    _element_row(109, "Mt", "meitnerium", 278),
    _element_row(110, "Ds", "darmstadtium", 281),
    _element_row(111, "Rg", "roentgenium", 282),
    _element_row(112, "Cn", "copernicium", 285),
    _element_row(113, "Nh", "nihonium", 286),
    _element_row(114, "Fl", "flerovium", 289),
    _element_row(115, "Mc", "moscovium", 290),
    _element_row(116, "Lv", "livermorium", 293),
    _element_row(117, "Ts", "tennessine", 294),
    _element_row(118, "Og", "oganesson", 294),
]


def atomic_element(
    symbol_A=None,
    symbol=None,
    name=None,
    Z=None,
    Z_ion=None,
    mass=None,
    A=None,
    abundance=None,
    use_D_T=True,
    return_most_abundant=True,
):
    if symbol:
        match = re.match(r"(\d*)([a-zA-Z]+)(\d*)$", symbol)
        if not match:
            raise ValueError(f"Wrong form of symbol: {symbol}")
        symbol = match.group(2)
        if symbol in {"d", "D"}:
            symbol = "D"
            if A is None:
                A = 2
        elif symbol in {"t", "T"}:
            symbol = "T"
            if A is None:
                A = 3
        else:
            symbol = symbol[0].upper() + symbol[1:].lower()

        if match.group(1) == "" and match.group(3) != "":
            A = int(match.group(3))
        elif match.group(1) != "" and match.group(3) != "":
            A = int(match.group(1))
            Z_ion = int(match.group(3))

    query = {
        "symbol_A": symbol_A,
        "symbol": symbol,
        "name": name,
        "Z": Z,
        "mass": mass,
        "A": A,
        "abundance": abundance,
    }
    query = {key: value for key, value in query.items() if value is not None}

    matches = {}
    for item in _ELEMENT_ROWS:
        if not use_D_T and item["symbol"] in {"D", "T"}:
            continue
        matched = True
        for key, value in query.items():
            if item[key] != value:
                matched = False
                break
        if matched:
            entry = copy.deepcopy(item)
            entry["Z_ion"] = int(entry["Z"] if Z_ion is None else Z_ion)
            matches[entry["symbol_A"]] = entry

    if return_most_abundant and len(matches) > 1:
        best = max(matches.values(), key=lambda entry: entry["abundance"])
        matches = {best["symbol_A"]: best}

    if not matches:
        items = ", ".join(f"{key}={value!r}" for key, value in query.items())
        raise ValueError(f"No atomic element satisfies {items}")

    return matches
