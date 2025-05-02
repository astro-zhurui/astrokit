#!/usr/bin/env python3
# Copied from https://hsc-gitlab.mtk.nao.ac.jp/ssp-software/data-access-tools/-/tree/master/pdr3
# Original license: MIT License
import argparse
import base64
import contextlib
import csv
import dataclasses
import getpass
import io
import math
import os
import re
import sys
import tarfile
import urllib.request

from typing import cast, Any, Callable, Dict, Generator, IO, List, Optional, Tuple, Union

__all__ = []
def export(obj):
    if isinstance(obj, str):
        name = obj
    else:
        name = obj.__name__
    __all__.append(name)
    return obj


api_url = "https://hsc-release.mtk.nao.ac.jp/psf/pdr3"

available_reruns = [
    "pdr3_dud_rev",
    "pdr3_dud",
    "pdr3_wide",
]

default_rerun = "pdr3_wide"

available_types = [
    "coadd", "warp",
]

default_type = "coadd"

default_centered = True
default_name = "{lineno}_psf_{type}_{ra:.5f}_{dec:+.5f}_{tract}_{patch[0]},{patch[1]}_{filter}"

export("ANYTRACT")
ANYTRACT = -1
export("ANYPATCH")
ANYPATCH = (-1, -1)
export("ALLFILTERS")
ALLFILTERS = "all"


def main():
    parser = argparse.ArgumentParser(
        fromfile_prefix_chars="@",
        description="""
            Download PSF images of coadd images from the website of HSC data release.
        """,
    )
    parser.add_argument("--ra", metavar="DEGREES", type=parse_longitude, help="""
        R.A.2000.
    """)
    parser.add_argument("--dec", metavar="DEGREES", type=parse_latitude, help="""
        Dec.2000.
    """)
    parser.add_argument("--filter", type=parse_filter_opt, help="""
        Filter name.
    """)
    parser.add_argument("--rerun", choices=available_reruns, default=default_rerun, help="""
        Rerun name.
    """)
    parser.add_argument("--tract", type=parse_tract_opt, help="""
        Tract number.
    """)
    parser.add_argument("--patch", metavar="X,Y", type=parse_patch_opt, help="""
        Patch x,y.
    """)
    parser.add_argument("--centered", metavar="BOOL", type=parse_bool, default=default_centered, help=f"""
        If true, PSF's center is at the center of the image. (Default: {default_centered})
    """)
    parser.add_argument("--type", choices=available_types, default=default_type, help="""
        Data type.
    """)
    parser.add_argument("--name", type=str, default=default_name, help=f"""
        Output name. (python's format string; default: "{default_name}")
    """)
    parser.add_argument("--list", metavar="PATH", type=str, help="""
        Path to a coordinate list.
        If this list is given, the other command-line arguments are optional.
        Missing fields in the list will default
        to the values given in the command-line.
    """)
    parser.add_argument("--listtype", choices=["auto", "txt", "csv"], default="auto", help="""
        How to interpret the argument of --list.
        "auto" (default): Follow the extension of the file name. /
        "txt": Fields are separated by one or more spaces. /
        "csv": Comma-separated volume.
    """)
    parser.add_argument("--user", type=str, help="""
        User account.
    """)
    parser.add_argument("--password", type=str, help="""
        Password.
        If you specify --password, your password is disclosed to everybody
        on the computer you use.
        Use of --password-env is recommended instead,
        especially when this script is run on a shared-use computer.
    """)
    parser.add_argument("--password-env", metavar="ENV", type=str, default="HSC_SSP_CAS_PASSWORD", help="""
        Name of the environment variable from which to read password.
        Use `read -s HSC_SSP_CAS_PASSWORD` to put your password into
        $HSC_SSP_CAS_PASSWORD.
    """)

    args = parser.parse_args()

    psfreq = PsfRequest.create(
        rerun=args.rerun,
        type=args.type,
        filter=args.filter,
        tract=args.tract,
        patch=args.patch,
        ra=args.ra,
        dec=args.dec,
        centered=args.centered,
        name=args.name,
    )

    if not args.password:
        args.password = os.environ.get(args.password_env)

    if args.list:
        with open_inputfile(sys.stdin if args.list == "-" else args.list) as f:
            psfreqs = read_psfreqs(f, default=psfreq, type=args.listtype)
    else:
        if not psfreq.iscomplete():
            raise RuntimeError(f"Specify either (--ra --dec) or --list.")
        psfreqs = [psfreq]

    download(psfreqs, user=args.user, password=args.password, onmemory=False)


@export
@dataclasses.dataclass(order=True)
class PsfRequest:
    """
    Request of PSF(s).

    The constructor is not intended to be called by usual users.
    Use PsfRequest.create() instead.

    Parameters
    ----------
    rerun
        Rerun name.
    type
        "coadd" (coadd), or
        "warp" (warp).
    filter
        Filter name.
        This member can be `ALLFILTERS`.
    tract
        Tract number.
        This member can be `ANYTRACT`.
    patch
        Patch x,y.
        This member can be `ANYPATCH`.
    ra
        R.A.2000, in degrees.
    dec
        Dec.2000, in degrees.
    centered
        Is PSF centered?
    name
        File name format (without extension ".fits")
    lineno
        Line number in a list file.
    """
    rerun: str = default_rerun
    type: str = default_type
    filter: str = ALLFILTERS
    tract: int = ANYTRACT
    patch: Tuple[int, int] = ANYPATCH
    ra: float = math.nan
    dec: float = math.nan
    centered: bool = default_centered
    name: str = default_name
    lineno: int = 0

    @staticmethod
    def create(
        rerun: Union[str, None] = None,
        type: Union[str, None] = None,
        filter: Union[str, None] = None,
        tract: Union[str, int, None] = None,
        patch: Union[str, Tuple[int, int], None] = None,
        ra: Union[str, float, None] = None,
        dec: Union[str, float, None] = None,
        centered: Union[str, bool, None] = None,
        name: Union[str, None] = None,
        lineno: Union[int, None] = None,
        default: Union["PsfRequest", None] = None,
    ) -> "PsfRequest":
        """
        Create a PsfRequest object.

        If any parameter is omitted,
        it defaults to the corresponding field of the `default` argument.

        Parameters
        ----------
        rerun
            Rerun name.
        type
            "coadd" (coadd),
            "warp" (warp).
        filter
            Filter name.
            This member can be `ALLFILTERS`.
        tract
            Tract number.
            This member can be `ANYTRACT`.
        patch
            Patch x,y.
            This member can be `ANYPATCH`.
        ra
            R.A.2000, in degrees.
            This argument can be a string like "12:34:56.789" (hours),
            "12h34m56.789s", "1.2345rad" (radians), etc.
        dec
            Dec.2000, in degrees.
            This argument can be a string like "-1:23:45.678" (degrees),
            "-1d23m45.678s", "1.2345rad" (radians), etc.
        centered
            Is PSF centered?
        name
            File name format (without extension ".fits")
        lineno
            Line number in a list file.
        default
            Default value.

        Returns
        -------
        psfreq
            Created `PsfRequest` object.
        """
        if default is None:
            psfreq = PsfRequest()
        else:
            psfreq = PsfRequest(*dataclasses.astuple(default))

        if rerun is not None:
            psfreq.rerun = parse_rerun(rerun)
        if type is not None:
            psfreq.type = parse_type(type)
        if filter is not None:
            psfreq.filter = parse_filter_opt(filter)
        if tract is not None:
            psfreq.tract = parse_tract_opt(tract)
        if patch is not None:
            psfreq.patch = parse_patch_opt(patch)
        if ra is not None:
            psfreq.ra = parse_longitude(ra)
        if dec is not None:
            psfreq.dec = parse_latitude(dec)
        if centered is not None:
            psfreq.centered = parse_bool(centered)
        if name is not None:
            psfreq.name = str(name)
        if lineno is not None:
            psfreq.lineno = int(lineno)

        return psfreq

    def iscomplete(self) -> bool:
        """
        Whether or not `self` is complete.

        If, for example, user creates a `PsfRequest` object:
            psfreq = PsfRequest.create(filter="i")
        then `psfreq` does not have valid values of `ra` and `dec`.
        In such a case, this function returns False.

        Returns
        -------
        iscomplete
            True if `self` is complete
        """
        return (self.ra == self.ra
            and self.dec == self.dec
        )

    def explode(self) -> List["PsfRequest"]:
        """
        Make copies of `self` with more specific values.

        Returns
        -------
        psfreqs
            List of `PsfRequest` objects, each being more specific than `self`.
        """
        if self.filter == ALLFILTERS:
            return [PsfRequest.create(filter=f, default=self) for f in _all_filters]
        else:
            return [PsfRequest.create(default=self)]


@export
def read_psfreqs(file: Union[str, IO], default: Optional[PsfRequest] = None, type: Optional[str] = None) -> List[PsfRequest]:
    """
    Read a file to get a list of `PsfRequest` objects.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.
    type
        File type. One of "auto" (=None), "txt", "csv".
        By default, the file type is guessed
        from the extension part of the file name.

    Returns
    -------
    psfreqs
        List of `PsfRequest` objects.
    """
    if (not type) or type == "auto":
        isfileobj = hasattr(file, "read")
        if isfileobj:
            name = getattr(file, "name", "(file name not available)")
        else:
            name = file
        _, ext = os.path.splitext(name)
        type = ext.lstrip(".") or "txt"

    if type == "txt":
        return read_psfreqs_from_txt(file, default=default)
    if type == "csv":
        return read_psfreqs_from_csv(file, default=default)

    raise ValueError(f"Invalid file type: {type}")


@export
def read_psfreqs_from_txt(file, default=None):
    """
    Read a space-separated volume to get a list of `PsfRequest` objects.
    The first line must contain column names.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.

    Returns
    -------
    psfreqs
        List of `PsfRequest` objects.
    """
    allfields = set(field.name for field in dataclasses.fields(PsfRequest))

    with open_inputfile(file) as f:
        f = io.TextIOWrapper(f, encoding="utf-8")

        fieldnames = re.sub(r"^#\??\s*", "", f.readline().strip().lower()).split()
        validfields = [(i, field) for i, field in enumerate(fieldnames) if field in allfields]
        if not validfields:
            raise RuntimeError("No column has a valid name in the list.")

        psfreqs = []
        for lineno, line in enumerate(f, start=2):
            row = line.strip().split()
            if len(row) != len(fieldnames):
                raise RuntimeError(f"line {lineno}: number of fields ({len(row)}) does not agree with what expected ({len(fieldnames)})")
            args = {"lineno": lineno}
            args.update((field, row[i]) for i, field in validfields)
            psfreqs.append(PsfRequest.create(default=default, **args))

        return psfreqs


@export
def read_psfreqs_from_csv(file, default=None):
    """
    Read a comma-separated volume to get a list of `PsfRequest` objects.
    The first line must contain column names.

    Parameters
    ----------
    file
        A file path or a file object.
    default
        Default values.
        Fields that cannot be obtained from the file
        defaults to the corresponding fields of this object.

    Returns
    -------
    psfreqs
        List of `PsfRequest` objects.
    """
    allfields = set(field.name for field in dataclasses.fields(PsfRequest))

    with open_inputfile(file) as f:
        reader = csv.reader(io.TextIOWrapper(f, encoding="utf-8", newline=""))

        fieldnames = next(reader)
        if len(fieldnames) > 0:
            fieldnames[0] = re.sub(r"^#\??\s*", "", fieldnames[0].strip())
        fieldnames = [field.strip().lower() for field in fieldnames]

        validfields = [(i, field) for i, field in enumerate(fieldnames) if field in allfields]
        if not validfields:
            raise RuntimeError("No column has a valid name in the list.")

        psfreqs = []
        for lineno, row in enumerate(reader, start=2):
            if len(row) != len(fieldnames):
                raise RuntimeError(f"line {lineno}: number of fields ({len(row)}) does not agree with what expected ({len(fieldnames)})")
            args = {"lineno": lineno}
            args.update((field, row[i]) for i, field in validfields)
            psfreqs.append(PsfRequest.create(default=default, **args))

        return psfreqs


@contextlib.contextmanager
def open_inputfile(file: Union[str, IO]) -> Generator[IO[bytes], None, None]:
    """
    Open a file with "rb" mode.

    If `file` is a text file object, `file.buffer` will be returned.

    Parameters
    ----------
    file
        A file path or a file object.

    Returns
    -------
    contextmanager
        Context manager.
        When the context is exitted,
        the file is closed if the file has been opened by this function.
        Otherwise, the file is kept open.
    """
    if hasattr(file, "read"):
        # This is already a file object
        yield getattr(file, "buffer", file)
    else:
        file = cast(str, file)
        f = open(file, "rb")
        try:
            yield f
        finally:
            f.close()


def parse_rerun(s: str) -> str:
    """
    Interpret a string representing a rerun name.

    Parameters
    ----------
    s
        Rerun name.

    Returns
    -------
    rerun
        Rerun name.
    """
    lower = s.lower()
    if lower in available_reruns:
        return lower
    raise ValueError(f"Invalid rerun: {s}")


def parse_type(s: str) -> str:
    """
    Interpret a string representing an image type.

    Parameters
    ----------
    s
        Image type.

    Returns
    -------
    type
        Image type.
    """
    lower = s.lower()
    if lower in available_types:
        return lower
    raise ValueError(f"Invalid type: {s}")



def parse_tract_opt(s: Union[str, int, None]) -> int:
    """
    Interpret a string (etc) representing a tract.

    Parameters
    ----------
    s
        Tract.
        This argument may be `ANYTRACT`.

    Returns
    -------
    tract
        Tract in an integer.
    """
    if s is None:
        return ANYTRACT
    if isinstance(s, int):
        return s

    s = s.lower()
    if s in ["any", "auto"]:
        return ANYTRACT
    return int(s)


def parse_patch_opt(s: Union[str, Tuple[int, int], None]) -> Tuple[int, int]:
    """
    Interpret a string (etc) representing a patch.

    Parameters
    ----------
    s
        Patch.
        This argument may be `ANYPATCH`.

    Returns
    -------
    patch
        Patch as a pair of integers.
    """
    if s is None:
        return ANYPATCH
    if isinstance(s, (tuple, list)):
        return (int(s[0]), int(s[1]))

    s = s.lower()
    if s in ["any", "auto"]:
        return ANYPATCH

    x, y = s.split(",")
    return (int(x), int(y))


def parse_bool(s: Union[str, bool]) -> bool:
    """
    Interpret a string (etc) representing a boolean value.

    Parameters
    ----------
    s
        A string (etc) representing a boolean value.

    Returns
    -------
    b
        True or False.
    """
    if isinstance(s, bool):
        return s

    return {
        "false": False,
        "f": False,
        "no": False,
        "n": False,
        "off": False,
        "0": False,
        "true": True,
        "t": True,
        "yes": True,
        "y": True,
        "on": True,
        "1": True,
    }[s.lower()]


def parse_longitude(s: Union[str, float]) -> float:
    """
    Interpret a longitude.

    Parameters
    ----------
    s
        A string representing a longitude,
        or a float value in degrees.

    Returns
    -------
    longitude
        Degrees.
    """
    type, value = _parse_angle(s)
    if type == "sex":
        return 15 * value
    else:
        return value


def parse_latitude(s: Union[str, float]) -> float:
    """
    Interpret a latitude.

    Parameters
    ----------
    s
        A string representing a latitude,
        or a float value in degrees.

    Returns
    -------
    latitude
        Degrees.
    """
    type, value = _parse_angle(s)
    return value


def parse_degree(s: Union[str, float]) -> float:
    """
    Interpret an angle, which is in degrees by default.

    Parameters
    ----------
    s
        A string representing an angle,
        or a float value in degrees.

    Returns
    -------
    angle
        Degrees.
    """
    type, value = _parse_angle(s)
    return value


def _parse_angle(s: Union[str, float]) -> Tuple[str, float]:
    """
    Interpret an angle.

    Parameters
    ----------
    s
        A string representing an angle.

    Returns
    -------
    type
      - "bare"
        `s` did not have its unit.
        What `angle` means must be decided by the caller.
      - "sex"
        `s` was in "99:99:99.999". It may be hours or degrees.
        What `angle` means must be decided by the caller.
      - "deg"
        `angle` is in degrees.

    angle
        a float value
    """
    try:
        if isinstance(s, (float, int)):
            return "bare", float(s)

        s = re.sub(r"\s", "", s).lower()
        m = re.match(r"\A(.+)(deg|degrees?|amin|arcmin|arcminutes?|asec|arcsec|arcseconds?|rad|radians?)\Z", s)
        if m:
            value, unit = m.groups()
            return "deg", float(value) * _angle_units[unit]

        m = re.match(r"\A([+\-]?)([0-9].*)d([0-9].*)m([0-9].*)s\Z", s)
        if m:
            sign_s, degrees, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "deg", sign * (float(seconds) / 3600 + float(minutes) / 60 + float(degrees))

        m = re.match(r"\A([+\-]?)([0-9].*)h([0-9].*)m([0-9].*)s\Z", s)
        if m:
            sign_s, hours, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "deg", 15.0 * sign * (float(seconds) / 3600 + float(minutes) / 60 + float(hours))

        m = re.match(r"\A([+\-]?)([0-9].*):([0-9].*):([0-9].*)\Z", s)
        if m:
            sign_s, degrees, minutes, seconds = m.groups()
            sign = -1.0 if sign_s == "-" else 1.0
            return "sex", sign * (float(seconds) / 3600 + float(minutes) / 60 + float(degrees))

        return "bare", float(s)

    except Exception:
        raise ValueError(f"Cannot interpret angle: '{s}'") from None


_angle_units = {
    "deg": 1.0,
    "degree": 1.0,
    "degrees": 1.0,
    "amin": 1.0 / 60,
    "arcmin": 1.0 / 60,
    "arcminute": 1.0 / 60,
    "arcminutes": 1.0 / 60,
    "asec": 1.0 / 3600,
    "arcsec": 1.0 / 3600,
    "arcsecond": 1.0 / 3600,
    "arcseconds": 1.0 / 3600,
    "rad": 180 / math.pi,
    "radian": 180 / math.pi,
    "radians": 180 / math.pi,
}


def parse_filter(s: str) -> str:
    """
    Interpret a filter name.

    Parameters
    ----------
    s
        A string representing a filter.
        This may be an alias of a filter name.
        (Like "g" for "HSC-G")

    Returns
    -------
    filter
        A filter name.
    """
    if s in _all_filters:
        return s

    for physicalname, info in _all_filters.items():
        if s in info["alias"]:
            return physicalname

    raise ValueError(f"filter '{s}' not found.")


_all_filters = dict([
    ("HSC-G", {"alias": {"W-S-G+", "g"}, "display": "g"}),
    ("HSC-R", {"alias": {"W-S-R+", "r"}, "display": "r"}),
    ("HSC-I", {"alias": {"W-S-I+", "i"}, "display": "i"}),
    ("HSC-Z", {"alias": {"W-S-Z+", "z"}, "display": "z"}),
    ("HSC-Y", {"alias": {"W-S-ZR", "y"}, "display": "y"}),
    ("IB0945", {"alias": {"I945"}, "display": "I945"}),
    ("NB0387", {"alias": {"N387"}, "display": "N387"}),
    ("NB0400", {"alias": {"N400"}, "display": "N400"}),
    ("NB0468", {"alias": {"N468"}, "display": "N468"}),
    ("NB0515", {"alias": {"N515"}, "display": "N515"}),
    ("NB0527", {"alias": {"N527"}, "display": "N527"}),
    ("NB0656", {"alias": {"N656"}, "display": "N656"}),
    ("NB0718", {"alias": {"N718"}, "display": "N718"}),
    ("NB0816", {"alias": {"N816"}, "display": "N816"}),
    ("NB0921", {"alias": {"N921"}, "display": "N921"}),
    ("NB0926", {"alias": {"N926"}, "display": "N926"}),
    ("NB0973", {"alias": {"N973"}, "display": "N973"}),
    ("NB1010", {"alias": {"N1010"}, "display": "N1010"}),
    ("ENG-R1", {"alias": {"109", "r1"}, "display": "r1"}),
    ("PH", {"alias": {"PH"}, "display": "PH"}),
    ("SH", {"alias": {"SH"}, "display": "SH"}),
    ("MegaCam-u" , {"alias": {"u2"}, "display": "MegaCam-u" }),
    ("MegaCam-uS", {"alias": {"u1"}, "display": "MegaCam-uS"}),
    ("VIRCAM-H"    , {"alias": {"Hvir", "hvir"}, "display": "VIRCAM-H"    }),
    ("VIRCAM-J"    , {"alias": {"Jvir", "jvir"}, "display": "VIRCAM-J"    }),
    ("VIRCAM-Ks"   , {"alias": {"Ksvir", "ksvir"}, "display": "VIRCAM-Ks"   }),
    ("VIRCAM-NB118", {"alias": {"NB118vir", "n118vir"}, "display": "VIRCAM-NB118"}),
    ("VIRCAM-Y"    , {"alias": {"Yvir", "yvir"}, "display": "VIRCAM-Y"    }),
    ("WFCAM-H", {"alias": {"Hwf", "hwf"}, "display": "WFCAM-H"}),
    ("WFCAM-J", {"alias": {"Jwf", "jwf"}, "display": "WFCAM-J"}),
    ("WFCAM-K", {"alias": {"Kwf", "kwf"}, "display": "WFCAM-K"}),
])


def parse_filter_opt(s: Optional[str]) -> str:
    """
    Interpret a filter name.
    The argument may be `ALLFILTERS`.or None
    (both have the same meaning).

    Parameters
    ----------
    s
        A string representing a filter.
        This may be an alias of a filter name.
        (Like "g" for "HSC-G")
        Or it may be `ALLFILTERS`.
        If `s` is None, it has the same meaning as `ALLFILTERS`.

    Returns
    -------
    filter
        A filter name.
        This can be `ALLFILTERS`
    """
    if s is None:
        return ALLFILTERS

    if s.lower() == ALLFILTERS:
        return ALLFILTERS
    return parse_filter(s)


@export
def download(psfreqs: Union[PsfRequest, List[PsfRequest]], user: Optional[str] = None, password: Optional[str] = None, *, onmemory: bool = True) -> Union[list, List[list], None]:
    """
    Download PSFs by sending `psfreqs`.

    Parameters
    ----------
    psfreqs
        A `PsfRequest` object or a list of `PsfRequest` objects
    user
        Username. If None, it will be asked interactively.
    password
        Password. If None, it will be asked interactively.
    onmemory
        Return `datalist` on memory.
        If `onmemory` is False, downloaded PSFs are written to files.

    Returns
    -------
    datalist
        If onmemory == False, `datalist` is None.
        If onmemory == True:
          - If `psfreqs` is a simple `PsfRequest` object,
            `datalist[j]` is a tuple `(metadata: dict, data: bytes)`.
            This is a list because there may be more than one file
            for a single `PsfRequest` (if, say, filter==ALLFILTERS).
            This list may also be empty, which means no data was found.
          - If `psfreqs` is a list of `PsfRequest` objects,
            `datalist[i]` corresponds to `psfreqs[i]`, and
            `datalist[i][j]` is a tuple `(metadata: dict, data: bytes)`.
    """
    isscalar = isinstance(psfreqs, PsfRequest)
    if isscalar:
        psfreqs = [cast(PsfRequest, psfreqs)]
    psfreqs = cast(List[PsfRequest], psfreqs)

    ret = _download(psfreqs, user, password, onmemory=onmemory)
    if isscalar and onmemory:
        ret = cast(List[list], ret)
        return ret[0]

    return ret


def _download(psfreqs: List[PsfRequest], user: Optional[str], password: Optional[str], *, onmemory: bool) -> Optional[List[list]]:
    """
    Download PSFs by sending `psfreqs`.

    Parameters
    ----------
    psfreqs
        A list of `PsfRequest` objects
    user
        Username. If None, it will be asked interactively.
    password
        Password. If None, it will be asked interactively.
    onmemory
        Return `datalist` on memory.
        If `onmemory` is False, downloaded PSFs are written to files.

    Returns
    -------
    datalist
        If onmemory == False, `datalist` is None.
        If onmemory == True,
        `datalist[i]` corresponds to `psfreqs[i]`, and
        `datalist[i][j]` is a tuple `(metadata: dict, data: bytes)`.
    """
    if not psfreqs:
        return [] if onmemory else None

    for psfreq in psfreqs:
        if not psfreq.iscomplete():
            raise RuntimeError(f"'ra' and 'dec' must be specified: {psfreq}")

    exploded_psfreqs: List[Tuple[PsfRequest, int]] = []
    for index, psfreq in enumerate(psfreqs):
        exploded_psfreqs.extend((r, index) for r in psfreq.explode())

    # Sort the psfreqs so that the server can use cache
    # as frequently as possible.
    # We will later use `index` to sort them back.
    exploded_psfreqs.sort()

    if not user:
        user = input("username? ").strip()
        if not user:
            raise RuntimeError("User name is empty.")

    if not password:
        password = getpass.getpass(prompt="password? ")
        if not password:
            raise RuntimeError("Password is empty.")

    chunksize = 990
    datalist: List[Tuple[int, dict, bytes]] = []

    for i in range(0, len(exploded_psfreqs), chunksize):
        ret = _download_chunk(exploded_psfreqs[i : i+chunksize], user, password, onmemory=onmemory)
        if onmemory:
            datalist += cast(list, ret)

    if onmemory:
        returnedlist: List[List[Tuple[dict, bytes]]] = [[] for i in range(len(psfreqs))]
        for index, metadata, data in datalist:
            returnedlist[index].append((metadata, data))

    return returnedlist if onmemory else None


def _download_chunk(psfreqs: List[Tuple[PsfRequest, Any]], user: str, password: str, *, onmemory: bool) -> Optional[list]:
    """
    Download PSFs by sending `psfreqs`.

    Parameters
    ----------
    psfreqs
        A list of `(PsfRequest, Any)`.
        The length of this list must be smaller than the server's limit.
        Each `PsfRequest` object must be explode()ed beforehand.
        The `Any` value attached to each `PsfRequest` object is a marker.
        The marker is used to indicate the `PsfRequest` in the returned list.
    user
        Username.
    password
        Password.
    onmemory
        Return `datalist` on memory.
        If `onmemory` is False, downloaded PSFs are written to files.

    Returns
    -------
    datalist
        If onmemory == False, `datalist` is None.
        If onmemory == True,
        each element is a tuple `(marker: Any, metadata: dict, data: bytes)`.
        For `marker`, see the comment for the parameter `psfreqs`.
        Two or more elements in this list may result
        from a single `PsfRequest` object.
    """
    fields = list(_format_psfreq_member.keys())
    coordlist = [f"#? {' '.join(fields)}"]
    for psfreq, index in psfreqs:
        coordlist.append(" ".join(_format_psfreq_member[field](getattr(psfreq, field)) for field in fields))

    boundary = "Boundary"
    header = (
        f'--{boundary}\r\n'
        f'Content-Disposition: form-data; name="list"; filename="coordlist.txt"\r\n'
        f'\r\n'
    )
    footer = (
        f'\r\n'
        f'--{boundary}--\r\n'
    )

    data = (header + "\n".join(coordlist) + footer).encode("utf-8")
    secret = base64.standard_b64encode(f"{user}:{password}".encode("utf-8")).decode("ascii")

    req = urllib.request.Request(
        api_url.rstrip("/") + "/cgi/getpsf?bulk=on",
        data=data,
        headers={
            "Authorization": f'Basic {secret}',
            "Content-Type": f'multipart/form-data; boundary="{boundary}"',
        },
        method="POST",
    )

    returnedlist = []

    with urllib.request.urlopen(req, timeout=3600) as fin:
        with tarfile.open(fileobj=fin, mode="r|") as tar:
            for info in tar:
                fitem = tar.extractfile(info)
                if fitem is None:
                    continue
                with fitem:
                    metadata = _tar_decompose_item_name(info.name)
                    psfreq, index = psfreqs[metadata["lineno"] - 2]
                    # Overwrite metadata's lineno (= lineno in this chunk)
                    # with psfreq's lineno (= global lineno)
                    # for fear of confusion.
                    metadata["lineno"] = psfreq.lineno
                    # Overwrite psfreq's tract and patch (which may be ANYTRACT, ANYPATCH)
                    # with metadata's tract and patch (which always have valid values)
                    # for fear of confusion.
                    psfreq.tract = metadata["tract"]
                    psfreq.patch = metadata["patch"]
                    metadata["psfreq"] = psfreq
                    if onmemory:
                        returnedlist.append((index, metadata, fitem.read()))
                    else:
                        filename = make_filename(metadata)
                        dirname = os.path.dirname(filename)
                        if dirname:
                            os.makedirs(dirname, exist_ok=True)
                        with open(filename, "wb") as fout:
                            _splice(fitem, fout)

    return returnedlist if onmemory else None


_format_psfreq_member: Dict[str, Callable[[str], Any]] = {
    "rerun": str,
    "type": str,
    "filter": str,
    "tract": lambda x: ("auto" if x == ANYTRACT else str(x)),
    "patch": lambda x: ("auto" if x == ANYPATCH else f"{x[0]},{x[1]}"),
    "ra": lambda x: f"{x:.16e}deg",
    "dec": lambda x: f"{x:.16e}deg",
    "centered": lambda x: ("true" if x else "false"),
}


def _tar_decompose_item_name(name: str) -> dict:
    """
    Get a metadata dictionary by decomposing an item name in a tar file.

    Parameters
    ----------
    name
        The name of an item in a tar file returned by the server.

    Returns
    -------
    metadata
        A dictionary that has the following keys:
          - "lineno": Line number (starting with 2).
          - "type": "coadd" or "warp".
          - "filter": Filter name.
          - "tract": Tract number.
          - "patch": Patch x,y.
          - "rerun": Rerun name.
          - "visit": (warp only) Visit number.
    """
    m = re.fullmatch(r"(?P<lineno>[0-9]+)-psf-calexp-(?P<rerun>[a-z_][a-z0-9_]*)-(?P<filter>[^/]+)-(?P<tract>[0-9]+)-(?P<x>[0-9]+),(?P<y>[0-9]+)-[+\-]?[0-9]+\.[0-9]+-[+\-]?[0-9]+\.[0-9]+\.fits", name)
    if m:
        metadata: Dict[str, Any] = m.groupdict()
        metadata["lineno"] = int(metadata["lineno"])
        metadata["type"] = "coadd"
        metadata["tract"] = int(metadata["tract"])
        metadata["patch"] = (int(metadata["x"]), int(metadata["y"]))
        del metadata["x"]
        del metadata["y"]
        return metadata

    m = re.fullmatch(r"(?P<lineno>[0-9]+)-psf-warp-(?P<rerun>[a-z_][a-z0-9_]*)-(?P<filter>[^/]+)-(?P<tract>[0-9]+)-(?P<x>[0-9]+),(?P<y>[0-9]+)-(?P<visit>[0-9]+)-[+\-]?[0-9]+\.[0-9]+-[+\-]?[0-9]+\.[0-9]+\.fits", name)
    if m:
        metadata: Dict[str, Any] = m.groupdict()
        metadata["lineno"] = int(metadata["lineno"])
        metadata["type"] = "warp"
        metadata["tract"] = int(metadata["tract"])
        metadata["patch"] = (int(metadata["x"]), int(metadata["y"]))
        del metadata["x"]
        del metadata["y"]
        metadata["visit"] = int(metadata["visit"])
        return metadata

    raise ValueError("File name not interpretable")


@export
def make_filename(metadata: dict) -> str:
    """
    Make a filename from `metadata` that is returned by `download(onmemory=True)`.

    Parameters
    ----------
    metadata
        A metadata dictionary.

    Returns
    -------
    filename
        File name.
    """
    psfreq = metadata["psfreq"]
    args = dataclasses.asdict(psfreq)
    type = args["type"]
    if type == "warp":
        args["name"] = f'{args["name"]}_{metadata["visit"]:06d}'

    name = args.pop("name")
    return name.format(**args) + ".fits"


def _splice(fin: IO[bytes], fout: IO[bytes]):
    """
    Read from `fin` and write to `fout` until the end of file.

    Parameters
    ----------
    fin
        Input file.
    fout
        Output file.
    """
    buffer = memoryview(bytearray(10485760))
    while True:
        n = fin.readinto(buffer)
        if n <= 0:
            break
        fout.write(buffer[:n])


if __name__ == "__main__":
    main()
