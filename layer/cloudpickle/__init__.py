# type: ignore
from .cloudpickle import *  # noqa
from .cloudpickle_fast import CloudPickler, dump, dumps  # noqa


# Conform to the convention used by python serialization libraries, which
# expose their Pickler subclass at top-level under the  "Pickler" name.
Pickler = CloudPickler

__version__ = "2.1.0"
