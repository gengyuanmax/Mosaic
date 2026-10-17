"""Dataset adapter interface (PREPROCESSING, not runtime).

A ``DatasetAdapter`` converts one source dataset into canonical samples (see
``canonical.build_canonical_sample``). Subclass it per source dataset when adding
a new dataset; the output is the standardized ``data/mmiu/datasets/<task>.json``
that the runtime loader then reads. Not imported by the runtime package.
"""

from abc import ABCMeta, abstractmethod


class DatasetAdapter(metaclass=ABCMeta):
    @abstractmethod
    def load_raw(self):
        raise NotImplementedError

    @abstractmethod
    def to_canonical(self, record):
        raise NotImplementedError

    @abstractmethod
    def validate(self, canonical):
        raise NotImplementedError
