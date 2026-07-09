"""olg — overlapping-gene (OLG) protein sequence design.

Design two proteins encoded in overlapping reading frames of one DNA sequence: an ``OLGDesign``
iteratively decodes both frames under the codon-overlap constraint, with per-frame amino-acid logits
supplied by pluggable decoder wrappers. Re-exports the public API (``OLGDesign``, ``DesignConfig`` /
``ProteinConfig``, ``Arrangement``, constants, and exceptions).
"""
from olg.constants import *
from olg.config import *
from olg.coordinates import *
from olg.compatibility import *
from olg.exceptions import *
from olg.design import *
