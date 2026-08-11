"""PDF -> memory records, feeding the memstrength stores.

Stages, deliberately separable:

    pdf.py      PDF -> RawPage           (needs pypdf; the only such module)
    segment.py  RawPage -> Candidate     (stdlib; where the judgement lives)
    admit.py    Candidate -> Memory+Signals, deduped, cold-start priors
    pipeline.py wiring plus a report

Segmentation and admission are stdlib-only so they can be tested against
hand-written fixtures rather than a 43MB PDF.
"""

from .segment import Candidate, segment, segment_page
from .admit import Admitted, admit, dedupe, memory_id, load_stores
from .pipeline import run, write_json, Report

__all__ = [
    "Candidate", "segment", "segment_page",
    "Admitted", "admit", "dedupe", "memory_id", "load_stores",
    "run", "write_json", "Report",
]
