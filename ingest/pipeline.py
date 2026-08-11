"""Wire the stages together and report what happened.

    read_pdf  ->  segment  ->  admit  ->  stores / JSON

Run:
    .venv/bin/python -m ingest FIFA-2026.pdf --out results/fifa_memories.json
    .venv/bin/python -m ingest FIFA-2026.pdf --sample 12
"""

import json

from memstrength.store import MemoryStore, SignalStore

from .admit import admit, load_stores
from .segment import segment


class Report(object):
    __slots__ = ("pages", "candidates", "admitted", "by_kind", "by_section", "dropped")

    def __init__(self):
        self.pages = 0
        self.candidates = 0
        self.admitted = 0
        self.by_kind = {}
        self.by_section = {}
        self.dropped = 0

    def summary(self):
        lines = [
            "pages read        %d" % self.pages,
            "candidates found  %d" % self.candidates,
            "duplicates merged %d" % self.dropped,
            "memories admitted %d" % self.admitted,
            "",
            "by kind:",
        ]
        for k in sorted(self.by_kind, key=lambda k: -self.by_kind[k]):
            lines.append("  %-12s %4d" % (k, self.by_kind[k]))
        lines.append("")
        lines.append("by section:")
        for s in sorted(self.by_section, key=lambda s: -self.by_section[s]):
            lines.append("  %-34s %4d" % (str(s)[:34], self.by_section[s]))
        return "\n".join(lines)


def run(pdf_path, source=None, memory_store=None, signal_store=None, **admit_kw):
    """Extract memories from a PDF. Returns (admitted, stores, report)."""
    from .pdf import read_pdf  # lazy: keeps the dependency at the edge

    source = source or _basename(pdf_path)
    report = Report()

    pages = list(read_pdf(pdf_path))
    report.pages = len(pages)

    candidates = list(segment(pages))
    report.candidates = len(candidates)

    admitted = list(admit(candidates, source, **admit_kw))
    report.admitted = len(admitted)
    report.dropped = report.candidates - report.admitted

    for a in admitted:
        report.by_kind[a.candidate.kind] = report.by_kind.get(a.candidate.kind, 0) + 1
        sec = a.candidate.section
        report.by_section[sec] = report.by_section.get(sec, 0) + 1

    memory_store = memory_store if memory_store is not None else MemoryStore()
    signal_store = signal_store if signal_store is not None else SignalStore()
    load_stores(admitted, memory_store, signal_store)

    return admitted, (memory_store, signal_store), report


def _basename(path):
    name = path.replace("\\", "/").split("/")[-1]
    return name[:-4] if name.lower().endswith(".pdf") else name


def write_json(admitted, path, source, report):
    payload = {
        "source": source,
        "counts": {
            "pages": report.pages,
            "candidates": report.candidates,
            "admitted": report.admitted,
            "by_kind": report.by_kind,
        },
        "memories": [a.as_dict() for a in admitted],
    }
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
