"""PDF -> raw pages. The only module that needs a third-party dependency.

Everything downstream works on RawPage, so segmentation and admission stay
stdlib-only and testable without a 43MB fixture.

    .venv/bin/pip install pypdf cryptography
"""


class RawPage(object):
    """One page of extracted text, with its provenance."""

    __slots__ = ("number", "lines", "section")

    def __init__(self, number, lines, section=None):
        self.number = number
        self.lines = lines
        self.section = section

    def __repr__(self):
        return "RawPage(%d, %r, %d lines)" % (self.number, self.section, len(self.lines))


def read_pdf(path, password=""):
    """Yield a RawPage per page. Requires pypdf; imported lazily so that
    importing this package does not hard-fail without it."""
    try:
        from pypdf import PdfReader
    except ImportError:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "pypdf is required to read PDFs: .venv/bin/pip install pypdf cryptography"
        )

    reader = PdfReader(path)
    if reader.is_encrypted:
        # Many published PDFs carry an owner password only; an empty user
        # password still opens them for reading.
        try:
            reader.decrypt(password)
        except Exception:
            pass

    for index, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
        yield RawPage(index + 1, lines)
