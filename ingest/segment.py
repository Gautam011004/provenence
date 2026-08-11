"""Raw pages -> candidate memories.

This is where the judgement lives, so it is stdlib-only and unit tested against
hand-written page fixtures rather than the PDF.

The document is a brand manual, and it carries three kinds of thing worth
remembering, which need different handling:

  guideline  prose. "The Legal Notice may be reduced proportionally if ..."
             Hard-wrapped across lines, so lines are rejoined into sentences.
  rule       an explicit do/don't. "Don't omit the Legal Notice."
             Already one line, already atomic.
  spec       a table. A colour is a name followed by a run of label/value
             lines (CMYK, RGB, HEX, PMS ...). Treated as ONE memory: split
             per line it becomes a dozen useless fragments, and "HEX #FFFFFF"
             means nothing without the WHITE above it.

Everything else -- headings, image captions, layout labels, placeholder text --
is context or noise and is dropped. What survives has to be a statement someone
could later be right or wrong about.
"""

import re

FOOTER = re.compile(r"^ASSETS\s+BRAND\s+MANUAL\s+EDITION", re.I)
SECTION_AND_PAGE = re.compile(r"^(.*?)\s+(\d{1,3})$")

# Placeholder copy used to demonstrate layouts. Never a real fact.
FILLER = re.compile(
    r"lorem ipsum|consetetur|sadipscing|dolor sit amet|takimata|voluptua|"
    r"gubergren|aliquyam|invidunt",
    re.I,
)

SENTENCE_END = re.compile(r"[.!?]['\"’”)]?$")
CLAUSE_END = re.compile(r"[.!?:;]['\"’”)]?$")

# "Don't ...", "Do not ...", "Always ...", "Never ..." -- atomic on their own.
# The manual is typeset with a curly apostrophe (U+2019), not an ASCII one, so
# both have to be accepted; matching only ASCII silently files most of the
# document's prohibitions as ordinary prose.
APOSTROPHES = "'\u2019\u02bc"
RULE_OPENER = re.compile(
    r"^(don[%s]?t|do not|never|always|avoid|ensure|make sure)\b" % APOSTROPHES,
    re.I | re.UNICODE,
)

# A label/value row in a spec table: "CMYK 0/0/0/100", "HEX #FFFFFF",
# "PMS Black C", "TCX 19-0303". Short, starts with an uppercase token.
SPEC_ROW = re.compile(
    r"^(CMYK|RGB|HEX|PMS|TCX|TPG|RAL|NCS|MADEIRA|PANTONE)\b[\s#:]*\S", re.I
)

MIN_LENGTH = 30

# Letter-spaced display type ("S I LV E R") comes out of the PDF with the
# tracking preserved as real spaces. Detectable because most tokens are one or
# two characters long.
SPEC_LABEL = re.compile(r"^([A-Z]+)", re.I)


def despace(text):
    """Rejoin letter-spaced display type: 'S I LV E R' -> 'SILVER'."""
    tokens = text.split()
    if len(tokens) < 3:
        return text
    short = sum(1 for t in tokens if len(t) <= 2)
    if short / float(len(tokens)) < 0.5:
        return text
    return "".join(tokens)


class Candidate(object):
    """A prospective memory, with everything needed to trace it back."""

    __slots__ = ("text", "kind", "section", "page", "heading", "confidence")

    def __init__(self, text, kind, section, page, heading=None, confidence=1.0):
        self.text = text
        self.kind = kind
        self.section = section
        self.page = page
        self.heading = heading
        # How much the extraction itself is trusted, 0..1. Below 1 means the
        # text is probably right but something about it -- usually which colour
        # name a value block belongs to -- could not be resolved from a
        # text-only read of a multi-column layout. It flows through to
        # source_trust rather than causing the record to be dropped: the values
        # are still real, the attribution is what is shaky.
        self.confidence = confidence

    def __repr__(self):
        return "Candidate(%s, p%s, %r)" % (self.kind, self.page, self.text[:48])

    def as_dict(self):
        return {
            "text": self.text,
            "kind": self.kind,
            "section": self.section,
            "page": self.page,
            "heading": self.heading,
            "confidence": self.confidence,
        }


def normalise_section(raw):
    """The manual renders section names in small caps, so pypdf reports them
    with scrambled case ('offIcIAl EmBlEm'). Fold to a stable Title Case."""
    if not raw:
        return None
    cleaned = re.sub(r"\s+", " ", raw).strip()
    return cleaned.title() if cleaned else None


def split_footer(lines):
    """Return (body_lines, section_name).

    The last two lines of most pages are 'ASSETS BRAND MANUAL EDITION 12'
    followed by '<SECTION> <page number>'. That is boilerplate to drop, and
    also the only reliable source of the section name.
    """
    for i, line in enumerate(lines):
        if FOOTER.match(line):
            section = None
            if i + 1 < len(lines):
                m = SECTION_AND_PAGE.match(lines[i + 1])
                if m:
                    section = normalise_section(m.group(1))
            return lines[:i], section
    return list(lines), None


def is_heading(line):
    """Short, essentially all caps, no terminal punctuation."""
    letters = [c for c in line if c.isalpha()]
    if not letters or len(line) > 46:
        return False
    if CLAUSE_END.search(line):
        return False
    upper = sum(1 for c in letters if c.isupper()) / float(len(letters))
    return upper > 0.85


def is_spec_row(line):
    return bool(SPEC_ROW.match(line)) and len(line) <= 40


def _dedupe_preserving_order(lines):
    """Mockups repeat the same label several times on a page ('FIFA PARTNER'
    four times). Keep the first occurrence only."""
    seen = set()
    out = []
    for line in lines:
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def _flush_prose(buffer, section, page, heading, out):
    if not buffer:
        return
    text = re.sub(r"\s+", " ", " ".join(buffer)).strip()
    buffer[:] = []
    if not text or FILLER.search(text):
        return
    if len(text) < MIN_LENGTH:
        return
    # A real statement ends like one. This is what rejects caption grids such
    # as "Colour tonal 4 Colour flat Pantone solid pattern fill", which are
    # runs of layout labels rather than anything assertable.
    if not SENTENCE_END.search(text):
        return
    kind = "rule" if RULE_OPENER.match(text) else "guideline"
    out.append(Candidate(text, kind, section, page, heading))


def _split_spec_blocks(rows):
    """One colour is one run of distinct labels.

    These pages lay colours out in columns, and the extractor reads them in an
    order that runs several swatches together. A repeated label (a second
    'CMYK') is the reliable marker that the next colour has started.
    """
    blocks, current, seen = [], [], set()
    for row in rows:
        m = SPEC_LABEL.match(row)
        label = m.group(1).upper() if m else row.upper()
        if label in seen:
            blocks.append(current)
            current, seen = [], set()
        seen.add(label)
        current.append(row)
    if current:
        blocks.append(current)
    return blocks


def _flush_spec(heading_run, rows, section, page, out):
    """Emit one memory per colour.

    These pages are laid out in columns, and the extractor reads every swatch
    name first and every value block after, so the names arrive as a run:
    WHITE, BLACK, then white's values, then black's. When the two counts match,
    pair them positionally. When they do not -- the reading order was more
    scrambled than that -- fall back to the joined heading with an index, which
    is at least honest about the uncertainty rather than mislabelling a colour.
    """
    if not heading_run or len(rows) < 2:
        return
    names = [despace(h) for h in heading_run]
    blocks = [b for b in _split_spec_blocks(rows) if len(b) >= 2]
    if not blocks:
        return

    paired = len(names) == len(blocks)
    joined = " ".join(names)
    for i, block in enumerate(blocks):
        if paired:
            label, confidence = names[i], 1.0
        elif len(blocks) == 1:
            label, confidence = joined, 1.0
        else:
            # Names and value blocks did not line up, so which name goes with
            # which swatch is a guess. Say so instead of asserting it.
            label = "%s (%d of %d)" % (joined, i + 1, len(blocks))
            confidence = 0.4
        out.append(Candidate("%s: %s" % (label, "; ".join(block)),
                             "spec", section, page, label, confidence))


def segment_page(page):
    """Turn one RawPage into candidate memories."""
    body, section = split_footer(page.lines)
    body = _dedupe_preserving_order(body)

    out = []
    prose = []
    heading = None
    heading_run = []
    last_was_heading = False
    spec_heading_run = None
    spec_rows = []

    for line in body:
        if is_spec_row(line):
            # Spec rows belong to the run of headings just above them, which
            # are the colour names. Prose in progress ends here.
            _flush_prose(prose, section, page.number, heading, out)
            if spec_heading_run is None:
                spec_heading_run = list(heading_run)
            spec_rows.append(re.sub(r"\s+", " ", line))
            continue

        if spec_rows:
            _flush_spec(spec_heading_run, spec_rows, section, page.number, out)
            spec_heading_run, spec_rows = None, []

        if is_heading(line):
            _flush_prose(prose, section, page.number, heading, out)
            # Two different things look identical here: a heading wrapped over
            # two lines ('GETTING IT' / 'RIGHT'), and two separate headings in
            # adjacent columns ('WHITE' / 'BLACK'). Keep both readings -- the
            # joined string for prose context, the run for pairing with specs.
            if last_was_heading:
                heading = heading + " " + line
                heading_run.append(line)
            else:
                heading = line
                heading_run = [line]
            last_was_heading = True
            continue
        last_was_heading = False

        prose.append(line)
        if CLAUSE_END.search(line):
            _flush_prose(prose, section, page.number, heading, out)

    _flush_prose(prose, section, page.number, heading, out)
    if spec_rows:
        _flush_spec(spec_heading_run, spec_rows, section, page.number, out)
    return out, section


def segment(pages):
    """Segment a sequence of RawPages, carrying the section forward.

    A handful of pages (covers, full-bleed artwork) have no footer, so they
    inherit the section of the page before them.
    """
    last_section = None
    for page in pages:
        candidates, section = segment_page(page)
        if section:
            last_section = section
        for c in candidates:
            if c.section is None:
                c.section = last_section
            page.section = c.section
            yield c
