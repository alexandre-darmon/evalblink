"""Draw the README's terminal visuals in the style of banner.svg.

    pip install fonttools
    python docs/assets/make_visuals.py /path/to/geist/fonts

Needs Geist-Regular.ttf, Geist-SemiBold.ttf and GeistMono-Regular.ttf (OFL,
https://github.com/vercel/geist-font). Text becomes outlines, like the banner's,
because an SVG shown through <img> can't load fonts. Glyphs are defined once and
reused, which keeps the files small. The figures are the README's text samples;
change both together.
"""

from __future__ import annotations

import sys
from html import escape
from pathlib import Path

from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

HERE = Path(__file__).resolve().parent

# The banner's palette; WARN is the one addition, for warnings and failures.
PAGE, EDGE, GLOW, GLOW_SOFT = "#F4F2F5", "#E6E3EA", "#E0DFF0", "#E9E7F2"
CARD, CARD_EDGE, RULE = "#FFFFFF", "#E3E1E8", "#EFEDF2"
INK, BODY, MUTED, ACCENT, WARN = "#0A0A0A", "#5F5F66", "#6B6B70", "#8C52EE", "#A8480C"

WIDTH, MARGIN, PAD = 1400, 48, 24
ROW, HEAD_ROW = 40, 42
SANS, BOLD, MONO = "Geist-Regular", "Geist-SemiBold", "GeistMono-Regular"
TEXT_SIZE = 19
CHAR = TEXT_SIZE * 0.6


class Canvas:
    def __init__(self, fonts, title):
        self.fonts, self.title = fonts, title
        self.glyphs, self.defs, self.body = {}, [], []

    def rect(self, x, y, w, h, fill, stroke=None, rx=0):
        # On the half pixel, so a 1px border stays crisp.
        edge = f' stroke="{stroke}"' if stroke else ""
        round_ = f' rx="{rx}"' if rx else ""
        self.body.append(
            f'<rect x="{x + 0.5}" y="{y + 0.5}" width="{w - 1}" height="{h - 1}"'
            f'{round_} fill="{fill}"{edge}/>'
        )

    def rule(self, x, y, w):
        self.body.append(
            f'<rect x="{x}" y="{y}" width="{w}" height="1" fill="{RULE}"/>'
        )

    def _glyph(self, font, size, char):
        key = font, size, char
        if key not in self.glyphs:
            face = self.fonts[font]
            name = face.getBestCmap().get(ord(char))
            if name is None:
                raise SystemExit(f"{font} has no glyph for {char!r}")
            glyph_set = face.getGlyphSet()
            k = size / face["head"].unitsPerEm
            pen = SVGPathPen(
                glyph_set, ntos=lambda v: f"{v:.2f}".rstrip("0").rstrip(".")
            )
            glyph_set[name].draw(TransformPen(pen, (k, 0, 0, -k, 0, 0)))
            gid = f"g{len(self.glyphs)}"
            if commands := pen.getCommands():
                self.defs.append(f'<path id="{gid}" d="{commands}"/>')
            self.glyphs[key] = (gid if commands else None, glyph_set[name].width * k)
        return self.glyphs[key]

    def width(self, text, font=MONO, size=TEXT_SIZE, track=0.0):
        return sum(self._glyph(font, size, c)[1] + track for c in text)

    def text(self, x, y, text, fill=BODY, font=MONO, size=TEXT_SIZE, end=False):
        # Geist is tracked in by 1 %, like the banner's tagline.
        track = 0.0 if font == MONO else -size / 100
        if end:
            x -= self.width(text, font, size, track)
        uses = []
        for char in text:
            gid, advance = self._glyph(font, size, char)
            if gid:
                uses.append(f'<use href="#{gid}" x="{x:.2f}" y="{y:.2f}"/>')
            x += advance + track
        self.body.append(f'<g fill="{fill}">{"".join(uses)}</g>')
        return x

    def warning(self, x, y, size=TEXT_SIZE):
        """A warning triangle: Geist Mono has no ⚠, and it replaces the emoji."""
        h = size * 0.82
        cx = x + h * 0.55
        self.body.append(
            f'<path d="M{cx:.1f} {y - h:.1f}L{x + h * 1.1:.1f} {y:.1f}H{x:.1f}Z" fill="none"'
            f' stroke="{WARN}" stroke-width="1.8" stroke-linejoin="round"/>'
            f'<path d="M{cx:.1f} {y - h * 0.62:.1f}V{y - h * 0.3:.1f}M{cx:.1f} {y - h * 0.15:.1f}'
            f'V{y - h * 0.12:.1f}" stroke="{WARN}" stroke-width="1.8" stroke-linecap="round"/>'
        )
        return x + h * 1.1 + 10

    def svg(self, height):
        title = escape(self.title, quote=True)
        return (
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {height}"'
            f' width="{WIDTH}" height="{height}" role="img" aria-label="{title}">'
            f"<title>{title}</title><defs>"
            f'<radialGradient id="glow" gradientUnits="userSpaceOnUse" cx="1320" cy="-60" r="760">'
            f'<stop offset="0" stop-color="{GLOW}"/>'
            f'<stop offset="0.55" stop-color="{GLOW_SOFT}" stop-opacity="0.6"/>'
            f'<stop offset="1" stop-color="{PAGE}" stop-opacity="0"/></radialGradient>'
            f"{''.join(self.defs)}</defs>"
            f'<rect x="0.5" y="0.5" width="{WIDTH - 1}" height="{height - 1}" rx="16"'
            f' fill="{PAGE}" stroke="{EDGE}"/>'
            f'<rect x="1" y="1" width="{WIDTH - 2}" height="{height - 2}" rx="15" fill="url(#glow)"/>'
            f"{''.join(self.body)}</svg>\n"
        )


def header(c, command):
    """The banner's top lines: its logo mark, smaller, and a command prompt."""
    c.body.append(
        f'<g transform="translate({MARGIN} 40) scale(0.625)">'
        f'<rect x="0.5" y="0.5" width="63" height="63" rx="14" fill="{CARD}" stroke="{CARD_EDGE}"/>'
        f'<path d="M19 20l12 12-12 12" fill="none" stroke="{INK}" stroke-width="6"'
        f' stroke-linecap="square" stroke-linejoin="miter"/>'
        f'<rect x="39" y="20" width="10" height="24" fill="{ACCENT}"/></g>'
    )
    c.text(MARGIN + 56, 67, f"$ {command}", MUTED, size=20)
    return 104


def table(c, y, title, columns, rows, marked=None, width=None, x=MARGIN, note=None):
    """A white card. ``columns`` are ``(label, chars, right_aligned)``."""
    width = width or WIDTH - 2 * MARGIN
    top = HEAD_ROW + 2 if title else 0
    if note:
        top += 26
    height = top + HEAD_ROW + ROW * len(rows) + 8
    c.rect(x, y, width, height, CARD, CARD_EDGE, 14)
    inner = x + PAD
    baseline = y
    if title:
        baseline += 34
        c.text(inner, baseline, title, INK, BOLD, 21)
        if note:
            baseline += 26
            c.text(inner, baseline, note, MUTED, size=16)
        baseline += 12
    free = width - 2 * PAD - sum(n for _, n, _ in columns) * CHAR
    gap = free / max(1, len(columns) - 1)
    lefts, cursor = [], inner
    for _, chars, _ in columns:
        lefts.append(cursor)
        cursor += chars * CHAR + gap
    baseline += 30
    for (label, chars, right), left in zip(columns, lefts):
        c.text(
            left + chars * CHAR if right else left, baseline, label, MUTED, end=right
        )
    baseline += 12
    c.rule(x + 1, baseline, width - 2)
    for i, row in enumerate(rows):
        if i:
            c.rule(inner, baseline + ROW * i, width - 2 * PAD)
        cell_y = baseline + ROW * i + 27
        if i == marked:  # the logo's purple bar marks what the row is about
            c.body.append(
                f'<rect x="{x}" y="{baseline + ROW * i + 8}" width="5" height="{ROW - 16}"'
                f' fill="{ACCENT}"/>'
            )
        for value, (_, chars, right), left in zip(row, columns, lefts):
            fill = INK if i == marked else BODY
            if isinstance(value, tuple):
                value, fill = value
            if value == "⚠":
                c.warning(left + chars * CHAR - TEXT_SIZE, cell_y)
                continue
            c.text(
                left + chars * CHAR if right else left, cell_y, value, fill, end=right
            )
    return y + height


def recommendation(c, x, y, width, picks, warnings):
    """A white card: one line per pick (label, then value), then the warnings.

    A pick's value may be ``(text, colour)``. Returns the card's bottom edge.
    """
    height = 44 + 36 * len(picks) + 12 + 36 * len(warnings) + 8
    c.rect(x, y, width, height, CARD, CARD_EDGE, 14)
    c.text(x + PAD, y + 34, "Recommendation", INK, BOLD, 21)
    line_y = y + 44
    for label, value in picks:
        fill = INK
        if isinstance(value, tuple):
            value, fill = value
        c.text(x + PAD, line_y + 26, label, MUTED, size=16)
        c.text(x + PAD + 150, line_y + 26, value, fill)
        line_y += 36
    c.rule(x + PAD, line_y + 6, width - 2 * PAD)
    line_y += 12
    for warning in warnings:
        left = c.warning(x + PAD, line_y + 25)
        c.text(left, line_y + 26, warning, WARN)
        line_y += 36
    return y + height


def verdict(c, y, label, text):
    """A closing line: a pill in the accent colour, then what it means."""
    pill = c.width(label, MONO, 18) + 28
    c.rect(MARGIN, y, pill, 34, ACCENT, rx=8)
    c.text(MARGIN + 14, y + 23, label, CARD, size=18)
    c.text(MARGIN + pill + 16, y + 23, text, BODY)
    return y + 34


def run_output(fonts):
    c = Canvas(
        fonts,
        "evalblink run: a progress line per case, a results table of six model and"
        " prompt combinations with score and 95% interval, cost and p50 latency,"
        " quality by tag, a recommendation naming the combos within noise and the"
        " fastest one, and a PASS verdict.",
    )
    y = header(c, "evalblink run bench.yaml") + 6
    c.text(
        MARGIN,
        y,
        "Running 300 cases (3 models × 2 prompts × 50 test cases), 5 at a time",
        MUTED,
    )
    y += 30
    for line, outcome, fill, latency in (
        ("[  1/300] openai/gpt-4o-mini / v1 / conv_001", "pass", BODY, "0.58s"),
        ("[  2/300] openai/gpt-4o-mini / v1 / conv_002", "fail", WARN, "0.66s"),
    ):
        left = c.text(c.text(MARGIN, y, line + "  ", BODY), y, outcome, fill)
        c.text(left, y, "  " + latency, MUTED)
        y += 28
    c.text(MARGIN, y, "…", MUTED)
    y += 28
    left = c.text(MARGIN, y, "[ 48/300] openai/gpt-4o-mini / v1 / conv_048  ", BODY)
    left = c.warning(left, y - 1)
    c.text(
        left,
        y,
        "API error: OpenRouter request timed out after 4 attempt(s)",
        WARN,
        size=17,
    )
    y += 28
    c.text(MARGIN, y, "…", MUTED)
    y += 22
    y = table(
        c,
        y,
        "Customer Support Classification",
        (
            ("Model", 28, False),
            ("Prompt", 6, False),
            ("Score (95% CI)", 14, True),
            ("Pass/Scored", 11, True),
            ("Errors", 6, True),
            ("Tokens in/out", 13, True),
            ("Cost", 9, True),
            ("p50 latency", 11, True),
        ),
        (
            (
                "openai/gpt-4o-mini",
                "v1",
                "87.8% (76–94)",
                "43/49",
                "1",
                "4018/294",
                "$0.000784",
                "0.61 s",
            ),
            (
                "openai/gpt-4o-mini",
                "v2",
                "90.0% (79–96)",
                "45/50",
                "0",
                "5000/300",
                "$0.000960",
                "0.71 s",
            ),
            (
                "anthropic/claude-sonnet-4.6",
                "v1",
                "94.0% (84–98)",
                "47/50",
                "0",
                "4100/300",
                "$0.017500",
                "1.54 s",
            ),
            (
                "anthropic/claude-sonnet-4.6",
                "v2",
                "96.0% (87–99)",
                "48/50",
                "0",
                "5000/300",
                "$0.021000",
                "1.53 s",
            ),
            (
                "mistralai/mistral-small-2603",
                "v1",
                "85.4% (73–93)",
                "41/48",
                "2",
                "3936/288",
                "$0.000768",
                "0.52 s",
            ),
            (
                "mistralai/mistral-small-2603",
                "v2",
                "88.0% (76–94)",
                "44/50",
                "0",
                "5000/300",
                "$0.000960",
                "0.50 s",
            ),
        ),
        marked=3,
        note="2026-06-10_143205_customer-support-classification",
    )
    y += 16
    half = (WIDTH - 2 * MARGIN - 16) // 2
    tags_bottom = table(
        c,
        y,
        "Quality by tag",
        (
            ("Tag", 11, False),
            ("Cases", 5, True),
            ("Quality", 7, True),
            ("Errors", 6, True),
            ("", 2, True),
        ),
        (
            ("edge_case", "5", ("60%", WARN), "0", "⚠"),
            ("non_english", "4", ("50%", WARN), "0", "⚠"),
            ("billing", "12", "100%", "0", ""),
            ("easy", "30", "100%", "0", ""),
            ("medium", "15", "100%", "0", ""),
        ),
        width=half,
        note="best combo: anthropic/claude-sonnet-4.6 / v2",
    )
    y = recommendation(
        c,
        MARGIN,
        tags_bottom + 16,
        WIDTH - 2 * MARGIN,
        (
            ("Best quality", "anthropic/claude-sonnet-4.6 / v2 (96.0%, 95% CI 87–99%)"),
            (
                "Within noise",
                (
                    (
                        "anthropic/claude-sonnet-4.6 / v1 (94.0%),"
                        " openai/gpt-4o-mini / v2 (90.0%) and 3 more"
                    ),
                    WARN,
                ),
            ),
            ("Best value", "openai/gpt-4o-mini / v1 (87.8% · $0.0008/run)"),
            ("Fastest", "mistralai/mistral-small-2603 / v2 (p50 0.50 s · 88.0%)"),
        ),
        (
            "3 pipeline error(s) excluded from scoring.",
            "edge_case quality is below 70%.",
            "non_english quality is below 70%.",
        ),
    )
    y += 32
    left = c.warning(MARGIN, y - 1)
    c.text(
        left,
        y,
        "3 of 300 case(s) hit a pipeline error (API / judge failure) and are excluded"
        " from scores.",
        WARN,
        size=18,
    )
    y += 18
    y = verdict(c, y, "PASS", "best score meets quality_threshold (85)")
    return c.svg(y + 40)


def dry_run(fonts):
    c = Canvas(
        fonts,
        "evalblink run --dry-run: estimated tokens and cost per model and prompt,"
        " $0.089304 in total, within the $10 budget, with no API calls.",
    )
    y = header(c, "evalblink run my_benchmark.yaml --dry-run")
    y = table(
        c,
        y - 10,
        "Dry run: estimated cost (no API calls)",
        (
            ("Model", 28, False),
            ("Prompt", 6, False),
            ("Est prompt tok", 14, True),
            ("Est completion tok", 18, True),
            ("Est cost", 9, True),
        ),
        (
            ("openai/gpt-4o-mini", "v1", "191", "800", "$0.010830"),
            ("openai/gpt-4o-mini", "v2", "246", "800", "$0.010843"),
            ("anthropic/claude-sonnet-4.6", "v1", "191", "800", "$0.022895"),
            ("anthropic/claude-sonnet-4.6", "v2", "246", "800", "$0.023064"),
            ("mistralai/mistral-small-2603", "v1", "191", "800", "$0.010830"),
            ("mistralai/mistral-small-2603", "v2", "246", "800", "$0.010843"),
        ),
    )
    y += 34
    c.text(MARGIN, y, "Judge cost (llm_judge cases)", MUTED)
    c.text(WIDTH - MARGIN, y, "$0.061942", BODY, end=True)
    y += 32
    c.text(MARGIN, y, "Estimated total cost", INK, BOLD, 21)
    c.text(WIDTH - MARGIN, y, "$0.089304", INK, end=True)
    y += 28
    c.text(
        MARGIN,
        y,
        "Estimate only — completion tokens assume max_tokens (capped at each model's"
        " output limit); prompt tokens ≈ chars/4.",
        MUTED,
        size=16,
    )
    y += 20
    y = verdict(c, y, "WITHIN BUDGET", "estimate $0.089304 ≤ max_cost_usd $10.000000")
    return c.svg(y + 40)


def compare(fonts):
    c = Canvas(
        fonts,
        "evalblink compare: quality, cost and p50 latency deltas between two runs"
        " per model and prompt. Run B improves quality on 2 of 3 shared combos,"
        " both within noise.",
    )
    y = header(c, "evalblink compare <run_a>.json <run_b>.json")
    y = table(
        c,
        y - 10,
        "Delta",
        (
            ("Model", 28, False),
            ("Prompt", 6, False),
            ("Quality ∆", 9, False),  # U+2206: Geist has no Greek Δ
            ("Cost ∆", 10, False),
            ("p50 latency ∆", 13, False),
            ("Notes", 12, False),
        ),
        (
            (
                "anthropic/claude-sonnet-4.6",
                "v1",
                ("+4.0% ↑", INK),
                "+0.0002 ↑",
                "-0.08 s ↓",
                ("within noise", WARN),
            ),
            (
                "openai/gpt-4o-mini",
                "v1",
                ("+6.0% ↑", INK),
                "stable",
                "stable",
                ("within noise", WARN),
            ),
            ("anthropic/claude-sonnet-4.6", "v2", "stable", "stable", "-0.12 s ↓", ""),
        ),
        note="2026-06-08_101500_customer-support-classification → 2026-06-10_143205_customer-support-classification",
    )
    y += 20
    y = verdict(
        c,
        y,
        "VERDICT",
        "Run B improves quality on 2/3 shared combo(s). None of these gains is"
        " beyond noise (95%).",
    )
    return c.svg(y + 40)


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    fonts = {
        name: TTFont(Path(sys.argv[1]) / f"{name}.ttf") for name in (SANS, BOLD, MONO)
    }
    for name, draw in (
        ("run-output", run_output),
        ("dry-run", dry_run),
        ("compare", compare),
    ):
        path = HERE / f"{name}.svg"
        path.write_text(draw(fonts), encoding="utf-8")
        print(f"{path.name}: {path.stat().st_size // 1024} KB")


if __name__ == "__main__":
    main()
