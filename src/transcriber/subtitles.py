"""Turn an aligned (and optionally diarized) transcript into subtitle cues and files."""

import copy
import json
import logging
import textwrap
from dataclasses import dataclass
from pathlib import Path

from .cleanup import drop_hallucinations, snap_speakers

log = logging.getLogger("transcriber")

MAX_LINE_CHARS = 42
MAX_LINES = 2
MAX_CUE_SECONDS = 6.0
MAX_GAP_SECONDS = 1.0
SENTENCE_END = (".", "?", "!", "…")
PARAGRAPH_GAP_SECONDS = 10.0  # in the .txt, a long silence starts a new paragraph even for the same speaker


@dataclass
class Cue:
    start: float
    end: float
    text: str
    speaker: str | None


def _words(result: dict) -> list[dict]:
    """Flatten segments into timed words, carrying each segment's speaker as a fallback.

    Words the aligner could not time (often numbers or symbols) inherit the
    previous word's timing so they stay in the right cue.
    """
    words = []
    for seg in result["segments"]:
        seg_words = seg.get("words") or [
            {"word": seg["text"].strip(), "start": seg["start"], "end": seg["end"]}
        ]
        for w in seg_words:
            text = w.get("word", "").strip()
            if not text:
                continue
            start = w.get("start", words[-1]["end"] if words else seg["start"])
            end = w.get("end", start)
            words.append(
                {"text": text, "start": start, "end": end, "speaker": w.get("speaker", seg.get("speaker"))}
            )
    return words


def build_cues(result: dict, names: dict[str, str] | None = None) -> list[Cue]:
    """Group words into readable cues: one speaker per cue, at most two lines, a few seconds long."""
    names = names or {}
    cues: list[Cue] = []
    current: list[dict] = []

    def flush():
        if current:
            speaker = current[0]["speaker"]
            cues.append(
                Cue(
                    start=current[0]["start"],
                    end=current[-1]["end"],
                    text=" ".join(w["text"] for w in current),
                    speaker=names.get(speaker, speaker),
                )
            )
            current.clear()

    for w in _words(result):
        if current:
            prev = current[-1]
            text_len = len(" ".join(x["text"] for x in current)) + 1 + len(w["text"])
            if (
                w["speaker"] != current[0]["speaker"]
                or w["start"] - prev["end"] > MAX_GAP_SECONDS
                or w["end"] - current[0]["start"] > MAX_CUE_SECONDS
                or text_len > MAX_LINE_CHARS * MAX_LINES
                or (prev["text"].endswith(SENTENCE_END) and text_len > MAX_LINE_CHARS)
            ):
                flush()
        current.append(w)
    flush()
    return cues


def _cue_lines(cue: Cue, prev_speaker: str | None) -> str:
    # Label only on speaker changes, so a monologue isn't cluttered with the same name on every cue.
    label = f"[{cue.speaker}] " if cue.speaker and cue.speaker != prev_speaker else ""
    return "\n".join(textwrap.wrap(label + cue.text, MAX_LINE_CHARS))


def _timestamp(seconds: float, sep: str) -> str:
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def to_srt(cues: list[Cue]) -> str:
    blocks, prev = [], None
    for i, cue in enumerate(cues, 1):
        blocks.append(
            f"{i}\n{_timestamp(cue.start, ',')} --> {_timestamp(cue.end, ',')}\n{_cue_lines(cue, prev)}\n"
        )
        prev = cue.speaker
    return "\n".join(blocks)


def to_vtt(cues: list[Cue]) -> str:
    blocks, prev = ["WEBVTT\n"], None
    for cue in cues:
        blocks.append(f"{_timestamp(cue.start, '.')} --> {_timestamp(cue.end, '.')}\n{_cue_lines(cue, prev)}\n")
        prev = cue.speaker
    return "\n".join(blocks)


def to_txt(cues: list[Cue]) -> str:
    """Readable transcript: one paragraph per speaker turn, stamped with its start time."""
    paragraphs: list[tuple[float, str | None, list[str]]] = []
    prev_end = 0.0
    for cue in cues:
        if paragraphs and paragraphs[-1][1] == cue.speaker and cue.start - prev_end < PARAGRAPH_GAP_SECONDS:
            paragraphs[-1][2].append(cue.text)
        else:
            paragraphs.append((cue.start, cue.speaker, [cue.text]))
        prev_end = cue.end
    out = []
    for start, speaker, texts in paragraphs:
        who = f"{speaker}: " if speaker else ""
        out.append(f"[{_timestamp(start, '.')[:8]}] {who}{' '.join(texts)}")
    return "\n\n".join(out) + "\n"


WRITERS = {"srt": to_srt, "vtt": to_vtt, "txt": to_txt}


def write_outputs(
    result: dict, base: Path, formats: list[str], names: dict[str, str] | None = None, clean: bool = True
) -> list[Path]:
    """Write `<base>.<fmt>` for each format.

    JSON is the raw, uncleaned result so it can be re-rendered later with different settings.
    """
    shown = copy.deepcopy(result)
    if clean:
        shown["segments"], dropped = drop_hallucinations(shown["segments"])
        for seg in dropped:
            log.info("  dropped likely hallucination at %s: %r", _timestamp(seg["start"], ".")[:8], seg["text"])
        snap_speakers(shown["segments"])
    cues = build_cues(shown, names)
    written = []
    for fmt in formats:
        path = base.with_name(f"{base.name}.{fmt}")
        if fmt == "json":
            path.write_text(json.dumps(result, ensure_ascii=False, indent=1))
        else:
            path.write_text(WRITERS[fmt](cues))
        written.append(path)
    return written
