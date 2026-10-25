"""Post-processing on the aligned transcript: drop Whisper hallucinations, snap speaker changes to clauses.

Both work on the result dict only, so they also apply when re-rendering a saved JSON.
"""

import re
import zlib
from collections import Counter

# Hallucination heuristics
REPEAT_WINDOW_SECONDS = 30.0  # the same line this many times within the window is a decoding loop
REPEAT_COUNT = 3
REPEAT_DENSITY = 0.75  # ...and makes up most of the lines in that stretch
MAX_COMPRESSION_RATIO = 2.4  # same threshold Whisper itself uses for "this text is too repetitive"
ISLAND_GAP_SECONDS = 5.0  # speech separated from everything else by at least this much silence...
ISLAND_MAX_WORDS = 2  # ...that is only a word or two...
ISLAND_MAX_SCORE = 0.5  # ...and poorly aligned to the audio, is almost always noise transcribed as words

# Not commas: "Ja, ik weet niet" is one speaker, and splitting there lets a short "Ja," flip.
CLAUSE_END = re.compile(r"[.;:?!…]$")


def _normalize(text: str) -> str:
    return re.sub(r"\W+", " ", text.lower()).strip()


def _too_compressible(text: str) -> bool:
    data = text.encode()
    return len(data) > 40 and len(data) / len(zlib.compress(data)) > MAX_COMPRESSION_RATIO


def _mean_score(segments: list[dict]) -> float:
    scores = [w["score"] for s in segments for w in s.get("words", []) if "score" in w]
    return sum(scores) / len(scores) if scores else 0.0


def drop_hallucinations(segments: list[dict]) -> tuple[list[dict], list[dict]]:
    """Return (kept, dropped) segments."""
    dropped_ids = set()

    for i, seg in enumerate(segments):
        norm = _normalize(seg["text"])
        if not norm or _too_compressible(seg["text"]):
            dropped_ids.add(i)
            continue
        nearby = [
            j for j, other in enumerate(segments)
            if abs(other["start"] - seg["start"]) <= REPEAT_WINDOW_SECONDS and _normalize(other["text"]) == norm
        ]  # fmt: skip
        # A loop is the same line over and over; "Ja." said several times in a conversation is not.
        if len(nearby) >= REPEAT_COUNT and len(nearby) / (nearby[-1] - nearby[0] + 1) >= REPEAT_DENSITY:
            dropped_ids.update(nearby)

    # Islands are judged after removing loops, since a loop can be what kept a fragment company.
    kept = [i for i in range(len(segments)) if i not in dropped_ids]
    islands: list[list[int]] = []
    for i in kept:
        if islands and segments[i]["start"] - segments[islands[-1][-1]]["end"] < ISLAND_GAP_SECONDS:
            islands[-1].append(i)
        else:
            islands.append([i])
    for island in islands:
        segs = [segments[i] for i in island]
        n_words = sum(len(s["text"].split()) for s in segs)
        if n_words <= ISLAND_MAX_WORDS and _mean_score(segs) < ISLAND_MAX_SCORE:
            dropped_ids.update(island)

    kept_segs = [s for i, s in enumerate(segments) if i not in dropped_ids]
    dropped_segs = [s for i, s in enumerate(segments) if i in dropped_ids]
    return kept_segs, dropped_segs


def snap_speakers(segments: list[dict]) -> None:
    """Give every word of a clause the same speaker, in place.

    Diarization turn boundaries and word timestamps rarely line up exactly, so word-level speakers flip a
    word or two early or late ("Wat is het | hier achter elkaar?"). A real change of speaker almost always
    falls at a sentence end, so each clause (split at . ? ! … etc.) takes the speaker that covers most of
    its duration.
    """
    for seg in segments:
        words = seg.get("words") or []
        clause: list[dict] = []
        for k, w in enumerate(words):
            clause.append(w)
            if k == len(words) - 1 or CLAUSE_END.search(w.get("word", "").strip()):
                votes: Counter = Counter()
                for cw in clause:
                    if cw.get("speaker"):
                        votes[cw["speaker"]] += max(cw.get("end", 0) - cw.get("start", 0), 0.01)
                if votes:
                    speaker = votes.most_common(1)[0][0]
                    for cw in clause:
                        cw["speaker"] = speaker
                clause = []
