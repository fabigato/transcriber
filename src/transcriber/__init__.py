"""transcriber: local, speaker-labelled transcripts and subtitles for any audio/video file."""

import argparse
import json
import logging
import subprocess
import sys
import warnings
from pathlib import Path

MEDIA_EXTENSIONS = {
    ".mp3", ".m4a", ".aac", ".wav", ".flac", ".ogg", ".opus", ".wma",
    ".mp4", ".m4v", ".mkv", ".mov", ".avi", ".webm", ".wmv", ".mpg", ".mpeg", ".ts", ".3gp",
}  # fmt: skip
FORMATS = ["srt", "vtt", "txt", "json"]

log = logging.getLogger("transcriber")


def find_media(inputs: list[str]) -> list[Path]:
    files = []
    for item in map(Path, inputs):
        if item.is_dir():
            files += sorted(p for p in item.rglob("*") if p.suffix.lower() in MEDIA_EXTENSIONS)
        elif item.is_file():
            files.append(item)
        else:
            log.error("Not found: %s", item)
    return files


def parse_names(spec: str | None) -> dict[str, str]:
    """'SPEAKER_00=Ana,SPEAKER_01=Bob' -> {'SPEAKER_00': 'Ana', 'SPEAKER_01': 'Bob'}"""
    if not spec:
        return {}
    return dict(pair.split("=", 1) for pair in spec.split(","))


def embed_subtitles(media: Path, srt: Path, language: str, out_dir: Path) -> Path:
    """Copy the media into an .mkv with the subtitles as a toggleable track. The original is untouched."""
    out = out_dir / f"{media.stem}.subtitled.mkv"
    lang3 = {"en": "eng", "es": "spa", "nl": "dut"}.get(language, language)
    subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-i", str(media), "-i", str(srt),
         "-map", "0", "-map", "1", "-c", "copy", "-c:s", "srt",
         "-metadata:s:s:0", f"language={lang3}", str(out)],
        check=True,
    )  # fmt: skip
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="transcriber",
        description="Transcribe audio/video locally with speaker labels. Writes subtitles next to each "
        "file as <name>.<lang>.srt so VLC, Jellyfin, Plex etc. pick them up automatically.",
    )
    parser.add_argument("inputs", nargs="+", help="media files or folders (searched recursively)")
    parser.add_argument("-l", "--language", help="e.g. en, es, nl (default: auto-detect per file)")
    parser.add_argument("-f", "--formats", default="srt,txt", help=f"comma-separated, from {','.join(FORMATS)}")
    parser.add_argument("-o", "--output-dir", type=Path, help="default: next to each input file")
    parser.add_argument("-n", "--speakers", type=int, help="exact number of speakers, if known (improves accuracy)")
    parser.add_argument("--min-speakers", type=int)
    parser.add_argument("--max-speakers", type=int)
    parser.add_argument("--no-diarize", action="store_true", help="skip speaker labelling (no HF token needed)")
    parser.add_argument("--names", help="rename speakers, e.g. SPEAKER_00=Ana,SPEAKER_01=Bob")
    parser.add_argument("--embed", action="store_true", help="also write <name>.subtitled.mkv with a soft subtitle track")
    parser.add_argument("--rerender", action="store_true",
                        help="inputs are .json transcripts from an earlier run: rewrite outputs (e.g. with --names) without re-transcribing")  # fmt: skip
    parser.add_argument("--model", default=None, help="MLX Whisper model (default: whisper-large-v3-turbo)")
    parser.add_argument("--skip-existing", action="store_true", help="skip files that already have outputs")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    # Our own progress at INFO; third-party libraries only when something goes wrong (or with -v).
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    log.setLevel(logging.INFO)
    if not args.verbose:
        warnings.filterwarnings("ignore")

    formats = [f.strip() for f in args.formats.split(",")]
    if bad := set(formats) - set(FORMATS):
        parser.error(f"unknown format(s): {', '.join(bad)}")
    names = parse_names(args.names)
    from .subtitles import write_outputs

    if args.rerender:
        for path in map(Path, args.inputs):
            result = json.loads(path.read_text())
            base = path.with_suffix("")  # movie.en.json -> movie.en
            for out in write_outputs(result, base, [f for f in formats if f != "json"], names):
                log.info("  wrote %s", out)
        return

    files = find_media(args.inputs)
    if not files:
        sys.exit("No media files found.")

    import whispermlx

    from .pipeline import DEFAULT_MODEL, Transcriber

    whispermlx.setup_logging(level="info" if args.verbose else "error")

    try:
        engine = Transcriber(args.model or DEFAULT_MODEL, args.language, diarize=not args.no_diarize)
    except Exception as e:
        if args.no_diarize:
            raise
        sys.exit(
            f"Could not load the speaker diarization model ({e.__class__.__name__}: {e}).\n"
            "It needs a free Hugging Face token: accept the terms at "
            "https://huggingface.co/pyannote/speaker-diarization-community-1, then run "
            "`uv run hf auth login` or export HF_TOKEN=... Or pass --no-diarize."
        )

    failed = 0
    for i, media in enumerate(files, 1):
        out_dir = args.output_dir or media.parent
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.skip_existing and list(out_dir.glob(f"{media.stem}.*.{formats[0]}")):
            log.info("[%d/%d] Skipping %s (already done)", i, len(files), media.name)
            continue
        log.info("[%d/%d] %s", i, len(files), media.name)
        try:
            result = engine.transcribe(
                str(media), args.speakers, args.min_speakers, args.max_speakers
            )
            base = out_dir / f"{media.stem}.{result['language']}"
            written = write_outputs(result, base, formats, names)
            if args.embed and media.suffix.lower() not in {".mp3", ".m4a", ".wav", ".flac", ".ogg", ".opus"}:
                srt = base.with_name(base.name + ".srt")
                if srt not in written:
                    written += write_outputs(result, base, ["srt"], names)
                written.append(embed_subtitles(media, srt, result["language"], out_dir))
            for out in written:
                log.info("  wrote %s", out)
        except Exception as e:
            failed += 1
            log.error("  failed: %s: %s", e.__class__.__name__, e)
            if args.verbose:
                raise
    if failed:
        sys.exit(f"{failed} of {len(files)} file(s) failed.")
