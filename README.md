# transcriber

Local transcription and subtitles for any audio or video file, with speaker labels, on Apple Silicon.

```
media ──ffmpeg──▶ 16 kHz audio ──VAD (silero)──▶ Whisper large-v3-turbo (MLX, GPU)
      ──▶ wav2vec2 word alignment ──▶ pyannote speaker diarization ──▶ .srt / .vtt / .txt / .json
```

Built on [whispermlx](https://github.com/KalebJS/whispermlx), a WhisperX port that uses mlx-whisper.
Nothing leaves your machine except the one-time model downloads.

## Setup

```bash
brew install ffmpeg   # if not already installed
uv sync
```

Speaker labels need a free Hugging Face token (one time):

1. Accept the terms at <https://huggingface.co/pyannote/speaker-diarization-community-1>.
2. Create a read token at <https://huggingface.co/settings/tokens>.
3. Run `uv run hf auth login` and paste it (or `export HF_TOKEN=...`).

## Usage

```bash
uv run transcriber video.mp4                     # -> video.<lang>.srt + video.<lang>.txt
uv run transcriber ~/Videos/holiday -n 3         # whole folder, 3 speakers
uv run transcriber talk.mp3 -l es -f srt,vtt,txt,json
uv run transcriber clip.avi --no-diarize         # no speaker labels, no token needed
uv run transcriber clip.mkv --embed              # also clip.subtitled.mkv with a toggleable subtitle track
```

The language (`en`, `es`, `nl`, ...) is auto-detected per file unless you pass `-l`, and goes into the
filename (`movie.nl.srt`). VLC, Jellyfin, Plex and IINA pick up a subtitle file with the same name next to
the video automatically.

### Naming speakers

Speakers come out as `SPEAKER_00`, `SPEAKER_01`, ... Pass `--names` when you know who is who, or fix them
afterwards from the JSON without re-transcribing:

```bash
uv run transcriber interview.mp4 -f srt,txt,json
uv run transcriber --rerender interview.en.json --names SPEAKER_00=Ana,SPEAKER_01=Bob
```

`--names 0=Ana,1=Bob` is short for the same thing. A name that matches no speaker is reported, not ignored.

Telling it the number of speakers (`-n 2`, or `--min-speakers/--max-speakers`) noticeably helps accuracy.

### Fixing mistakes

`.srt` is plain text: open it in any editor (or [Subtitle Edit](https://www.nikse.dk/subtitleedit) /
Aegisub for a video-synced editor), fix the line, and reload it in the player. That's why sidecar files are
the default rather than burning subtitles into the picture.

## Notes

- Speed on an M4 Max: about 15 s for 5 minutes of audio once the models are cached (the first run downloads ~2 GB).
- Diarization works best with a few speakers taking turns. Overlapping speech, music and very short
  interjections are where it slips.
- Silence and background noise are skipped by voice-activity detection, which keeps Whisper from
  inventing text there. What still slips through is filtered when writing subtitles: lines repeated in a
  loop, overly repetitive text, and a stray word or two of poorly aligned text in a long silence. Dropped
  lines are printed, the `.json` keeps everything, and `--no-filter` turns this off.
- Speaker changes are snapped to sentence ends, so a turn boundary that's a word off doesn't split a
  sentence between two people. `--no-filter` also turns this off.
