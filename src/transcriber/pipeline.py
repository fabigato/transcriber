"""Transcription pipeline: ffmpeg decode -> Whisper (MLX) -> word alignment -> speaker diarization."""

import gc
import logging
import os

import whispermlx
from whispermlx.audio import load_audio
from whispermlx.diarize import DiarizationPipeline, assign_word_speakers

log = logging.getLogger("transcriber")

DEFAULT_MODEL = "mlx-community/whisper-large-v3-turbo"
# Torch side (VAD, alignment, diarization). Whisper itself always runs on the GPU through MLX.
DEVICE = "mps"


class Transcriber:
    """Holds the loaded models so a batch of files only pays the loading cost once."""

    def __init__(self, model: str = DEFAULT_MODEL, language: str | None = None, diarize: bool = True):
        self.language = language
        self.asr = whispermlx.load_model(model, device="cpu", language=language, vad_method="silero")
        self._align_models: dict[str, tuple] = {}
        self.diarizer = None
        if diarize:
            token = os.environ.get("HF_TOKEN") or None  # None falls back to `hf auth login` credentials
            self.diarizer = DiarizationPipeline(token=token, device=DEVICE)

    def _align_model(self, language: str):
        if language not in self._align_models:
            self._align_models[language] = whispermlx.load_align_model(language_code=language, device=DEVICE)
        return self._align_models[language]

    def transcribe(
        self,
        media_path: str,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> dict:
        # ffmpeg reads any container/codec and hands back 16 kHz mono float32.
        audio = load_audio(media_path)
        log.info("Transcribing %s (%.0f s of audio)", media_path, len(audio) / 16000)
        result = self.asr.transcribe(audio, language=self.language)
        language = result["language"]
        log.info("Language: %s", language)

        try:
            model, metadata = self._align_model(language)
            result = whispermlx.align(result["segments"], model, metadata, audio, DEVICE)
        except ValueError as e:  # no alignment model for this language: keep segment-level timing
            log.warning("Skipping word alignment: %s", e)
        result["language"] = language

        if self.diarizer:
            log.info("Diarizing speakers")
            turns = self.diarizer(
                audio, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers
            )
            result = assign_word_speakers(turns, result, fill_nearest=True)
            result["language"] = language

        gc.collect()
        return result
