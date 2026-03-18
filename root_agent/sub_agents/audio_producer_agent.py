# audio_producer_agent.py
# Converts a formatted two-host podcast script into a WAV audio file using
# Gemini TTS via the Cloud Text-to-Speech client with MultiSpeakerMarkup
# structured turns. Supports writing output locally, to GCS, or as an ADK
# artifact.

import io
import json
import logging
import os
import re
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.cloud import texttospeech
from google.genai import types  # still used for artifact Blob

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "agent_configuration.json"

with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

# ---------------------------------------------------------------------------
# Model Configuration
# ---------------------------------------------------------------------------

TTS_MODEL            = _CFG["models"]["tts"]          # e.g. "gemini-2.5-flash-preview-tts"
AUDIO_PRODUCER_MODEL = _CFG["models"]["audio_producer"]

# ---------------------------------------------------------------------------
# Podcast Speaker Setup
# ---------------------------------------------------------------------------

HOST_1 = _CFG["hosts"]["host_1"]
HOST_2 = _CFG["hosts"]["host_2"]

# FLATTENED DESCRIPTIONS: Removed markdown and line breaks to prevent TTS read-aloud
ALEX_DESCRIPTION = (
    f"{HOST_1['name']} is the curious host. Personality: {HOST_1['description']}. "
    "Voice: Warm, upbeat, and accessible. Often introduces topics and asks clarifying questions."
)

JORDAN_DESCRIPTION = (
    f"{HOST_2['name']} is the expert host. Personality: {HOST_2['description']}. "
    "Voice: Calm authority with occasional dry wit. Brings data and history, and sometimes pauses to think."
)

# ---------------------------------------------------------------------------
# Audio Output Configuration
# ---------------------------------------------------------------------------

_AUDIO  = _CFG["audio"]
_OUTPUT = _CFG["output"]

AUDIO_SAMPLE_RATE  = _AUDIO["sample_rate"]
AUDIO_CHANNELS     = _AUDIO["channels"]
AUDIO_SAMPLE_WIDTH = _AUDIO["sample_width"]
CHUNK_TARGET_WORDS = _AUDIO["chunk_target_words"]

OUTPUT_DIR        = _OUTPUT["local_output_dir"]
GCS_OUTPUT_BUCKET = _OUTPUT["gcs_output_bucket"]
OUTPUT_MODE       = _OUTPUT.get("output_mode", "gcs")  # "artifact" | "gcs" | "local"

# Cloud TTS model name used in VoiceSelectionParams.
# The TTS_MODEL value from config is a Gemini model string (e.g.
# "gemini-2.5-flash-preview-tts"); the Cloud TTS API expects the equivalent
# short form (e.g. "gemini-2.5-flash-tts").  We strip the "-preview" infix so
# both config formats work transparently.
_TTS_API_MODEL = TTS_MODEL.replace("-preview-tts", "-tts")

# ---------------------------------------------------------------------------
# Cloud TTS Client
# ---------------------------------------------------------------------------

tts_client = texttospeech.TextToSpeechClient()

# ---------------------------------------------------------------------------
# Director's Notes Builder
# ---------------------------------------------------------------------------

def _build_director_notes(
    podcast_title: str,
    style_guidance: Optional[str],
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    Builds the Director's Notes prompt that is passed as the `prompt` field
    of SynthesisInput. Formatted as a dense paragraph without Markdown to
    prevent the Gemini-TTS model from mistakenly reading the text aloud.
    """
    continuation_note = ""
    if total_chunks > 1:
        if chunk_index == 0:
            continuation_note = "This is the opening segment."
        elif chunk_index == total_chunks - 1:
            continuation_note = "This is the closing segment."
        else:
            continuation_note = f"This is segment {chunk_index + 1} of {total_chunks}. The hosts are mid-discussion."

    extra_style = f"Additional style guidance: {style_guidance}. " if style_guidance else ""

    # FLATTENED PROMPT: A single conversational block prevents "script reading" behavior
    return (
        f"System Instructions (Do not read aloud): You are producing a podcast episode titled '{podcast_title}'. "
        f"The scene is a modern podcast studio where two knowledgeable hosts are having a focused but relaxed conversation. "
        f"{continuation_note} "
        f"{ALEX_DESCRIPTION} "
        f"{JORDAN_DESCRIPTION} "
        f"Delivery rules: Both hosts must sound completely natural and conversational. "
        f"Perform inline markers: [pause] as a brief thinking pause, [laughs] as a light laugh, and [sighs] as an exhale. "
        f"Use natural hesitations and contractions. The pace should be 130 to 150 words per minute. "
        f"{extra_style}"
        f"CRITICAL: These are backend director notes. Absolutely no part of this text should be spoken aloud."
    )

# ---------------------------------------------------------------------------
# Script Parser — text → structured Turn list
# ---------------------------------------------------------------------------

def _parse_script_to_turns(
    script: str,
) -> list[texttospeech.MultiSpeakerMarkup.Turn]:
    """
    Parses a formatted script into a list of MultiSpeakerMarkup.Turn objects.

    Expects lines in the form:
        Alex: Some dialogue here...
        Jordan: Response here...

    Consecutive lines by the same speaker are merged into a single turn.
    Inline stage markers ([pause], [laughs], [sighs]) are preserved in the
    text — the TTS model interprets them via the director's notes prompt.
    """
    host_names = {HOST_1["name"], HOST_2["name"]}
    # Match "SpeakerName: dialogue" — speaker name may contain spaces
    turn_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in host_names) + r"):\s*(.+)",
        re.MULTILINE,
    )

    turns: list[texttospeech.MultiSpeakerMarkup.Turn] =[]
    for match in turn_pattern.finditer(script):
        speaker = match.group(1)
        text    = match.group(2).strip()
        if not text:
            continue
        # Merge into previous turn if the same speaker continues
        if turns and turns[-1].speaker == speaker:
            turns[-1] = texttospeech.MultiSpeakerMarkup.Turn(
                speaker=speaker,
                text=turns[-1].text + " " + text,
            )
        else:
            turns.append(
                texttospeech.MultiSpeakerMarkup.Turn(speaker=speaker, text=text)
            )

    if not turns:
        raise ValueError(
            "No speaker turns found in script. "
            f"Expected lines starting with '{HOST_1['name']}:' or '{HOST_2['name']}:'."
        )

    logger.info("Parsed %d speaker turns from script.", len(turns))
    return turns


# ---------------------------------------------------------------------------
# Turn-based Chunking
# ---------------------------------------------------------------------------

def _chunk_turns(
    turns: list[texttospeech.MultiSpeakerMarkup.Turn],
    target_words: int,
) -> list[list[texttospeech.MultiSpeakerMarkup.Turn]]:
    """
    Splits a turn list into chunks, each roughly target_words words long.
    Splits only at turn boundaries so no turn is ever cut mid-sentence.
    """
    total_words = sum(len(t.text.split()) for t in turns)
    if total_words <= target_words:
        return [turns]

    chunks: list[list[texttospeech.MultiSpeakerMarkup.Turn]] = []
    current: list[texttospeech.MultiSpeakerMarkup.Turn] =[]
    current_words = 0

    for turn in turns:
        turn_words = len(turn.text.split())
        if current_words + turn_words > target_words and current:
            chunks.append(current)
            current       = [turn]
            current_words = turn_words
        else:
            current.append(turn)
            current_words += turn_words

    if current:
        chunks.append(current)

    logger.info(
        "Split %d turns into %d chunk(s) (total ~%d words)",
        len(turns), len(chunks), total_words,
    )
    return chunks


# ---------------------------------------------------------------------------
# WAV Helpers
# ---------------------------------------------------------------------------

def _build_wav_bytes(
    pcm_data: bytes,
    channels: int = AUDIO_CHANNELS,
    rate: int = AUDIO_SAMPLE_RATE,
    sample_width: int = AUDIO_SAMPLE_WIDTH,
) -> bytes:
    """Wraps raw PCM bytes in a WAV header and returns the result as bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


def _write_wav_local(path: str, wav_bytes: bytes) -> None:
    """Writes WAV bytes to a local file path."""
    with open(path, "wb") as f:
        f.write(wav_bytes)


def _write_wav_gcs(gcs_path: str, wav_bytes: bytes) -> str:
    """
    Uploads WAV bytes to Google Cloud Storage.

    Args:
        gcs_path:  Full GCS destination URI (e.g. gs://bucket/file.wav).
        wav_bytes: WAV-headered audio bytes to upload.

    Returns:
        The authenticated GCS URL of the uploaded file.
    """
    from google.cloud import storage

    path_parts  = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name   = path_parts[1] if len(path_parts) > 1 else ""

    storage_client = storage.Client()
    bucket         = storage_client.bucket(bucket_name)
    blob           = bucket.blob(blob_name)

    generation_match_precondition = 0
    blob.upload_from_string(
        wav_bytes,
        content_type="audio/wav",
        if_generation_match=generation_match_precondition,
    )

    authenticated_url = f"https://storage.cloud.google.com/{bucket_name}/{blob_name}"
    logger.info("Audio uploaded to %s", authenticated_url)
    return authenticated_url


# ---------------------------------------------------------------------------
# Podcast Generation Function Tool
# ---------------------------------------------------------------------------

async def generate_podcast_audio(
    script: str,
    podcast_title: str,
    write_to_gcs: bool = False,
    gcs_output_bucket: Optional[str] = None,
    output_dir: Optional[str] = None,
    style_guidance: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Converts a formatted two-host podcast script to a WAV audio file using
    the Cloud Text-to-Speech API with MultiSpeakerMarkup structured turns.

    The script is parsed into explicit Turn objects so dialogue and director's
    notes are fully separated — the model is never at risk of speaking the
    prompt instructions.

    Output destination is controlled by output_mode in agent_configuration.json:
      - "artifact"  Save as an ADK artifact (presented as a download link in the UI).
      - "gcs"       Upload to Google Cloud Storage (write_to_gcs must be True).
      - "local"     Write to a local directory on disk.

    Args:
        script:            Formatted script with speaker labels matching host names
                           defined in agent_configuration.json.
        podcast_title:     Episode title (used for filename and TTS prompt context).
        write_to_gcs:      Legacy flag — still honoured when output_mode is "gcs".
        gcs_output_bucket: GCS bucket URI to upload to when output_mode is "gcs".
        output_dir:        Local directory to write the WAV when output_mode is "local".
        style_guidance:    Optional extra style notes included in director's notes.
        tool_context:      Injected automatically by ADK. Required for artifact output.

    Returns:
        A dict with keys:
        - success (bool)
        - output_path (str)
        - output_url (str)
        - artifact_saved (bool)
        - duration_seconds (int)
        - duration_minutes (float)
        - chunks_processed (int)
        - model_used (str)
        - error (str)
    """
    try:
        safe_title      = re.sub(r"[^\w\-_]", "_", podcast_title)[:50]
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{safe_title}_{timestamp}.wav"

        # -- 1. Parse script into structured turns ---------------------------
        all_turns = _parse_script_to_turns(script)

        # -- 2. Split into chunks at turn boundaries -------------------------
        turn_chunks = _chunk_turns(all_turns, CHUNK_TARGET_WORDS)
        logger.info(
            "Processing %d TTS chunk(s) for '%s'", len(turn_chunks), podcast_title
        )

        # -- 3. Voice configuration (shared across all chunks) ---------------
        voice = texttospeech.VoiceSelectionParams(
            language_code="en-US",
            model_name=_TTS_API_MODEL,
            multi_speaker_voice_config=texttospeech.MultiSpeakerVoiceConfig(
                speaker_voice_configs=[
                    texttospeech.MultispeakerPrebuiltVoice(
                        speaker_alias=HOST_1["name"],
                        speaker_id=HOST_1["voice"],
                    ),
                    texttospeech.MultispeakerPrebuiltVoice(
                        speaker_alias=HOST_2["name"],
                        speaker_id=HOST_2["voice"],
                    ),
                ]
            ),
        )

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=AUDIO_SAMPLE_RATE,
        )

        # -- 4. Synthesise each chunk ----------------------------------------
        all_audio_frames: list[bytes] =[]
        chunks_processed = 0

        for i, turns in enumerate(turn_chunks):
            logger.info("  Generating audio for chunk %d/%d...", i + 1, len(turn_chunks))

            director_notes = _build_director_notes(
                podcast_title=podcast_title,
                style_guidance=style_guidance,
                chunk_index=i,
                total_chunks=len(turn_chunks),
            )

            synthesis_input = texttospeech.SynthesisInput(
                multi_speaker_markup=texttospeech.MultiSpeakerMarkup(turns=turns),
                prompt=director_notes,
            )

            response = tts_client.synthesize_speech(
                input=synthesis_input,
                voice=voice,
                audio_config=audio_config,
            )

            all_audio_frames.append(response.audio_content)
            chunks_processed += 1

        # -- 5. Combine raw PCM frames and wrap in WAV header ----------------
        # LINEAR16 from Cloud TTS is raw 16-bit signed little-endian PCM —
        # no WAV header included, same as the genai SDK inline_data output.
        combined_pcm = b"".join(all_audio_frames)
        wav_bytes    = _build_wav_bytes(combined_pcm)

        bytes_per_second = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        duration_seconds = len(combined_pcm) // bytes_per_second

        # -- 6. Output routing -----------------------------------------------
        effective_mode = OUTPUT_MODE
        if write_to_gcs and effective_mode not in ("artifact", "gcs"):
            effective_mode = "gcs"

        if effective_mode == "artifact":
            if tool_context is None:
                raise RuntimeError(
                    "output_mode is 'artifact' but tool_context is None. "
                    "Ensure the Runner is configured with an ArtifactService."
                )

            artifact_part = types.Part(
                inline_data=types.Blob(
                    data=wav_bytes,
                    mime_type="audio/wav",
                )
            )
            version = await tool_context.save_artifact(
                filename=output_filename,
                artifact=artifact_part,
            )
            logger.info(
                "Audio saved as ADK artifact '%s' version %d (%.1f minutes)",
                output_filename, version, duration_seconds / 60,
            )
            return {
                "success":          True,
                "output_path":      output_filename,
                "output_url":       "",
                "artifact_saved":   True,
                "artifact_version": version,
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed,
                "model_used":       _TTS_API_MODEL,
                "error":            "",
            }

        elif effective_mode == "gcs":
            bucket     = (gcs_output_bucket or GCS_OUTPUT_BUCKET).rstrip("/")
            gcs_uri    = f"{bucket}/{output_filename}"
            output_url = _write_wav_gcs(gcs_uri, wav_bytes)

            logger.info(
                "Audio uploaded to %s (%.1f minutes)", gcs_uri, duration_seconds / 60
            )
            return {
                "success":          True,
                "output_path":      gcs_uri,
                "output_url":       output_url,
                "artifact_saved":   False,
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed,
                "model_used":       _TTS_API_MODEL,
                "error":            "",
            }

        else:
            if output_dir is None:
                output_dir = os.environ.get("PODCAST_OUTPUT_DIR", OUTPUT_DIR)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            local_path = os.path.join(output_dir, output_filename)

            _write_wav_local(local_path, wav_bytes)

            logger.info(
                "Audio saved to %s (%.1f minutes)", local_path, duration_seconds / 60
            )
            return {
                "success":          True,
                "output_path":      os.path.abspath(local_path),
                "output_url":       "",
                "artifact_saved":   False,
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed,
                "model_used":       _TTS_API_MODEL,
                "error":            "",
            }

    except Exception as exc:
        logger.error("generate_podcast_audio failed: %s", exc)
        return {
            "success":          False,
            "output_path":      "",
            "output_url":       "",
            "artifact_saved":   False,
            "duration_seconds": 0,
            "duration_minutes": 0,
            "chunks_processed": 0,
            "model_used":       "",
            "error":            str(exc),
        }


# ---------------------------------------------------------------------------
# Audio Producer Agent
# ---------------------------------------------------------------------------

audio_producer_agent = Agent(
    name="audio_producer_agent",
    model=AUDIO_PRODUCER_MODEL,
    description=(
        "Converts a formatted podcast script into a high-quality multi-speaker "
        "WAV audio file using Gemini TTS via the Cloud Text-to-Speech API. "
        "Supports local, GCS, and ADK artifact output."
    ),
    instruction="""
You are the **Audio Producer** agent. Your sole responsibility is to take a
completed, formatted podcast script and produce a high-quality audio file.

## Your Process

1. Receive the formatted script (below), review the script and create a fun title for the script:
    {base_script}

2. Call `generate_podcast_audio` with the script, title, and any style guidance.
   - The output destination is controlled by `output_mode` in agent_configuration.json.
   - For "gcs" mode: pass `write_to_gcs=True` and optionally a `gcs_output_bucket`.
   - For "artifact" and "local" modes: no extra flags needed — the tool handles it.
3. Report back using the output format below.

## What You Should Know About the TTS Process

The `generate_podcast_audio` tool will:
- Parse the script into structured speaker turns (MultiSpeakerMarkup) so
  dialogue and director's notes are fully separated
- Build Director's Notes (Audio Profile + Scene + Style guidance) passed as
  a separate prompt field — the model never speaks these aloud
- Load host names and voices from agent_configuration.json
- Synthesise each chunk via the Cloud Text-to-Speech API
- Save the result as a 24kHz mono PCM WAV file

## Your Output
After calling the tool, return in markdown the following output format:

## Title: [The generated title of the podcast] ([calculated podcast length])
-------------------------------------------------------------------------
## Script:
[The podcast script]

-------------------------------------------------------------------------

## Podcast:
- If `artifact_saved` is True in the tool result: state that the podcast has
  been presented in the chat session player
- If `output_url` is non-empty: provide it as a clickable GCS URL.

Example output (artifact mode):
## Title: The Joy of Fishing (1:30)
-------------------------------------------------------------------------
## Script:
Alex: Hey Jordan, do you like fishing?

Jordan: I sure do!
...
-------------------------------------------------------------------------

Example output (GCS mode):
## Podcast Location:
https://storage.cloud.google.com/podcast_agent_output/The_Joy_of_Fishing_20260316_174813.wav

If the tool returns an error, report it clearly to the orchestrator with
suggestions (e.g., check that host voice IDs in agent_configuration.json
are valid Gemini TTS voice names).
""",
    tools=[generate_podcast_audio],
    output_key="final_output",
)