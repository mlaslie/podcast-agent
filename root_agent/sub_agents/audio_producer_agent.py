# audio_producer_agent.py
# Converts a formatted two-host podcast script into a WAV audio file using
# Gemini TTS. Supports writing output locally or to Google Cloud Storage.

import io
import json
import logging
import os
import re
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

from google import genai
from google.adk.agents import Agent
from google.genai import types

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

TTS_MODEL            = _CFG["models"]["tts"]
AUDIO_PRODUCER_MODEL = _CFG["models"]["audio_producer"]

# ---------------------------------------------------------------------------
# Vertex AI Client
# ---------------------------------------------------------------------------

_VERTEX = _CFG["vertex_ai"]

client = genai.Client(
    vertexai=True,
    project=_VERTEX["project"],
    location=_VERTEX["location"],
)

# ---------------------------------------------------------------------------
# Podcast Speaker Setup
# ---------------------------------------------------------------------------

HOST_1 = _CFG["hosts"]["host_1"]
HOST_2 = _CFG["hosts"]["host_2"]

ALEX_DESCRIPTION = f"""
## {HOST_1['name']} — The Curious Host
Personality: {HOST_1['description']}
Voice: Warm, upbeat, accessible. Often the one to introduce topics and ask
"so what does that actually mean for our listeners?" questions.
"""

JORDAN_DESCRIPTION = f"""
## {HOST_2['name']} — The Expert Host
Personality: {HOST_2['description']}
Voice: Calm authority with occasional dry wit. Brings the receipts — data,
history, nuance. Sometimes pauses before answering to think it through.
"""

# ---------------------------------------------------------------------------
# Audio Output Configuration
# ---------------------------------------------------------------------------

_AUDIO  = _CFG["audio"]
_OUTPUT = _CFG["output"]

AUDIO_SAMPLE_RATE  = _AUDIO["sample_rate"]
AUDIO_CHANNELS     = _AUDIO["channels"]
AUDIO_SAMPLE_WIDTH = _AUDIO["sample_width"]
TTS_MAX_CHARS      = _AUDIO["tts_max_chars"]
CHUNK_TARGET_WORDS = _AUDIO["chunk_target_words"]

OUTPUT_DIR        = _OUTPUT["local_output_dir"]
GCS_OUTPUT_BUCKET = _OUTPUT["gcs_output_bucket"]


# ---------------------------------------------------------------------------
# TTS Prompt Builder (Director's Notes style)
# ---------------------------------------------------------------------------

def _build_tts_prompt(
    script_chunk: str,
    podcast_title: str,
    style_guidance: Optional[str],
    chunk_index: int,
    total_chunks: int,
) -> str:
    """
    Builds a rich Director's Notes TTS prompt per the Gemini prompting guide.
    This is the key to getting natural, expressive speech output.
    """
    continuation_note = ""
    if total_chunks > 1:
        if chunk_index == 0:
            continuation_note = "This is the opening segment of the podcast."
        elif chunk_index == total_chunks - 1:
            continuation_note = "This is the closing segment of the podcast."
        else:
            continuation_note = (
                f"This is segment {chunk_index + 1} of {total_chunks} "
                "of an ongoing podcast conversation. The hosts are mid-discussion."
            )

    extra_style = ""
    if style_guidance:
        extra_style = f"\nAdditional Style Notes: {style_guidance}\n"

#
# The below big prompt kind sometime trigger an issue where Gemini TTS includes
# The scene, style, title...etc in the audio generation
# Seems to happen rarely with 2.5 Flash TTS
# Have not tested with 2.5 Pro TTS yet
#

    prompt = f"""# AUDIO PROFILE: Two-Host Podcast
## "{podcast_title}"

## THE SCENE: Modern Podcast Studio
Two knowledgeable hosts — {HOST_1['name']} and {HOST_2['name']} — are seated
across from each other in a warmly-lit, acoustically-treated studio. The
"ON AIR" light glows red. There's genuine warmth and chemistry between them.
They have prepared notes but speak as if in natural conversation. The energy
is focused but relaxed — like two smart friends who happen to be recording.
{continuation_note}

{ALEX_DESCRIPTION}
{JORDAN_DESCRIPTION}

### DIRECTOR'S NOTES

Style:
- Both hosts sound completely natural and conversational — warm, engaged,
  and genuinely interested in each other's points.
- Inline markers in the script must be performed:
  * [pause] → a brief 0.5–1 second natural thinking pause
  * [laughs] → a brief, genuine light laugh
  * [sighs] → a thoughtful, reflective exhale
- "um," and "you know," → performed as natural hesitations, not awkward gaps
- Contractions everywhere — "it's", "we're", "they've", "can't"
- Natural sentence-final intonation — statements fall, questions rise,
  excited points get a slight upward lift mid-sentence
- Energy modulates with content: thoughtful moments slow down slightly,
  exciting revelations pick up pace
- Hosts react to each other even in short interjections ("Totally.",
  "Right.", "Exactly.") — these should sound connected, not robotic

Pacing:
- Conversational pace: approximately 130–150 words per minute
- Natural micro-pauses between clauses within a speaker's turn
- Clean, brief pause between speaker transitions

{HOST_1['name']}'s Voice Profile ({HOST_1['voice']}):
- Upbeat, curious, accessible energy
- Slight rise in pitch when asking questions
- Warm laughter when appropriate
- Emphasises words naturally for clarity

{HOST_2['name']}'s Voice Profile ({HOST_2['voice']}):
- Measured, authoritative, occasionally dry
- Slight deliberative pause before answering complex points
- Confident downward inflection on key facts
- Warm but controlled enthusiasm
{extra_style}

### TRANSCRIPT
{script_chunk}"""

    return prompt


# ---------------------------------------------------------------------------
# Script Chunking
# ---------------------------------------------------------------------------

def _split_script_into_chunks(script: str, target_words_per_chunk: int) -> list[str]:
    """
    Splits a long script into chunks at natural speaker-turn boundaries.
    Tries to keep each chunk under target_words_per_chunk words while
    ensuring chunks don't split mid-turn.
    """
    total_words = len(script.split())
    if total_words <= target_words_per_chunk:
        return [script]

    # Split on speaker turns (lines starting with Alex: or Jordan:)
    # Keep the separator with each turn
    turns = re.split(r"(?=\n(?:Alex|Jordan):)", script)
    turns = [t.strip() for t in turns if t.strip()]

    chunks = []
    current_chunk_turns = []
    current_word_count = 0

    for turn in turns:
        turn_words = len(turn.split())

        if current_word_count + turn_words > target_words_per_chunk and current_chunk_turns:
            chunks.append("\n\n".join(current_chunk_turns))
            current_chunk_turns = [turn]
            current_word_count = turn_words
        else:
            current_chunk_turns.append(turn)
            current_word_count += turn_words

    if current_chunk_turns:
        chunks.append("\n\n".join(current_chunk_turns))

    logger.info(
        "Script split into %d chunk(s) (total %d words)", len(chunks), total_words
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

    Follows the official GCS Python client library pattern:
    https://cloud.google.com/storage/docs/uploading-objects#python

    Args:
        gcs_path:  Full GCS destination URI (e.g. gs://bucket/file.wav).
        wav_bytes: WAV-headered audio bytes to upload.

    Returns:
        The authenticated GCS URL (https://storage.cloud.google.com/bucket/blob)
        of the uploaded file.
    """
    from google.cloud import storage

    path_parts  = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name   = path_parts[1] if len(path_parts) > 1 else ""

    storage_client = storage.Client()
    bucket         = storage_client.bucket(bucket_name)
    blob           = bucket.blob(blob_name)

    # generation_match_precondition=0 ensures the upload aborts if an object
    # with the same name already exists, preventing accidental overwrites.
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

def generate_podcast_audio(
    script: str,
    podcast_title: str,
    write_to_gcs: bool = False,
    gcs_output_bucket: Optional[str] = None,
    output_dir: Optional[str] = None,
    style_guidance: Optional[str] = None,
) -> dict:
    """
    Converts a formatted two-host podcast script to a WAV audio file.

    Uses Gemini TTS with multi-speaker voice config.
    Long scripts are split into chunks and concatenated.

    Args:
        script:            Formatted script with speaker labels matching host names
                           defined in agent_configuration.json.
        podcast_title:     Episode title (used for filename and TTS prompt context).
        write_to_gcs:      If True, upload the WAV to GCS instead of writing locally.
                           Returns the authenticated GCS URL. Defaults to False.
        gcs_output_bucket: GCS bucket URI to upload to when write_to_gcs=True.
                           Defaults to gcs_output_bucket in agent_configuration.json.
        output_dir:        Local directory to write the WAV file when write_to_gcs=False.
                           Defaults to local_output_dir in agent_configuration.json.
        style_guidance:    Optional extra style notes to include in the TTS prompt.

    Returns:
        A dict with keys:
        - success (bool)
        - output_path (str): Local absolute path (write_to_gcs=False) or
                             GCS URI (write_to_gcs=True).
        - output_url (str):  Authenticated GCS URL when write_to_gcs=True, else empty string.
        - duration_seconds (int): Estimated audio duration.
        - duration_minutes (float): Estimated audio duration in minutes.
        - chunks_processed (int): Number of TTS API calls made.
        - model_used (str)
        - error (str)
    """
    try:
        safe_title      = re.sub(r"[^\w\-_]", "_", podcast_title)[:50]
        timestamp       = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{safe_title}_{timestamp}.wav"

        # Split script into manageable chunks if needed
        chunks = _split_script_into_chunks(script, CHUNK_TARGET_WORDS)
        logger.info("Processing %d TTS chunk(s) for '%s'", len(chunks), podcast_title)

        all_audio_frames: list[bytes] = []
        chunks_processed = 0

        for i, chunk in enumerate(chunks):
            logger.info("  Generating audio for chunk %d/%d...", i + 1, len(chunks))

            tts_prompt = _build_tts_prompt(
                script_chunk=chunk,
                podcast_title=podcast_title,
                style_guidance=style_guidance,
                chunk_index=i,
                total_chunks=len(chunks),
            )

            response = client.models.generate_content(
                model=TTS_MODEL,
                contents=tts_prompt,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        multi_speaker_voice_config=types.MultiSpeakerVoiceConfig(
                            speaker_voice_configs=[
                                types.SpeakerVoiceConfig(
                                    speaker=HOST_1["name"],
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=HOST_1["voice"]
                                        )
                                    ),
                                ),
                                types.SpeakerVoiceConfig(
                                    speaker=HOST_2["name"],
                                    voice_config=types.VoiceConfig(
                                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                            voice_name=HOST_2["voice"]
                                        )
                                    ),
                                ),
                            ]
                        )
                    ),
                ),
            )

            audio_data = response.candidates[0].content.parts[0].inline_data.data
            all_audio_frames.append(audio_data)
            chunks_processed += 1

        # Combine raw PCM frames and wrap in WAV header
        combined_pcm = b"".join(all_audio_frames)
        wav_bytes    = _build_wav_bytes(combined_pcm)

        # Estimate duration: 16-bit mono PCM = 2 bytes/sample × 24000 samples/sec
        bytes_per_second = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        duration_seconds = len(combined_pcm) // bytes_per_second

        # -------------------------------------------------------------------
        # Output: GCS or local (local not implement in upstream agent)
        # -------------------------------------------------------------------
        if write_to_gcs:
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
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed,
                "model_used":       TTS_MODEL,
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
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed,
                "model_used":       TTS_MODEL,
                "error":            "",
            }

    except Exception as exc:
        logger.error("generate_podcast_audio failed: %s", exc)
        return {
            "success":          False,
            "output_path":      "",
            "output_url":       "",
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
        "WAV audio file using Gemini TTS. Supports local and GCS output."
    ),
    instruction="""
You are the **Audio Producer** agent. Your sole responsibility is to take a
completed, formatted podcast script and produce a high-quality audio file.

## Your Process

1. Receive the formatted script (below), review the script and create a fun title for the script:
    {base_script}

2. Call `generate_podcast_audio` with the script, title, and any style guidance.
   - If the orchestrator specifies `write_to_gcs=True`, pass that flag and
     optionally a `gcs_output_bucket`. The file will be uploaded to GCS and
     the authenticated GCS URL returned instead of a local path.
   - If `write_to_gcs` is not specified, the default behaviour writes locally.
3. Report back the output location and audio duration estimate.

## What You Should Know About the TTS Process

The `generate_podcast_audio` tool will:
- Construct a rich Director's Notes prompt (Audio Profile + Scene + Director's
  Notes) around the script to guide the TTS model
- Load host names and voices from agent_configuration.json
- Pass the script to the TTS model with multi-speaker config
- Save the result as a 24kHz mono PCM WAV file

## Your Output
After calling the tool, return in markdown the following output format:

## Title: [The generated title of the podcast] ([calculated podcast length])
-------------------------------------------------------------------------
## Script:
[The podcast script]

-------------------------------------------------------------------------

## Podcast: 
a clickable Google Cloud Storage authenticated URL or local system path to podcast file

Example output:
## Title: The Joy of Fishing (1:30)
-------------------------------------------------------------------------
## Script:
Jack: Hey Jordan, do you like fishing?

Jordan: I sure do!

Jack: Well you'll going to love today's topic...we're talking about fishing.

Jordon: Well today's my lucky day I guess

...
-------------------------------------------------------------------------

## Podcast Location: 
https://storage.cloud.google.com/podcast_agent_output/The_Joy_of_Fishing_20260316_174813.wav


If the tool returns an error, report it clearly to the orchestrator with
suggestions (e.g., script may be too long for a single TTS call — suggest
chunking).
""",
    tools=[generate_podcast_audio],
    output_key="final_output"
)