# audio_producer_agent.py
# Converts a formatted podcast script into audio using the Cloud Text-to-Speech
# unary API (google-cloud-texttospeech SDK) with Gemini TTS models.
#
# TTS path:
#   Duo mode  — MultiSpeakerMarkup structured turns via synthesize_speech().
#               Long scripts are split into turn-boundary chunks and the
#               resulting PCM frames are concatenated.
#   Solo mode — Single-speaker Gemini TTS via synthesize_speech() with
#               the prompt field for director notes. Script paragraphs are
#               split into byte-safe chunks.
#
# The Cloud TTS unary API was chosen over the genai SDK because:
#   - MultiSpeakerMarkup turn structure prevents inter-chunk speech artifacts
#   - The API is stable and well-tested for podcast-length content
#   - Streaming API does not support Gemini TTS prebuilt voices
#
# Output behaviour is controlled by two config flags:
#
#   gemini_enterprise: true
#     → JPEG artifact  (album art, renders inline in chat)
#     → WAV artifact   (plays in browser via Gemini Enterprise)
#     → MP3 → GCS      (tagged with art/title, downloadable for personal players)
#       GCS_OUTPUT_BUCKET must be set when this flag is true.
#
#   gemini_enterprise: false  (default)
#     → JPEG artifact  (album art, renders inline in chat)
#     → MP3 artifact   (tagged with art/title, downloadable from chat)

import io
import json
import logging
import os
import re
import wave
from datetime import datetime
from pathlib import Path
from typing import Optional

import lameenc
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TCON, TDRC, APIC
from google.cloud import texttospeech
from google import genai as genai_client
from google.genai import types
from google.adk.agents import Agent
from google.adk.tools import ToolContext

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

# Strip "-preview" infix so both config formats work transparently.
# e.g. "gemini-2.5-flash-preview-tts" → "gemini-2.5-flash-tts"
_TTS_API_MODEL = TTS_MODEL.replace("-preview-tts", "-tts")

# ---------------------------------------------------------------------------
# Cloud TTS client  (unary synthesis)
# ---------------------------------------------------------------------------

tts_client = texttospeech.TextToSpeechClient()

# ---------------------------------------------------------------------------
# Vertex AI genai client  (image generation only)
# ---------------------------------------------------------------------------

_VERTEX       = _CFG["vertex_ai"]
_image_client = genai_client.Client(
    vertexai=True,
    project=_VERTEX["project"],
    location=_VERTEX["location"],
)

# ---------------------------------------------------------------------------
# Podcast Speaker Setup
# ---------------------------------------------------------------------------

HOST_1    = _CFG["hosts"]["host_1"]
HOST_2    = _CFG["hosts"]["host_2"]
SOLO_HOST = _CFG["solo_host"]

# Flattened descriptions — no markdown, prevents TTS reading them aloud
ALEX_DESCRIPTION = (
    f"{HOST_1['name']} is the curious host. Personality: {HOST_1['description']}. "
    "Voice: Warm, upbeat, and accessible. Often introduces topics and asks clarifying questions."
)
JORDAN_DESCRIPTION = (
    f"{HOST_2['name']} is the expert host. Personality: {HOST_2['description']}. "
    "Voice: Calm authority with occasional dry wit. Brings data and history, and sometimes pauses to think."
)
SOLO_DESCRIPTION = (
    f"{SOLO_HOST['name']} is a knowledgeable narrator. Personality: {SOLO_HOST['description']}. "
    "Voice: Clear, warm, and authoritative. Guides the listener through concepts step by step."
)

# ---------------------------------------------------------------------------
# Audio / Output Configuration
# ---------------------------------------------------------------------------

_AUDIO  = _CFG["audio"]
_OUTPUT = _CFG["output"]

AUDIO_SAMPLE_RATE  = _AUDIO["sample_rate"]
AUDIO_CHANNELS     = _AUDIO["channels"]
AUDIO_SAMPLE_WIDTH = _AUDIO["sample_width"]
CHUNK_TARGET_WORDS = _AUDIO["chunk_target_words"]

OUTPUT_DIR        = _OUTPUT["local_output_dir"]
GCS_OUTPUT_BUCKET = _OUTPUT["gcs_output_bucket"]
OUTPUT_MODE       = _OUTPUT.get("output_mode", "gcs")   # "artifact" | "gcs" | "local"
GEMINI_ENTERPRISE = _OUTPUT.get("gemini_enterprise", False)

_PODCAST    = _CFG.get("podcast", {})
SHOW_NAME   = _PODCAST.get("show_name", "Podcast")
MP3_BITRATE = _PODCAST.get("mp3_bitrate", 128)

_ART_CFG         = _CFG.get("album_art", {})
ART_ENABLED      = _ART_CFG.get("enabled", True)
ART_IMAGE_SIZE   = _ART_CFG.get("image_size", "1K")
ART_ASPECT_RATIO = _ART_CFG.get("aspect_ratio", "1:1")

IMAGE_MODEL = _CFG["models"].get("image", "gemini-3.1-flash-image-preview")

# ---------------------------------------------------------------------------
# Director's Notes Builder
# ---------------------------------------------------------------------------

def _build_director_notes(
    podcast_title: str,
    style_guidance: Optional[str],
    chunk_index: int,
    total_chunks: int,
    narrator_mode: str = "duo",
) -> str:
    """
    Builds the prompt field passed to SynthesisInput.

    The Cloud TTS prompt field is a SHORT style instruction — equivalent to
    the docs examples: "Narrate in a calm, professional tone."
    It must NOT be written as an LLM system prompt or contain long instructions,
    as the TTS model may read those aloud as part of the audio output.

    Kept well under the 4,000 byte per-field limit.
    """
    extra = f" {style_guidance}." if style_guidance else ""

    if narrator_mode == "solo":
        return (
            f"Narrate the following as a knowledgeable, warm educator speaking "
            f"directly to a curious developer audience. Natural pace, clear delivery."
            f"{extra}"
        )
    else:
        return (
            f"Perform the following as a natural two-host podcast conversation. "
            f"Both hosts are engaged, conversational, and knowledgeable."
            f"{extra}"
        )


# ---------------------------------------------------------------------------
# Script Parser — duo mode
# ---------------------------------------------------------------------------

def _parse_script_to_turns(
    script: str,
) -> list[texttospeech.MultiSpeakerMarkup.Turn]:
    """
    Parses a two-host script into MultiSpeakerMarkup.Turn objects.
    Consecutive lines by the same speaker are merged into one turn.
    """
    host_names   = {HOST_1["name"], HOST_2["name"]}
    turn_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in host_names) + r"):\s*(.+)",
        re.MULTILINE,
    )

    turns: list[texttospeech.MultiSpeakerMarkup.Turn] = []
    for m in turn_pattern.finditer(script):
        speaker, text = m.group(1), m.group(2).strip()
        if not text:
            continue
        if turns and turns[-1].speaker == speaker:
            turns[-1] = texttospeech.MultiSpeakerMarkup.Turn(
                speaker=speaker, text=turns[-1].text + " " + text
            )
        else:
            turns.append(texttospeech.MultiSpeakerMarkup.Turn(speaker=speaker, text=text))

    if not turns:
        raise ValueError(
            f"No speaker turns found. Expected lines starting with "
            f"'{HOST_1['name']}:' or '{HOST_2['name']}'."
        )

    logger.info("Parsed %d speaker turns (duo mode).", len(turns))
    return turns


# ---------------------------------------------------------------------------
# Script Parser — solo mode
# ---------------------------------------------------------------------------

def _parse_script_to_paragraphs(script: str) -> list[str]:
    """
    Parses a solo-narrator script into plain text paragraphs,
    stripping the leading speaker label (e.g. "Alex: ").
    """
    label = re.compile(r"^" + re.escape(SOLO_HOST["name"]) + r":\s*", re.MULTILINE)
    paragraphs = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        text = label.sub("", line).strip()
        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError(
            f"No narrator paragraphs found. "
            f"Expected lines starting with '{SOLO_HOST['name']}'."
        )

    logger.info("Parsed %d paragraphs (solo mode).", len(paragraphs))
    return paragraphs


# ---------------------------------------------------------------------------
# Chunking — duo mode (split at turn boundaries)
# ---------------------------------------------------------------------------

def _chunk_turns(
    turns: list[texttospeech.MultiSpeakerMarkup.Turn],
    target_words: int,
) -> list[list[texttospeech.MultiSpeakerMarkup.Turn]]:
    """Splits turn list into word-count-bounded chunks at turn boundaries."""
    total = sum(len(t.text.split()) for t in turns)
    if total <= target_words:
        return [turns]

    chunks, current, count = [], [], 0
    for turn in turns:
        tw = len(turn.text.split())
        if count + tw > target_words and current:
            chunks.append(current)
            current, count = [turn], tw
        else:
            current.append(turn)
            count += tw
    if current:
        chunks.append(current)

    logger.info("Split %d turns → %d chunk(s) (~%d words, duo).", len(turns), len(chunks), total)
    return chunks


# ---------------------------------------------------------------------------
# Chunking — solo mode (split at paragraph boundaries)
# ---------------------------------------------------------------------------

def _chunk_paragraphs(
    paragraphs: list[str],
    target_words: int,
) -> list[str]:
    """Joins paragraphs into word-count-bounded text chunks."""
    total = sum(len(p.split()) for p in paragraphs)
    if total <= target_words:
        return ["\n\n".join(paragraphs)]

    chunks, current, count = [], [], 0
    for para in paragraphs:
        pw = len(para.split())
        if count + pw > target_words and current:
            chunks.append("\n\n".join(current))
            current, count = [para], pw
        else:
            current.append(para)
            count += pw
    if current:
        chunks.append("\n\n".join(current))

    logger.info("Split %d paragraphs → %d chunk(s) (~%d words, solo).", len(paragraphs), len(chunks), total)
    return chunks


# ---------------------------------------------------------------------------
# WAV Builder
# ---------------------------------------------------------------------------

def _build_wav_bytes(
    pcm_data: bytes,
    channels: int     = AUDIO_CHANNELS,
    rate: int         = AUDIO_SAMPLE_RATE,
    sample_width: int = AUDIO_SAMPLE_WIDTH,
) -> bytes:
    """Wraps raw LINEAR16 PCM bytes in a WAV header."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# GCS Upload Helper
# ---------------------------------------------------------------------------

def _upload_to_gcs(gcs_path: str, data_bytes: bytes, content_type: str) -> str:
    """Uploads bytes to GCS and returns the authenticated URL."""
    from google.cloud import storage

    parts       = gcs_path[5:].split("/", 1)
    bucket_name = parts[0]
    blob_name   = parts[1] if len(parts) > 1 else ""

    storage.Client().bucket(bucket_name).blob(blob_name).upload_from_string(
        data_bytes, content_type=content_type
    )
    url = f"https://storage.cloud.google.com/{bucket_name}/{blob_name}"
    logger.info("Uploaded to %s", url)
    return url


# ---------------------------------------------------------------------------
# Album Art Generation
# ---------------------------------------------------------------------------

def _generate_album_art(
    podcast_title: str,
    topic_summary: str,
    key_themes: str,
    target_audience: str,
) -> Optional[bytes]:
    """
    Generates square JPEG album art via the Gemini image model on Vertex AI.
    Returns JPEG bytes on success, None on failure (non-fatal).
    """
    if not ART_ENABLED:
        return None

    prompt = (
        f"Create square podcast cover art for an episode titled '{podcast_title}'.\n\n"
        f"Episode summary: {topic_summary}\n"
        f"Key themes: {key_themes}\n"
        f"Target audience: {target_audience}\n\n"
        f"Design requirements:\n"
        f"- The episode title '{podcast_title}' MUST appear as large, bold, clearly "
        f"legible text. Make the title the dominant visual element.\n"
        f"- Imagery, colour palette and style must directly reflect the episode themes.\n"
        f"- Modern, professional podcast cover art aesthetic.\n"
        f"- High contrast so the title reads at small sizes.\n"
        f"- No border. Fill the entire square. No show name or host names."
    )

    try:
        response = _image_client.models.generate_content(
            model=IMAGE_MODEL,
            contents=prompt,
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE"],
                image_config=types.ImageConfig(
                    aspect_ratio=ART_ASPECT_RATIO,
                    image_size=ART_IMAGE_SIZE,
                    output_mime_type="image/jpeg",
                ),
            ),
        )
        for part in response.candidates[0].content.parts:
            if part.inline_data and part.inline_data.data:
                logger.info("_generate_album_art: %d bytes for '%s'",
                            len(part.inline_data.data), podcast_title)
                return part.inline_data.data

        logger.warning("_generate_album_art: no image returned for '%s'", podcast_title)
        return None

    except Exception as exc:
        logger.warning("_generate_album_art failed (non-fatal): %s", exc)
        return None


# ---------------------------------------------------------------------------
# PCM → MP3 + ID3 Tagging
# ---------------------------------------------------------------------------

def _build_mp3_bytes(
    pcm_data: bytes,
    podcast_title: str,
    album_art_jpeg: Optional[bytes],
    narrator_mode: str = "duo",
) -> bytes:
    """
    Encodes raw 16-bit PCM to MP3 via lameenc, then prepends ID3v2.3 tags.

    ID3 tags are written into a separate empty buffer and prepended to the
    raw MP3 bytes. Writing tags into a buffer that already contains MP3 data
    overwrites the start of the audio — causing only the first few seconds
    to play back correctly.
    """
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(AUDIO_SAMPLE_RATE)
    encoder.set_channels(AUDIO_CHANNELS)
    encoder.set_quality(2)

    mp3_chunks = []
    chunk_size  = AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH * 10   # 10-second slices
    for offset in range(0, len(pcm_data), chunk_size):
        mp3_chunks.append(encoder.encode(pcm_data[offset:offset + chunk_size]))
    mp3_chunks.append(encoder.flush())
    mp3_bytes = b"".join(mp3_chunks)

    artist = (SOLO_HOST["name"] if narrator_mode == "solo"
              else f"{HOST_1['name']} & {HOST_2['name']}")

    tags = ID3()
    tags.add(TIT2(encoding=3, text=podcast_title))
    tags.add(TALB(encoding=3, text=SHOW_NAME))
    tags.add(TPE1(encoding=3, text=artist))
    tags.add(TCON(encoding=3, text="Podcast"))
    tags.add(TDRC(encoding=3, text=datetime.now().strftime("%Y")))

    if album_art_jpeg:
        tags.add(APIC(encoding=0, mime="image/jpeg", type=3,
                      desc="Cover", data=album_art_jpeg))
        logger.info("_build_mp3_bytes: APIC tag added (%d bytes)", len(album_art_jpeg))

    tag_buf = io.BytesIO()
    tags.save(tag_buf, v2_version=3)
    tagged_mp3 = tag_buf.getvalue() + mp3_bytes

    logger.info("_build_mp3_bytes: %d PCM bytes → %d MP3 bytes (tagged)",
                len(pcm_data), len(tagged_mp3))
    return tagged_mp3


# ---------------------------------------------------------------------------
# Podcast Generation Function Tool
# ---------------------------------------------------------------------------

async def generate_podcast_audio(
    script: str,
    podcast_title: str,
    narrator_mode: str = "duo",
    topic_summary: str = "",
    key_themes: str = "",
    target_audience: str = "",
    write_to_gcs: bool = False,
    gcs_output_bucket: Optional[str] = None,
    output_dir: Optional[str] = None,
    style_guidance: Optional[str] = None,
    tool_context: Optional[ToolContext] = None,
) -> dict:
    """
    Converts a formatted podcast script to audio with album art.

    TTS uses the Cloud Text-to-Speech unary API (synthesize_speech):
      Solo  — single-speaker Gemini TTS with prompt field for director notes.
      Duo   — MultiSpeakerMarkup structured turns; supports chunking without
              inter-chunk speech boundary artifacts.

    Output is controlled by output_mode + gemini_enterprise in
    agent_configuration.json.

    Returns:
        dict with: success, output_path, output_url, mp3_gcs_url,
        artifact_saved, wav_artifact_saved, art_artifact_saved,
        art_filename, duration_seconds, duration_minutes,
        chunks_processed, model_used, narrator_mode, error.
    """
    try:
        safe_title   = re.sub(r"[^\w\-_]", "_", podcast_title)[:50]
        timestamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp3_filename = f"{safe_title}_{timestamp}.mp3"
        wav_filename = f"{safe_title}_{timestamp}.wav"
        art_filename = f"{safe_title}_{timestamp}_cover.jpg"

        narrator_mode = (narrator_mode or "duo").lower().strip()
        is_solo       = narrator_mode == "solo"

        logger.info("generate_podcast_audio: narrator_mode=%s title='%s'",
                    narrator_mode, podcast_title)

        all_pcm: list[bytes] = []
        chunks_processed = 0

        audio_config = texttospeech.AudioConfig(
            audio_encoding=texttospeech.AudioEncoding.LINEAR16,
            sample_rate_hertz=AUDIO_SAMPLE_RATE,
        )

        # ── SOLO MODE ────────────────────────────────────────────────────────
        if is_solo:
            paragraphs  = _parse_script_to_paragraphs(script)
            text_chunks = _chunk_paragraphs(paragraphs, CHUNK_TARGET_WORDS)

            logger.info("Processing %d TTS chunk(s) for '%s' (solo).",
                        len(text_chunks), podcast_title)

            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=SOLO_HOST["voice"],
                model_name=_TTS_API_MODEL,
            )

            for i, chunk in enumerate(text_chunks):
                logger.info("  Solo chunk %d/%d...", i + 1, len(text_chunks))

                notes = _build_director_notes(
                    podcast_title=podcast_title,
                    style_guidance=style_guidance,
                    chunk_index=i,
                    total_chunks=len(text_chunks),
                    narrator_mode="solo",
                )

                response = tts_client.synthesize_speech(
                    input=texttospeech.SynthesisInput(text=chunk, prompt=notes),
                    voice=voice,
                    audio_config=audio_config,
                )
                all_pcm.append(response.audio_content)
                chunks_processed += 1

        # ── DUO MODE ─────────────────────────────────────────────────────────
        else:
            all_turns   = _parse_script_to_turns(script)
            turn_chunks = _chunk_turns(all_turns, CHUNK_TARGET_WORDS)

            logger.info("Processing %d TTS chunk(s) for '%s' (duo).",
                        len(turn_chunks), podcast_title)

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

            for i, turns in enumerate(turn_chunks):
                logger.info("  Duo chunk %d/%d...", i + 1, len(turn_chunks))

                notes = _build_director_notes(
                    podcast_title=podcast_title,
                    style_guidance=style_guidance,
                    chunk_index=i,
                    total_chunks=len(turn_chunks),
                    narrator_mode="duo",
                )

                response = tts_client.synthesize_speech(
                    input=texttospeech.SynthesisInput(
                        multi_speaker_markup=texttospeech.MultiSpeakerMarkup(turns=turns),
                        prompt=notes,
                    ),
                    voice=voice,
                    audio_config=audio_config,
                )
                all_pcm.append(response.audio_content)
                chunks_processed += 1

        # -- Assemble, generate art, encode ---------------------------------
        combined_pcm     = b"".join(all_pcm)
        bytes_per_second = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        duration_seconds = len(combined_pcm) // bytes_per_second

        wav_bytes = _build_wav_bytes(combined_pcm)

        logger.info("Generating album art for '%s'...", podcast_title)
        album_art_jpeg = _generate_album_art(
            podcast_title=podcast_title,
            topic_summary=topic_summary or podcast_title,
            key_themes=key_themes,
            target_audience=target_audience,
        )

        logger.info("Encoding PCM → MP3 for '%s'...", podcast_title)
        mp3_bytes = _build_mp3_bytes(
            pcm_data=combined_pcm,
            podcast_title=podcast_title,
            album_art_jpeg=album_art_jpeg,
            narrator_mode=narrator_mode,
        )

        # -- Output routing --------------------------------------------------
        effective_mode = OUTPUT_MODE
        if write_to_gcs and effective_mode not in ("artifact", "gcs"):
            effective_mode = "gcs"

        art_artifact_saved = False
        wav_artifact_saved = False
        mp3_gcs_url        = ""

        # ── Artifact mode ────────────────────────────────────────────────────
        if effective_mode == "artifact":
            if tool_context is None:
                raise RuntimeError(
                    "output_mode is 'artifact' but tool_context is None. "
                    "Ensure the Runner is configured with an ArtifactService."
                )

            if album_art_jpeg:
                await tool_context.save_artifact(
                    filename=art_filename,
                    artifact=types.Part(
                        inline_data=types.Blob(data=album_art_jpeg, mime_type="image/jpeg")
                    ),
                )
                art_artifact_saved = True
                logger.info("Album art saved as artifact '%s'.", art_filename)

            if GEMINI_ENTERPRISE:
                bucket = (gcs_output_bucket or GCS_OUTPUT_BUCKET)
                if not bucket:
                    raise RuntimeError(
                        "gemini_enterprise is True but no GCS bucket configured."
                    )
                bucket = bucket.rstrip("/")

                wav_version = await tool_context.save_artifact(
                    filename=wav_filename,
                    artifact=types.Part(
                        inline_data=types.Blob(data=wav_bytes, mime_type="audio/wav")
                    ),
                )
                wav_artifact_saved = True
                logger.info("WAV saved as artifact '%s' v%d (%.1f min).",
                            wav_filename, wav_version, duration_seconds / 60)

                mp3_gcs_url = _upload_to_gcs(
                    f"{bucket}/{mp3_filename}", mp3_bytes, "audio/mpeg"
                )
                logger.info("MP3 uploaded to GCS: %s/%s", bucket, mp3_filename)

                return {
                    "success": True, "output_path": wav_filename, "output_url": "",
                    "mp3_gcs_url": mp3_gcs_url, "artifact_saved": True,
                    "wav_artifact_saved": True, "art_artifact_saved": art_artifact_saved,
                    "art_filename": art_filename if art_artifact_saved else "",
                    "duration_seconds": duration_seconds,
                    "duration_minutes": round(duration_seconds / 60, 1),
                    "chunks_processed": chunks_processed, "model_used": _TTS_API_MODEL,
                    "narrator_mode": narrator_mode, "error": "",
                }

            else:
                mp3_version = await tool_context.save_artifact(
                    filename=mp3_filename,
                    artifact=types.Part(
                        inline_data=types.Blob(data=mp3_bytes, mime_type="audio/mpeg")
                    ),
                )
                logger.info("MP3 saved as artifact '%s' v%d (%.1f min).",
                            mp3_filename, mp3_version, duration_seconds / 60)

                return {
                    "success": True, "output_path": mp3_filename, "output_url": "",
                    "mp3_gcs_url": "", "artifact_saved": True,
                    "wav_artifact_saved": False, "art_artifact_saved": art_artifact_saved,
                    "art_filename": art_filename if art_artifact_saved else "",
                    "duration_seconds": duration_seconds,
                    "duration_minutes": round(duration_seconds / 60, 1),
                    "chunks_processed": chunks_processed, "model_used": _TTS_API_MODEL,
                    "narrator_mode": narrator_mode, "error": "",
                }

        # ── GCS mode ─────────────────────────────────────────────────────────
        elif effective_mode == "gcs":
            bucket     = (gcs_output_bucket or GCS_OUTPUT_BUCKET).rstrip("/")
            output_url = _upload_to_gcs(f"{bucket}/{mp3_filename}", mp3_bytes, "audio/mpeg")
            logger.info("MP3 uploaded to %s (%.1f min).", f"{bucket}/{mp3_filename}",
                        duration_seconds / 60)
            return {
                "success": True, "output_path": f"{bucket}/{mp3_filename}",
                "output_url": output_url, "mp3_gcs_url": output_url,
                "artifact_saved": False, "wav_artifact_saved": False,
                "art_artifact_saved": False, "art_filename": "",
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed, "model_used": _TTS_API_MODEL,
                "narrator_mode": narrator_mode, "error": "",
            }

        # ── Local mode ────────────────────────────────────────────────────────
        else:
            if output_dir is None:
                output_dir = os.environ.get("PODCAST_OUTPUT_DIR", OUTPUT_DIR)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            local_path = os.path.join(output_dir, mp3_filename)
            with open(local_path, "wb") as f:
                f.write(mp3_bytes)
            logger.info("MP3 saved to %s (%.1f min).", local_path, duration_seconds / 60)
            return {
                "success": True, "output_path": os.path.abspath(local_path),
                "output_url": "", "mp3_gcs_url": "",
                "artifact_saved": False, "wav_artifact_saved": False,
                "art_artifact_saved": False, "art_filename": "",
                "duration_seconds": duration_seconds,
                "duration_minutes": round(duration_seconds / 60, 1),
                "chunks_processed": chunks_processed, "model_used": _TTS_API_MODEL,
                "narrator_mode": narrator_mode, "error": "",
            }

    except Exception as exc:
        logger.error("generate_podcast_audio failed: %s", exc)
        return {
            "success": False, "output_path": "", "output_url": "",
            "mp3_gcs_url": "", "artifact_saved": False,
            "wav_artifact_saved": False, "art_artifact_saved": False,
            "art_filename": "", "duration_seconds": 0, "duration_minutes": 0,
            "chunks_processed": 0, "model_used": "",
            "narrator_mode": narrator_mode if "narrator_mode" in dir() else "unknown",
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# Audio Producer Agent
# ---------------------------------------------------------------------------

audio_producer_agent = Agent(
    name="audio_producer_agent",
    model=AUDIO_PRODUCER_MODEL,
    description=(
        "Converts a formatted podcast script into audio with dynamically "
        "generated album art and embedded ID3 tags. Supports solo narrator "
        "and two-host duo modes. Output format depends on the gemini_enterprise "
        "flag in agent_configuration.json."
    ),
    instruction="""
You are the **Audio Producer** agent. Your sole responsibility is to take a
completed, formatted podcast script and produce a high-quality podcast output
with dynamically generated album art.

## Your Process

1. Receive the formatted script and narrator mode:
    Script: {base_script}
    Narrator Mode: {narrator_mode:duo}

2. Extract the following from the script content:
   - **Episode title** – punchy and engaging (5 words max if possible)
   - **Topic summary** – one sentence describing what the episode covers
   - **Key themes** – 3–5 comma-separated keywords
   - **Target audience** – inferred from the content tone

3. Call `generate_podcast_audio` with:
   - `script`, `podcast_title`, `topic_summary`, `key_themes`, `target_audience`
   - `narrator_mode` — pass through exactly as received (`"solo"` or `"duo"`)
   - `style_guidance` — optional tone notes
   - For "gcs" mode only: `write_to_gcs=True` and optionally `gcs_output_bucket`

## Your Output
After calling the tool, return in this exact markdown format:

## Title: [Episode title] ([duration])
-------------------------------------------------------------------------
## Album Art:
[If `art_artifact_saved` is True: "Episode cover art was generated and will appear below."]

## Script:
[The podcast script]

-------------------------------------------------------------------------

## Podcast:
[If `wav_artifact_saved` is True AND `mp3_gcs_url` is non-empty:
  "Your podcast is ready:
   - 🎧 **Listen now** — the WAV player will appear in the chat above.
   - 📥 **Download MP3** (includes album art and title) — [mp3_gcs_url]"]

[If `artifact_saved` is True AND `wav_artifact_saved` is False:
  "Your podcast MP3 with embedded album art is ready — click the download button to save it."]

[If `output_url` is non-empty and no artifacts:
  Provide it as a clickable GCS URL.]

If the tool returns an error, report it clearly and DO NOT retry — stop and explain the issue to the user.
""",
    tools=[generate_podcast_audio],
    output_key="final_output",
)