# audio_producer_agent.py
# Converts a formatted podcast script into audio using Gemini TTS.
# Supports two narrator modes controlled by the `narrator_mode` state key:
#   "duo"  (default) — two-host MultiSpeakerMarkup TTS (Alex & Jordan)
#   "solo"           — single-voice TTS (Alex only)
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
from mutagen.id3 import ID3, TIT2, TALB, TPE1, TCON, TDRC, APIC, ID3NoHeaderError
from google.adk.agents import Agent
from google.adk.tools import ToolContext
from google.cloud import texttospeech
from google import genai as genai_client
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
# Podcast Speaker Setup
# ---------------------------------------------------------------------------

HOST_1      = _CFG["hosts"]["host_1"]
HOST_2      = _CFG["hosts"]["host_2"]
SOLO_HOST   = _CFG["solo_host"]

# FLATTENED DESCRIPTIONS: Removed markdown and line breaks to prevent TTS read-aloud
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

OUTPUT_DIR         = _OUTPUT["local_output_dir"]
GCS_OUTPUT_BUCKET  = _OUTPUT["gcs_output_bucket"]
OUTPUT_MODE        = _OUTPUT.get("output_mode", "gcs")   # "artifact" | "gcs" | "local"
GEMINI_ENTERPRISE  = _OUTPUT.get("gemini_enterprise", False)

_PODCAST    = _CFG.get("podcast", {})
SHOW_NAME   = _PODCAST.get("show_name", "Podcast")
MP3_BITRATE = _PODCAST.get("mp3_bitrate", 128)

_ART_CFG         = _CFG.get("album_art", {})
ART_ENABLED      = _ART_CFG.get("enabled", True)
ART_IMAGE_SIZE   = _ART_CFG.get("image_size", "1K")
ART_ASPECT_RATIO = _ART_CFG.get("aspect_ratio", "1:1")

IMAGE_MODEL = _CFG["models"].get("image", "gemini-3.1-flash-image-preview")

# Strip "-preview" infix so both config formats work transparently.
_TTS_API_MODEL = TTS_MODEL.replace("-preview-tts", "-tts")

# ---------------------------------------------------------------------------
# Cloud TTS Client
# ---------------------------------------------------------------------------

tts_client = texttospeech.TextToSpeechClient()

# ---------------------------------------------------------------------------
# Vertex AI Image Client
# ---------------------------------------------------------------------------

_VERTEX = _CFG["vertex_ai"]
_image_client = genai_client.Client(
    vertexai=True,
    project=_VERTEX["project"],
    location=_VERTEX["location"],
)

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
    Builds the Director's Notes prompt passed as the `prompt` field of
    SynthesisInput. Formatted as a dense paragraph without Markdown to
    prevent the Gemini-TTS model from mistakenly reading the text aloud.
    """
    continuation_note = ""
    if total_chunks > 1:
        if chunk_index == 0:
            continuation_note = "This is the opening segment."
        elif chunk_index == total_chunks - 1:
            continuation_note = "This is the closing segment."
        else:
            continuation_note = (
                f"This is segment {chunk_index + 1} of {total_chunks}. "
                "The narrator is mid-explanation." if narrator_mode == "solo"
                else f"This is segment {chunk_index + 1} of {total_chunks}. "
                "The hosts are mid-discussion."
            )

    extra_style = f"Additional style guidance: {style_guidance}. " if style_guidance else ""

    if narrator_mode == "solo":
        scene = (
            f"System Instructions (Do not read aloud): You are producing an educational podcast episode titled '{podcast_title}'. "
            f"The scene is a professional recording studio where a single knowledgeable narrator is delivering a clear, engaging lesson. "
            f"{continuation_note} "
            f"{SOLO_DESCRIPTION} "
            f"Delivery rules: The narrator must sound natural, warm, and authoritative — like a trusted expert speaking directly to the listener. "
            f"Perform inline markers: [short pause]/[medium pause]/[long pause] for pauses of increasing length, [laughing] as a light laugh, and [sigh] as a thoughtful exhale. "
            f"Use natural hesitations and contractions. The pace should be 120 to 140 words per minute, slightly slower than a conversation to aid comprehension. "
            f"{extra_style}"
            f"CRITICAL: These are backend director notes. Absolutely no part of this text should be spoken aloud."
        )
    else:
        scene = (
            f"System Instructions (Do not read aloud): You are producing a podcast episode titled '{podcast_title}'. "
            f"The scene is a modern podcast studio where two knowledgeable hosts are having a focused but relaxed conversation. "
            f"{continuation_note} "
            f"{ALEX_DESCRIPTION} "
            f"{JORDAN_DESCRIPTION} "
            f"Delivery rules: Both hosts must sound completely natural and conversational. "
            f"Perform inline markers: [short pause]/[medium pause]/[long pause] for pauses of increasing length, [laughing] as a light laugh, and [sigh] as an exhale. "
            f"Use natural hesitations and contractions. The pace should be 130 to 150 words per minute. "
            f"{extra_style}"
            f"CRITICAL: These are backend director notes. Absolutely no part of this text should be spoken aloud."
        )

    return scene


# ---------------------------------------------------------------------------
# Script Parser — text → structured Turn list (duo mode)
# ---------------------------------------------------------------------------

def _parse_script_to_turns_duo(
    script: str,
) -> list[texttospeech.MultiSpeakerMarkup.Turn]:
    """
    Parses a two-host formatted script into MultiSpeakerMarkup.Turn objects.

    Expects lines in the form:
        Alex: Some dialogue here...
        Jordan: Response here...

    Consecutive lines by the same speaker are merged into a single turn.
    Inline stage markers ([short pause], [medium pause], [long pause], [laughing], [sigh], [uhm]) are preserved.
    """
    host_names = {HOST_1["name"], HOST_2["name"]}
    turn_pattern = re.compile(
        r"^(" + "|".join(re.escape(n) for n in host_names) + r"):\s*(.+)",
        re.MULTILINE,
    )

    turns: list[texttospeech.MultiSpeakerMarkup.Turn] = []
    for match in turn_pattern.finditer(script):
        speaker = match.group(1)
        text    = match.group(2).strip()
        if not text:
            continue
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
            f"Expected lines starting with '{HOST_1['name']}:' or '{HOST_2['name']}'."
        )

    logger.info("Parsed %d speaker turns from script (duo mode).", len(turns))
    return turns


# ---------------------------------------------------------------------------
# Script Parser — text → plain text paragraphs (solo mode)
# ---------------------------------------------------------------------------

def _parse_script_to_paragraphs_solo(script: str) -> list[str]:
    """
    Parses a solo-narrator script into a list of plain text paragraphs,
    stripping the leading speaker label (e.g. "Alex: ") from each line.

    Each paragraph becomes a separate TTS synthesis unit so chunks can
    still be split at natural boundaries.
    """
    speaker_name = SOLO_HOST["name"]
    label_pattern = re.compile(
        r"^" + re.escape(speaker_name) + r":\s*",
        re.MULTILINE,
    )

    paragraphs = []
    for line in script.splitlines():
        line = line.strip()
        if not line:
            continue
        # Strip the "Alex: " prefix if present; keep the rest
        text = label_pattern.sub("", line).strip()
        if text:
            paragraphs.append(text)

    if not paragraphs:
        raise ValueError(
            "No narrator paragraphs found in solo script. "
            f"Expected lines starting with '{speaker_name}:'."
        )

    logger.info("Parsed %d paragraphs from script (solo mode).", len(paragraphs))
    return paragraphs


# ---------------------------------------------------------------------------
# Turn-based Chunking (duo mode)
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
    current: list[texttospeech.MultiSpeakerMarkup.Turn] = []
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
        "Split %d turns into %d chunk(s) (total ~%d words, duo mode)",
        len(turns), len(chunks), total_words,
    )
    return chunks


# ---------------------------------------------------------------------------
# Paragraph-based Chunking (solo mode)
# ---------------------------------------------------------------------------

def _chunk_paragraphs(
    paragraphs: list[str],
    target_words: int,
) -> list[str]:
    """
    Joins paragraphs into chunks of approximately target_words each.
    Returns a list of text strings, one per TTS call.
    """
    total_words = sum(len(p.split()) for p in paragraphs)
    if total_words <= target_words:
        return [" ".join(paragraphs)]

    chunks: list[str] = []
    current_parts: list[str] = []
    current_words = 0

    for para in paragraphs:
        para_words = len(para.split())
        if current_words + para_words > target_words and current_parts:
            chunks.append(" ".join(current_parts))
            current_parts = [para]
            current_words = para_words
        else:
            current_parts.append(para)
            current_words += para_words

    if current_parts:
        chunks.append(" ".join(current_parts))

    logger.info(
        "Split %d paragraphs into %d chunk(s) (total ~%d words, solo mode)",
        len(paragraphs), len(chunks), total_words,
    )
    return chunks


# ---------------------------------------------------------------------------
# WAV Builder
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


# ---------------------------------------------------------------------------
# GCS Upload Helper
# ---------------------------------------------------------------------------

def _upload_to_gcs(gcs_path: str, data_bytes: bytes, content_type: str) -> str:
    """Uploads bytes to GCS and returns the authenticated URL."""
    from google.cloud import storage

    path_parts  = gcs_path[5:].split("/", 1)
    bucket_name = path_parts[0]
    blob_name   = path_parts[1] if len(path_parts) > 1 else ""

    blob = storage.Client().bucket(bucket_name).blob(blob_name)
    blob.upload_from_string(data_bytes, content_type=content_type)

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
    Generates dynamic square JPEG album art via the Gemini image model on
    Vertex AI. The episode title is explicitly requested as rendered text.

    Returns JPEG bytes on success, or None on failure (non-fatal).
    """
    if not ART_ENABLED:
        return None

    prompt = (
        f"Create square podcast cover art for an episode titled "
        f'"{podcast_title}".\n\n'
        f"Episode summary: {topic_summary}\n"
        f"Key themes: {key_themes}\n"
        f"Target audience: {target_audience}\n\n"
        f"Design requirements:\n"
        f'- The episode title "{podcast_title}" MUST appear as large, bold, '
        f"clearly legible text. Make the title the dominant visual element.\n"
        f"- Choose imagery, colour palette, and style that directly reflects "
        f"the episode themes — make it unique to this episode.\n"
        f"- Modern, professional podcast cover art aesthetic.\n"
        f"- High contrast so the title is easy to read at small sizes.\n"
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
                logger.info(
                    "_generate_album_art: %d bytes for '%s'",
                    len(part.inline_data.data), podcast_title,
                )
                return part.inline_data.data

        logger.warning("_generate_album_art: no image in response for '%s'", podcast_title)
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
    Encodes raw 16-bit PCM to MP3 via lameenc, then writes ID3v2.3 tags
    (including the APIC cover art frame) using mutagen.
    """
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(MP3_BITRATE)
    encoder.set_in_sample_rate(AUDIO_SAMPLE_RATE)
    encoder.set_channels(AUDIO_CHANNELS)
    encoder.set_quality(2)  # 2 = high quality

    mp3_chunks = []
    chunk_size  = AUDIO_SAMPLE_RATE * AUDIO_SAMPLE_WIDTH * 10  # 10-second chunks
    for offset in range(0, len(pcm_data), chunk_size):
        mp3_chunks.append(encoder.encode(pcm_data[offset:offset + chunk_size]))
    mp3_chunks.append(encoder.flush())
    mp3_bytes = b"".join(mp3_chunks)

    buf = io.BytesIO(mp3_bytes)
    buf.seek(0)

    try:
        tags = ID3(buf)
    except ID3NoHeaderError:
        tags = ID3()

    # Artist tag reflects narrator mode
    if narrator_mode == "solo":
        artist_tag = SOLO_HOST["name"]
    else:
        artist_tag = f"{HOST_1['name']} & {HOST_2['name']}"

    tags.add(TIT2(encoding=3, text=podcast_title))
    tags.add(TALB(encoding=3, text=SHOW_NAME))
    tags.add(TPE1(encoding=3, text=artist_tag))
    tags.add(TCON(encoding=3, text="Podcast"))
    tags.add(TDRC(encoding=3, text=datetime.now().strftime("%Y")))

    if album_art_jpeg:
        tags.add(APIC(
            encoding=0,
            mime="image/jpeg",
            type=3,       # 3 = Front cover
            desc="Cover",
            data=album_art_jpeg,
        ))
        logger.info("_build_mp3_bytes: APIC tag added (%d bytes)", len(album_art_jpeg))

    buf.seek(0)
    tags.save(buf, v2_version=3)
    tagged_mp3 = buf.getvalue()

    logger.info(
        "_build_mp3_bytes: %d PCM bytes → %d MP3 bytes (tagged)",
        len(pcm_data), len(tagged_mp3),
    )
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

    Supports two narrator modes:
      narrator_mode = "solo"  — single-voice TTS using SOLO_HOST config
      narrator_mode = "duo"   — two-host MultiSpeakerMarkup TTS (default)

    Output behaviour is controlled by output_mode + gemini_enterprise in
    agent_configuration.json.

    Args:
        script:            Formatted script with speaker labels.
        podcast_title:     Episode title — filename, ID3 TIT2, and art text.
        narrator_mode:     "solo" for single narrator, "duo" for two hosts.
        topic_summary:     One-sentence summary for the art prompt.
        key_themes:        Comma-separated keywords for the art prompt.
        target_audience:   Audience description for art tone/style.
        write_to_gcs:      Legacy flag honoured when output_mode is "gcs".
        gcs_output_bucket: GCS bucket URI override (falls back to config default).
        output_dir:        Local directory for local mode.
        style_guidance:    Optional TTS director's note additions.
        tool_context:      Injected by ADK. Required for artifact output.

    Returns:
        dict with: success, output_path, output_url, mp3_gcs_url,
        artifact_saved, wav_artifact_saved, art_artifact_saved,
        art_filename, duration_seconds, duration_minutes,
        chunks_processed, model_used, narrator_mode, error.
    """
    try:
        safe_title       = re.sub(r"[^\w\-_]", "_", podcast_title)[:50]
        timestamp        = datetime.now().strftime("%Y%m%d_%H%M%S")
        mp3_filename     = f"{safe_title}_{timestamp}.mp3"
        wav_filename     = f"{safe_title}_{timestamp}.wav"
        art_filename     = f"{safe_title}_{timestamp}_cover.jpg"

        narrator_mode = (narrator_mode or "duo").lower().strip()
        is_solo = narrator_mode == "solo"

        logger.info(
            "generate_podcast_audio: narrator_mode=%s title='%s'",
            narrator_mode, podcast_title,
        )

        all_audio_frames: list[bytes] = []
        chunks_processed = 0

        # ── SOLO MODE ────────────────────────────────────────────────────────
        if is_solo:
            paragraphs   = _parse_script_to_paragraphs_solo(script)
            text_chunks  = _chunk_paragraphs(paragraphs, CHUNK_TARGET_WORDS)

            logger.info(
                "Processing %d TTS chunk(s) for '%s' (solo mode)",
                len(text_chunks), podcast_title,
            )

            # Gemini TTS single-speaker — uses name= + model_name= on
            # VoiceSelectionParams (no MultiSpeakerVoiceConfig needed).
            # The prompt field IS supported for single-speaker Gemini TTS,
            # so we pass director notes for style/pacing control.
            voice = texttospeech.VoiceSelectionParams(
                language_code="en-US",
                name=SOLO_HOST["voice"],
                model_name=_TTS_API_MODEL,
            )
            audio_config = texttospeech.AudioConfig(
                audio_encoding=texttospeech.AudioEncoding.LINEAR16,
                sample_rate_hertz=AUDIO_SAMPLE_RATE,
            )

            for i, text_chunk in enumerate(text_chunks):
                logger.info(
                    "  Generating audio for chunk %d/%d (solo)...",
                    i + 1, len(text_chunks),
                )

                director_notes = _build_director_notes(
                    podcast_title=podcast_title,
                    style_guidance=style_guidance,
                    chunk_index=i,
                    total_chunks=len(text_chunks),
                    narrator_mode="solo",
                )

                synthesis_input = texttospeech.SynthesisInput(
                    text=text_chunk,
                    prompt=director_notes,
                )

                response = tts_client.synthesize_speech(
                    input=synthesis_input,
                    voice=voice,
                    audio_config=audio_config,
                )

                all_audio_frames.append(response.audio_content)
                chunks_processed += 1

        # ── DUO MODE ─────────────────────────────────────────────────────────
        else:
            all_turns   = _parse_script_to_turns_duo(script)
            turn_chunks = _chunk_turns(all_turns, CHUNK_TARGET_WORDS)

            logger.info(
                "Processing %d TTS chunk(s) for '%s' (duo mode)",
                len(turn_chunks), podcast_title,
            )

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

            for i, turns in enumerate(turn_chunks):
                logger.info(
                    "  Generating audio for chunk %d/%d (duo)...",
                    i + 1, len(turn_chunks),
                )

                director_notes = _build_director_notes(
                    podcast_title=podcast_title,
                    style_guidance=style_guidance,
                    chunk_index=i,
                    total_chunks=len(turn_chunks),
                    narrator_mode="duo",
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

        # -- Assemble PCM, generate art, build WAV + MP3 ---------------------
        combined_pcm = b"".join(all_audio_frames)

        bytes_per_second = AUDIO_SAMPLE_RATE * AUDIO_CHANNELS * AUDIO_SAMPLE_WIDTH
        duration_seconds = len(combined_pcm) // bytes_per_second

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

        wav_bytes = _build_wav_bytes(combined_pcm)

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
                art_part = types.Part(
                    inline_data=types.Blob(data=album_art_jpeg, mime_type="image/jpeg")
                )
                await tool_context.save_artifact(filename=art_filename, artifact=art_part)
                art_artifact_saved = True
                logger.info("Album art saved as artifact '%s'", art_filename)

            if GEMINI_ENTERPRISE:
                bucket = (gcs_output_bucket or GCS_OUTPUT_BUCKET)
                if not bucket:
                    raise RuntimeError(
                        "gemini_enterprise is True but no GCS bucket is configured. "
                        "Set gcs_output_bucket in agent_configuration.json or pass it explicitly."
                    )
                bucket = bucket.rstrip("/")

                wav_part = types.Part(
                    inline_data=types.Blob(data=wav_bytes, mime_type="audio/wav")
                )
                wav_version = await tool_context.save_artifact(
                    filename=wav_filename, artifact=wav_part
                )
                wav_artifact_saved = True
                logger.info(
                    "WAV saved as artifact '%s' v%d (%.1f min)",
                    wav_filename, wav_version, duration_seconds / 60,
                )

                mp3_gcs_path = f"{bucket}/{mp3_filename}"
                mp3_gcs_url  = _upload_to_gcs(mp3_gcs_path, mp3_bytes, "audio/mpeg")
                logger.info("MP3 uploaded to GCS: %s", mp3_gcs_path)

                return {
                    "success":            True,
                    "output_path":        wav_filename,
                    "output_url":         "",
                    "mp3_gcs_url":        mp3_gcs_url,
                    "artifact_saved":     True,
                    "wav_artifact_saved": wav_artifact_saved,
                    "art_artifact_saved": art_artifact_saved,
                    "art_filename":       art_filename if art_artifact_saved else "",
                    "duration_seconds":   duration_seconds,
                    "duration_minutes":   round(duration_seconds / 60, 1),
                    "chunks_processed":   chunks_processed,
                    "model_used":         _TTS_API_MODEL,
                    "narrator_mode":      narrator_mode,
                    "error":              "",
                }

            else:
                mp3_part = types.Part(
                    inline_data=types.Blob(data=mp3_bytes, mime_type="audio/mpeg")
                )
                mp3_version = await tool_context.save_artifact(
                    filename=mp3_filename, artifact=mp3_part
                )
                logger.info(
                    "MP3 saved as artifact '%s' v%d (%.1f min)",
                    mp3_filename, mp3_version, duration_seconds / 60,
                )

                return {
                    "success":            True,
                    "output_path":        mp3_filename,
                    "output_url":         "",
                    "mp3_gcs_url":        "",
                    "artifact_saved":     True,
                    "wav_artifact_saved": False,
                    "art_artifact_saved": art_artifact_saved,
                    "art_filename":       art_filename if art_artifact_saved else "",
                    "duration_seconds":   duration_seconds,
                    "duration_minutes":   round(duration_seconds / 60, 1),
                    "chunks_processed":   chunks_processed,
                    "model_used":         _TTS_API_MODEL,
                    "narrator_mode":      narrator_mode,
                    "error":              "",
                }

        # ── GCS mode ─────────────────────────────────────────────────────────
        elif effective_mode == "gcs":
            bucket     = (gcs_output_bucket or GCS_OUTPUT_BUCKET).rstrip("/")
            gcs_uri    = f"{bucket}/{mp3_filename}"
            output_url = _upload_to_gcs(gcs_uri, mp3_bytes, "audio/mpeg")
            logger.info("MP3 uploaded to %s (%.1f min)", gcs_uri, duration_seconds / 60)
            return {
                "success":            True,
                "output_path":        gcs_uri,
                "output_url":         output_url,
                "mp3_gcs_url":        output_url,
                "artifact_saved":     False,
                "wav_artifact_saved": False,
                "art_artifact_saved": False,
                "art_filename":       "",
                "duration_seconds":   duration_seconds,
                "duration_minutes":   round(duration_seconds / 60, 1),
                "chunks_processed":   chunks_processed,
                "model_used":         _TTS_API_MODEL,
                "narrator_mode":      narrator_mode,
                "error":              "",
            }

        # ── Local mode ────────────────────────────────────────────────────────
        else:
            if output_dir is None:
                output_dir = os.environ.get("PODCAST_OUTPUT_DIR", OUTPUT_DIR)
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            local_path = os.path.join(output_dir, mp3_filename)
            with open(local_path, "wb") as f:
                f.write(mp3_bytes)
            logger.info("MP3 saved to %s (%.1f min)", local_path, duration_seconds / 60)
            return {
                "success":            True,
                "output_path":        os.path.abspath(local_path),
                "output_url":         "",
                "mp3_gcs_url":        "",
                "artifact_saved":     False,
                "wav_artifact_saved": False,
                "art_artifact_saved": False,
                "art_filename":       "",
                "duration_seconds":   duration_seconds,
                "duration_minutes":   round(duration_seconds / 60, 1),
                "chunks_processed":   chunks_processed,
                "model_used":         _TTS_API_MODEL,
                "narrator_mode":      narrator_mode,
                "error":              "",
            }

    except Exception as exc:
        logger.error("generate_podcast_audio failed: %s", exc)
        return {
            "success":            False,
            "output_path":        "",
            "output_url":         "",
            "mp3_gcs_url":        "",
            "artifact_saved":     False,
            "wav_artifact_saved": False,
            "art_artifact_saved": False,
            "art_filename":       "",
            "duration_seconds":   0,
            "duration_minutes":   0,
            "chunks_processed":   0,
            "model_used":         "",
            "narrator_mode":      narrator_mode if "narrator_mode" in dir() else "unknown",
            "error":              str(exc),
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
   - **Key themes** – 3–5 comma-separated keywords (e.g. "space exploration, NASA, Mars missions")
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

If the tool returns an error, report it clearly with suggestions and DO NOT retry — stop immediately and explain the issue to the user.
""",
    tools=[generate_podcast_audio],
    output_key="final_output",
)