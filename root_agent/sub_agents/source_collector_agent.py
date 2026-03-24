# source_collector_agent.py
# Fetches and consolidates research content from Google Cloud Storage paths
# and free-form topics into a single research package for the script writer.
#
# All supported file types are passed to Gemini natively via Part.from_uri —
# no bytes are downloaded in the agent process for any format.
#
# Supported formats and their MIME types (see _EXT_TO_MIME):
#   .pdf   → application/pdf   (Gemini native PDF vision)
#   .txt   → text/plain
#   .md    → text/plain
#   .html  → text/html
#   .csv   → text/csv
#   .json  → application/json

import json
import logging
import os
from datetime import date
from pathlib import Path
from typing import Optional

from google.adk.agents import Agent
from google.adk.agents.callback_context import CallbackContext
from google.adk.models.llm_request import LlmRequest
from google.adk.models.llm_response import LlmResponse
from google.adk.tools import google_search
from google.adk.tools import load_artifacts
from google.adk.tools.agent_tool import AgentTool
from google.genai import types

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load Configuration
# ---------------------------------------------------------------------------

_CONFIG_PATH = Path(__file__).parent.parent / "agent_configuration.json"

with open(_CONFIG_PATH) as _f:
    _CFG = json.load(_f)

DATA_COLLECTOR_MODEL = _CFG["models"]["source_collector"]
SEARCH_AGENT_MODEL   = _CFG["models"]["search_agent"]

# ---------------------------------------------------------------------------
# MIME type mapping for supported formats
# All types are passed to Gemini natively via Part.from_uri in the callback —
# ---------------------------------------------------------------------------

_EXT_TO_MIME: dict[str, str] = {
    ".pdf":  "application/pdf",
    ".txt":  "text/plain",
    ".md":   "text/plain",
    ".html": "text/html",
    ".csv":  "text/csv",
    ".json": "application/json",
}

# ---------------------------------------------------------------------------
# GCS Document Fetcher Function Tool
# ---------------------------------------------------------------------------

def fetch_gcs_document(gcs_path: str) -> dict:
    """
    Validates a GCS document path and returns its URI and MIME type.

    No bytes are downloaded. The before_model_callback (_inject_gcs_parts)
    intercepts this tool result and injects a native types.Part.from_uri()
    into the outbound LLM request so Gemini reads the file directly from GCS.

    Supported formats: .pdf, .txt, .md, .html, .csv, .json

    Args:
        gcs_path: GCS URI in the format gs://bucket-name/path/to/file.

    Returns:
        A dict with keys:
          - success    (bool)
          - gcs_path   (str)
          - filename   (str)
          - mime_type  (str)   MIME type Gemini will use to read the file
          - file_type  (str)   file extension
          - error      (str)
    """
    try:
        if not gcs_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: '{gcs_path}'. Must start with gs://")

        blob_name = gcs_path[5:].split("/", 1)[1] if "/" in gcs_path[5:] else ""
        filename  = os.path.basename(blob_name)
        ext       = os.path.splitext(filename)[1].lower()
        mime_type = _EXT_TO_MIME.get(ext)

        if not mime_type:
            raise ValueError(
                f"Unsupported file type '{ext}' for '{filename}'. "
                f"Supported types: {', '.join(sorted(_EXT_TO_MIME))}"
            )

        bucket_name = gcs_path[5:].split("/", 1)[0]
        logger.info(
            "fetch_gcs_document: bucket=%s file=%s mime_type=%s",
            bucket_name,
            filename,
            mime_type,
        )

        return {
            "success":   True,
            "gcs_path":  gcs_path,
            "filename":  filename,
            "mime_type": mime_type,
            "file_type": ext,
            "error":     "",
        }

    except Exception as exc:
        logger.error("fetch_gcs_document failed for %s: %s", gcs_path, exc)
        return {
            "success":   False,
            "gcs_path":  gcs_path,
            "filename":  "",
            "mime_type": "",
            "file_type": "",
            "error":     str(exc),
        }


# ---------------------------------------------------------------------------
# Before-model callback — injects GCS file parts into the LLM request
# ---------------------------------------------------------------------------

# fetch_gcs_document returns only metadata (URI + MIME type) for every file.
# This callback scans tool results in the outbound request and, for each
# fetch_gcs_document result, replaces the raw JSON with a native
# types.Part.from_uri() so Gemini reads the file directly from GCS —
# no bytes downloaded, no extraction step, one consistent path for all types.

def _inject_gcs_parts(
    callback_context: CallbackContext,
    llm_request: LlmRequest,
) -> Optional[LlmResponse]:
    """
    Converts fetch_gcs_document tool results into native GCS file parts so
    Gemini reads file content directly via gs:// URI.

    - fetch_gcs_document results → types.Part.from_uri(gcs_path, mime_type)

    Uploaded files are handled separately by SaveFilesAsArtifactsPlugin +
    the load_artifacts tool — no bytes pass through this callback for uploads.
    """
    import json as _json

    for content in llm_request.contents or []:
        new_parts = []
        for part in content.parts or []:
            if part.text:
                try:
                    result = _json.loads(part.text)
                    if not (isinstance(result, dict) and result.get("success") and result.get("mime_type")):
                        raise ValueError("not a file tool result")

                    mime_type = result["mime_type"]
                    filename  = result.get("filename", "unknown")

                    # ── GCS file ──────────────────────────────────────────
                    if result.get("gcs_path", "").startswith("gs://"):
                        gcs_uri = result["gcs_path"]
                        logger.debug(
                            "_inject_gcs_parts: injecting GCS %s part for %s",
                            mime_type, gcs_uri,
                        )
                        new_parts.append(
                            types.Part.from_uri(file_uri=gcs_uri, mime_type=mime_type)
                        )
                        new_parts.append(
                            types.Part.from_text(
                                text=(
                                    f"[File loaded natively from GCS: {filename} "
                                    f"({gcs_uri}, {mime_type}). Read it in full and "
                                    "include all key points, figures, and details in "
                                    "your research.]"
                                )
                            )
                        )
                        continue  # drop the raw JSON tool-result part

                except (ValueError, _json.JSONDecodeError, TypeError, KeyError):
                    pass  # not a recognised file tool result — leave unchanged

            new_parts.append(part)

        content.parts = new_parts

    return None  # None = let the request proceed normally


# ---------------------------------------------------------------------------
# GCS Folder Lister Function Tool
# ---------------------------------------------------------------------------

def list_gcs_folder(gcs_folder_path: str, max_files: int = 50) -> dict:
    """
    Lists all files in a Google Cloud Storage folder (prefix).

    Only returns files with supported extensions: .pdf, .txt, .md, .html, .csv, .json.
    Unsupported file types are reported in skipped_files.

    Args:
        gcs_folder_path: GCS URI ending with / e.g. gs://my-bucket/docs/
        max_files: Maximum number of supported files to return (default 50).

    Returns:
        A dict with keys:
          - success           (bool)
          - gcs_folder_path   (str)
          - files             (list[str])  gs:// URIs for each supported file found
          - total_files       (int)
          - truncated         (bool)  True if more files exist than max_files
          - skipped_files     (list[str])  files excluded due to unsupported type
          - error             (str)
    """
    try:
        from google.cloud import storage

        if not gcs_folder_path.startswith("gs://"):
            raise ValueError(f"Invalid GCS path: '{gcs_folder_path}'")

        path_parts  = gcs_folder_path[5:].split("/", 1)
        bucket_name = path_parts[0]
        prefix      = path_parts[1] if len(path_parts) > 1 else ""

        client    = storage.Client()
        all_blobs = list(
            client.list_blobs(bucket_name, prefix=prefix, max_results=max_files * 5)
        )

        supported, skipped = [], []
        for b in all_blobs:
            if b.name.endswith("/"):
                continue  # skip folder pseudo-entries
            ext = os.path.splitext(b.name)[1].lower()
            uri = f"gs://{bucket_name}/{b.name}"
            if ext in _EXT_TO_MIME:
                supported.append(uri)
            else:
                skipped.append(uri)

        truncated = len(supported) > max_files
        files     = supported[:max_files]

        logger.info(
            "list_gcs_folder: bucket=%s prefix=%r — %d supported file(s)%s:\n%s",
            bucket_name,
            prefix,
            len(files),
            " (truncated)" if truncated else "",
            "\n".join(f"  {f}" for f in files),
        )
        if skipped:
            logger.warning(
                "list_gcs_folder: skipped %d unsupported file(s):\n%s",
                len(skipped),
                "\n".join(f"  {f}" for f in skipped),
            )

        return {
            "success":         True,
            "gcs_folder_path": gcs_folder_path,
            "files":           files,
            "total_files":     len(files),
            "truncated":       truncated,
            "skipped_files":   skipped,
            "error":           "",
        }

    except Exception as exc:
        logger.error("list_gcs_folder failed for %s: %s", gcs_folder_path, exc)
        return {
            "success":         False,
            "gcs_folder_path": gcs_folder_path,
            "files":           [],
            "total_files":     0,
            "truncated":       False,
            "skipped_files":   [],
            "error":           str(exc),
        }


# ---------------------------------------------------------------------------
# Google Search Sub-Agent | Used as AgentTool as Workaround
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Current date — computed once at import time and embedded in instructions
# so both the search agent and source collector are grounded to today.
# ---------------------------------------------------------------------------
_TODAY     = date.today()
_TODAY_STR = _TODAY.strftime("%B %d, %Y")   # e.g. "March 22, 2026"
_YEAR      = _TODAY.strftime("%Y")
_MONTH_YEAR = _TODAY.strftime("%B %Y")

search_agent = Agent(
    model=SEARCH_AGENT_MODEL,
    name="search_agent",
    instruction=f"""
    You are a specialist Google Search researcher.

    ## Date Context
    Today is {_TODAY_STR}. Use this date to correctly interpret any relative
    time references in the research request (e.g. "last week", "this month",
    "recent", "latest"). Always anchor search queries to real calendar dates
    and never use training data for time-sensitive topics.

    ## Language
    The research request will specify a target language for the podcast.
    Conduct your searches in that target language wherever possible — use
    search queries written in the target language so results are in the
    correct language and cultural context. For example, if the target
    language is French, search in French. If it is English, search in English.

    ## Your Process
    1. Analyse the research request. Identify any time references and convert
       them to explicit date ranges based on today's date:
       - "last week"  → the 7-day window ending yesterday
       - "this week"  → Monday through today
       - "this month" → {_MONTH_YEAR}
       - "recent" / "latest" → the past 7-14 days unless context suggests longer
    2. Construct targeted search queries that include the explicit date range
       or year (e.g. "top AI news {_YEAR}", "AI announcements {_MONTH_YEAR}").
    3. Run MULTIPLE searches — at minimum one broad query and two to three
       specific follow-up queries to capture the most important stories.
    4. Cross-reference results: if a story appears in multiple sources, flag
       it as higher-confidence. Note anything from only one source.
    5. Produce a comprehensive, factual overview citing sources and dates
       for every claim. Do NOT fill gaps with training data — if search
       returns no results for a claim, say so explicitly.

    ## Anti-Hallucination Rules
    - NEVER report events, releases, or announcements that did not appear
      in your actual search results for this session.
    - If results are sparse or ambiguous, note the limitation rather than
      supplementing with assumed knowledge.
    - Always include the publication date of sources when available.
    """,
    tools=[google_search],
)


# ---------------------------------------------------------------------------
# Podcast Source Research/Collector Agent
# ---------------------------------------------------------------------------
source_collector_agent = Agent(
    name="source_collector_agent",
    model=DATA_COLLECTOR_MODEL,
    description=(
        "Fetches and consolidates content from Google Cloud Storage "
        "paths, uploaded files, and free-form topics into a single research package."
    ),
    before_model_callback=_inject_gcs_parts,
    instruction=f"""
    You are a research agent whose research will be used to generate
    an insightful podcast. Your goal is to take the user-provided input
    of GCS bucket(s), GCS file(s), uploaded file(s), and/or free-form
    text about the podcast subject, and use your available tools to compose
    a very comprehensive research document as the output.

    ## Date Context
    Today is {_TODAY_STR}. Use this when interpreting any relative time
    references (e.g. "last week", "recent", "latest") and pass this date
    context explicitly when invoking the search_agent tool so it can
    construct correctly date-bounded search queries.

    ## Language Context
    The target podcast language is: {{language_name:English (United States)}}
    ({{language_code:en-US}}). Pass this to the search_agent so it runs
    queries in the correct language. If source documents are in a different
    language, still use them — the script writer will handle translation.

    Supported file types in GCS and uploads: .pdf, .txt, .md, .html, .csv, .json.
    Any other file types in a folder will be automatically skipped.

    When the user has uploaded files directly to the chat:
      1. ALWAYS call load_artifacts first to load the uploaded file content
         into your context. This is the ONLY way to read uploaded files —
         do NOT infer, guess, or use the filename to determine content.
      2. Read each loaded file in full.
      3. For each file, produce a comprehensive overview covering ALL key
         facts, figures, specs, model names, years, and details found in
         the actual file content. Never substitute information from your
         training data for what is in the file.

    When a GCS bucket or folder is provided:
      1. Call list_gcs_folder to enumerate all supported files in the folder.
      2. Call fetch_gcs_document for each file, one by one.
         All file types are loaded natively into your context via GCS URI.
         Read each file in full.
      3. For each document, produce a verbose and comprehensive overview
         covering key points, figures, quotes, and interesting details.
      4. Repeat until all files are processed.

    When a GCS document path is provided directly (not a folder):
      1. Call fetch_gcs_document for the path.
      2. Read the file in full and produce a comprehensive overview.

    When the request includes free-form text topics:
      - Pass the full topic to the search_agent tool INCLUDING any time
        references, today's date ({_TODAY_STR}), AND the target language
        so the search agent can construct correctly date-bounded queries
        in the right language.
        Example: instead of passing "top AI news last week", pass
        "top AI news from the week of [dates]. Today is {_TODAY_STR}.
        Target language: {{language_name:English (United States)}}."
      - Run the search_agent at least twice for time-sensitive topics:
        once for the broad topic and once for specific stories or follow-ups.
      - Produce a comprehensive and verbose overview of the findings.

    CRITICAL: All facts, names, model numbers, years, and specifications in
    your research output MUST come from the actual loaded file content or
    search results — never from assumptions, filenames, or training data.
    If search results do not cover a claimed fact, omit it entirely.

    Return all overviews combined as your output. This content will be
    consolidated and turned into a podcast script by another agent.
    """,
    tools=[
        AgentTool(agent=search_agent),
        fetch_gcs_document,
        list_gcs_folder,
        load_artifacts,
    ],
    output_key="generated_research",
)