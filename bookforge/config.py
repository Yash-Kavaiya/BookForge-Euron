"""Central, env-driven configuration for BookForge.

All LLM calls route through the Euron EURI API (OpenAI-compatible).  Set EURI_API_KEY to enable.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

QaCriticMode = Literal["once", "always", "never"]


VISUAL_CAPS: dict[str, dict[str, int]] = {
    "light": {"diagrams": 1, "frames": 3, "illustrations": 0, "charts": 2},
    "standard": {"diagrams": 2, "frames": 4, "illustrations": 1, "charts": 4},
    "rich": {"diagrams": 4, "frames": 5, "illustrations": 2, "charts": 6},
}


class Settings(BaseSettings):
    """Runtime configuration. Every field maps to BOOKFORGE_<NAME> env vars."""

    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="BOOKFORGE_", extra="ignore"
    )

    # --- EURI API (OpenAI-compatible) ---
    euri_api_base: str = "https://api.euron.one/api/v1/euri"
    euri_api_key: str = Field(
        default="",
        validation_alias=AliasChoices("EURI_API_KEY", "BOOKFORGE_EURI_API_KEY"),
        description="Bearer key for the EURI API (never hardcode)",
    )

    @property
    def euri_configured(self) -> bool:
        """True when the EURI API key is available."""
        return bool(self.euri_api_key)

    # --- Model selection ---
    analyst_model: str = "euri-chat-fast"
    writer_model: str = "euri-chat-smart"
    critic_model: str = "euri-chat-fast"
    refiner_model: str = ""  # empty -> same as writer_model

    # --- LLM limits ---
    llm_max_output_tokens: int = 8192  # EURI max output cap
    llm_max_concurrency: int = 2  # EURI: 2 concurrent requests
    # EURI models have a 32k-token context. Non-Latin scripts (e.g. Hindi
    # auto-captions) cost ~2-3 chars/token, so cap the transcript the analyst
    # sees; long videos are sampled evenly across their whole length.
    transcript_max_chars: int = 36_000

    # --- Book metadata ---
    book_title: str = ""  # empty -> "<channel/playlist> — The Complete Book"
    book_subtitle: str = ""
    book_author: str = ""  # empty -> "Compiled by BookForge-Euron ..."

    # --- Branding (header / footer / logos / cover) --------------------------
    # Empty header/footer text keeps the defaults: chapter title in the header,
    # channel name (left) and the chapter's video link (right) in the footer.
    # The page number is always in the centre of the footer.
    header_left: str = ""
    header_right: str = ""
    footer_text: str = ""
    # Logo files (PNG/JPG/PDF/SVG). `logo` is used for every position listed
    # in `logo_positions` unless a position-specific logo is given.
    logo: str = ""
    logo_title: str = ""
    logo_header: str = ""
    logo_footer: str = ""
    logo_positions: str = "title,header"  # comma list of title|header|footer
    # Cover: an existing image file, else generate one with the image model.
    cover_image: str = ""
    cover_generate: bool = True
    cover_prompt: str = ""  # empty -> built from the book title
    image_model: str = "euri-image-fast"
    list_of_figures: bool = True

    # --- Visuals ---------------------------------------------------------------
    # light: 1 diagram, 3 screenshots, no illustrations; standard: 2 diagrams,
    # 4 screenshots, 1 illustration; rich: 4 diagrams, 5 screenshots,
    # 2 illustrations per chapter. Charts appear only for numbers stated in
    # the transcript, at every level.
    visuals: Literal["light", "standard", "rich"] = "standard"
    diagram_engine: Literal["tikz", "graphviz", "auto"] = "tikz"
    illustrations_enabled: bool = True

    # --- Source selection (channel URLs) ---------------------------------------
    channel_sort: Literal["newest", "popular"] = "newest"
    channel_group_by_playlists: bool = True
    channel_max_playlists: int = 25

    # Low temperatures keep extraction/judgement deterministic and grounded;
    # the writer gets a little more room for prose.
    analyst_temperature: float = 0.1
    writer_temperature: float = 0.3
    critic_temperature: float = 0.0

    # --- Media pipeline ----------------------------------------------------
    frame_interval_sec: int = 5  # screenshot cadence (per the flow)
    frame_phash_threshold: int = 6  # hamming distance <= -> duplicate
    max_frames_per_chapter: int = 0  # 0 -> chosen by `visuals` level
    download_resolution: int = 480  # capped video height
    whisper_model: str = "base"  # faster-whisper size (fallback STT)

    # --- Scope / safety rails ----------------------------------------------
    # "auto": by upload_date when known; "playlist": keep YouTube order;
    # "reverse": flip newest-first playlists (flat extraction has no dates).
    video_order: Literal["auto", "playlist", "reverse"] = "auto"
    max_videos: int | None = None  # None = entire channel
    min_video_duration_sec: int = 30  # skip shorts/trailers
    max_video_duration_sec: int = 3 * 3600

    # --- Quality loop ------------------------------------------------------
    qa_max_iterations: int = 3
    compile_latex: bool = True  # disable on hosts without pdflatex
    # When to spend an LLM call on the critic. Deterministic machine checks
    # (structure, asset refs, labels, unsupported numbers, equations, compile)
    # always run first and gate the LLM critic:
    #   once   - LLM critic judges the chapter the first time machine checks
    #            pass; later machine passes (after a refine) are accepted.
    #   always - every machine pass is also reviewed by the LLM critic.
    #   never  - machine checks alone decide (fastest).
    qa_llm_critic_mode: QaCriticMode = "once"
    # Numbers in chapter prose must appear in the analysis or transcript.
    # Set False to demote the check from "major" to "minor" (never blocks).
    qa_strict_numbers: bool = True

    # --- Parallelization ---------------------------------------------------
    max_concurrent_videos: int = 1  # kept for compatibility; ADK graph is sequential
    enable_parallel_videos: bool = False
    media_prefetch_concurrency: int = 3  # upcoming videos downloaded/transcribed in parallel
    frame_hash_workers: int = 4  # threads hashing/scoring frames during dedupe
    asset_render_workers: int = 4  # threads rendering charts/tables per chapter

    # --- Reuse across reruns -----------------------------------------------
    # analysis.json is cached per chapter (keyed by transcript hash) so a
    # resumed run never pays for the analyst twice; a chapter that was
    # written but failed QA resumes in the QA loop instead of being rewritten.
    reuse_cached_analysis: bool = True
    reuse_written_chapters: bool = True

    # --- Langfuse ----------------------------------------------------------
    langfuse_public_key: str = ""  # set via BOOKFORGE_LANGFUSE_PUBLIC_KEY
    langfuse_secret_key: str = ""  # set via BOOKFORGE_LANGFUSE_SECRET_KEY
    langfuse_host: str = "https://cloud.langfuse.com"  # set via BOOKFORGE_LANGFUSE_HOST
    langfuse_enabled: bool = False  # set via BOOKFORGE_LANGFUSE_ENABLED

    # --- Housekeeping ------------------------------------------------------
    workspace_root: Path = Path("data")
    log_level: str = "INFO"

    @property
    def workspace_root_abs(self) -> Path:
        return self.workspace_root.expanduser().resolve()

    @property
    def visual_caps(self) -> dict[str, int]:
        """Per-chapter visual budget for the configured `visuals` level."""
        caps = VISUAL_CAPS[self.visuals]
        if self.max_frames_per_chapter > 0:
            caps = {**caps, "frames": self.max_frames_per_chapter}
        if not self.illustrations_enabled:
            caps = {**caps, "illustrations": 0}
        return caps

    @property
    def effective_refiner_model(self) -> str:
        return self.refiner_model or self.writer_model

    def model_for_stage(self, stage: str) -> str:
        """Model name for a pipeline stage: analyst | writer | refiner | critic."""
        return {
            "analyst": self.analyst_model,
            "writer": self.writer_model,
            "refiner": self.effective_refiner_model,
            "critic": self.critic_model,
        }[stage]

    def temperature_for_stage(self, stage: str) -> float:
        return {
            "analyst": self.analyst_temperature,
            "writer": self.writer_temperature,
            "refiner": self.writer_temperature,
            "critic": self.critic_temperature,
        }[stage]

    @field_validator("max_videos")
    @classmethod
    def max_videos_positive(cls, v: int | None) -> int | None:
        if v is not None and v < 1:
            raise ValueError("max_videos must be >= 1")
        return v


@lru_cache
def get_settings() -> Settings:
    return Settings()


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure structured logging for BookForge."""
    import logging

    if json_format:
        fmt = '{"time":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","message":"%(message)s"}'
    else:
        fmt = "%(asctime)s %(levelname)-7s [%(name)s] %(message)s"

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    # Quiet noisy third-party loggers
    for noisy in ("urllib3", "httpcore", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
