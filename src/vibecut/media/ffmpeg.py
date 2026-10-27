"""ffmpeg/ffprobe discovery + bounded execution wrapper.

Discovery order: ``VIBECUT_FFMPEG``/``VIBECUT_FFPROBE`` env vars, then PATH.
No absolute machine paths are hard-coded anywhere in this package.
"""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from vibecut.core.proc import ProcResult, run_bounded


class FFmpegMissingError(RuntimeError):
    pass


def _resolve(explicit: Optional[str], env_name: str, name: str) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.is_file():
            return p.resolve()
        raise FFmpegMissingError(f"explicit tool path does not exist: {explicit}")
    env = os.environ.get(env_name)
    if env:
        p = Path(env)
        if p.is_file():
            return p.resolve()
        raise FFmpegMissingError(f"{env_name} points to a non-existent file: {env}")
    which = shutil.which(name)
    if which:
        return Path(which).resolve()
    return None


def find_tools(
    *, ffmpeg: Optional[str] = None, ffprobe: Optional[str] = None
) -> tuple[Optional[Path], Optional[Path]]:
    """Locate ffmpeg and ffprobe. Returns (path|None, path|None)."""
    return _resolve(ffmpeg, "VIBECUT_FFMPEG", "ffmpeg"), _resolve(ffprobe, "VIBECUT_FFPROBE", "ffprobe")


@dataclass(frozen=True)
class FFmpegTools:
    """ffmpeg/ffprobe executable pair.

    Immutable: the tool contract for caching is the raw version/build text
    (see :meth:`identity`), not a hash - hashing is only applied once at the
    cache-key layer, which already digests this text.
    """

    ffmpeg: Path
    ffprobe: Path

    @classmethod
    def discover(cls) -> "FFmpegTools":
        ffmpeg, ffprobe = find_tools()
        missing = [n for n, p in (("ffmpeg", ffmpeg), ("ffprobe", ffprobe)) if p is None]
        if missing:
            raise FFmpegMissingError(
                "required tool(s) not found: "
                + ", ".join(missing)
                + ". Install FFmpeg or set VIBECUT_FFMPEG / VIBECUT_FFPROBE."
            )
        return cls(ffmpeg=ffmpeg, ffprobe=ffprobe)  # type: ignore[arg-type]

    def run(self, argv: Sequence[str], *, timeout_s: float, cancel_event=None) -> ProcResult:
        return run_bounded(argv, timeout_s=timeout_s, cancel_event=cancel_event)

    def ffmpeg_argv(self, *args: str) -> tuple[str, ...]:
        return (str(self.ffmpeg), "-hide_banner", "-nostdin", "-loglevel", "error", *args)

    def ffprobe_argv(self, *args: str) -> tuple[str, ...]:
        return (str(self.ffprobe), "-v", "error", *args)

    def identity(self, *, timeout_s: float = 15.0) -> str:
        """Tool contract for cache keys: raw version/build text of both tools.

        Preserve version, compiler, configuration and library versions; two
        builds of the same release can provide different processing tools.
        """
        parts = []
        for exe in (self.ffmpeg, self.ffprobe):
            res = self.run((str(exe), "-version"), timeout_s=timeout_s)
            if res.returncode != 0:
                raise FFmpegMissingError(f"cannot run {exe}: rc={res.returncode}")
            lines = res.stdout_tail.decode("utf-8", "replace").splitlines()
            if not lines:
                raise FFmpegMissingError(f"{exe} -version produced no output")
            parts.append("\n".join(line.strip() for line in lines))
        return " | ".join(parts)
