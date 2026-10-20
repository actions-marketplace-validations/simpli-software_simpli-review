"""Where the review settings are read from.

A pull request can edit any file it contains, so the files that steer the review (settings, rules,
context files, the PR template) must not come from the PR's own commit. In PR mode they are read
from the PR's base commit: through the GitHub contents API first (works with a shallow checkout
and `persist-credentials: false`), then `git show` when the commit is present locally. In local
mode they are read from the `--base` ref. Scan mode reads the checked-out default branch.

Reads never fall back to the working tree. A failure raises ConfigSourceError and the caller
fails the review.
"""

from __future__ import annotations

import posixpath
import re
import subprocess
from pathlib import Path
from typing import Any, Protocol

import httpx

from simpli_review.github import GitHubError

_SHA = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")


class ConfigSourceError(Exception):
    """The review settings could not be read from their trusted source."""


class FileSource(Protocol):
    label: str  # shown in messages and in the --json output

    def read(self, path: str) -> str | None:
        """The file's text, or None when it does not exist there. Raises ConfigSourceError otherwise."""
        ...


def clean_path(path: str) -> str | None:
    """A repo-relative POSIX path, or None for anything that points outside the repository."""
    cleaned = posixpath.normpath(path.strip().replace("\\", "/"))
    if cleaned in ("", ".") or cleaned.startswith(("/", "../")) or cleaned == "..":
        return None
    return cleaned


class DirSource:
    """A directory on disk: the checked-out tree (scan mode, tests)."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.label = "working tree"

    def read(self, path: str) -> str | None:
        rel = clean_path(path)
        if rel is None:
            return None
        target = self.root / rel
        if not target.is_file():
            return None
        try:
            return target.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as e:
            raise ConfigSourceError(f"cannot read {rel} ({type(e).__name__})") from e


class GitSource:
    """Files of one commit, read with `git show`. `ref` may be a branch name or a commit id."""

    def __init__(self, root: Path, ref: str) -> None:
        self.root = root
        self.ref = ref
        self.commit = ""
        self.label = ref
        if not ref or ref.startswith("-") or any(c in ref for c in "\0\n"):
            raise ConfigSourceError("invalid base reference")

    def verify(self) -> GitSource:
        out = self._git("rev-parse", "--verify", "--quiet", f"{self.ref}^{{commit}}")
        commit = out.strip()
        if not _SHA.fullmatch(commit):
            raise ConfigSourceError(f"the base commit {self.ref[:12]} is not available in this checkout")
        self.commit = commit
        self.label = f"{self.ref}@{commit[:7]}"
        return self

    def read(self, path: str) -> str | None:
        rel = clean_path(path)
        if rel is None:
            return None
        if not self.commit:
            raise ConfigSourceError("base commit was not verified")
        for line in self._git("ls-tree", "-z", self.commit, "--", rel).split("\0"):
            meta, _, name = line.partition("\t")
            fields = meta.split()
            if name == rel and len(fields) == 3 and fields[1] == "blob" and fields[0] != "120000":
                return self._git("show", f"{self.commit}:{rel}")
        return None

    def _git(self, *args: str) -> str:
        try:
            done = subprocess.run(["git", *args], cwd=self.root, capture_output=True, timeout=30, check=False)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise ConfigSourceError(f"git failed ({type(e).__name__})") from e
        if done.returncode != 0:
            raise ConfigSourceError(f"git {args[0]} failed")
        try:
            return done.stdout.decode("utf-8")
        except UnicodeDecodeError as e:
            raise ConfigSourceError("a settings file is not valid UTF-8") from e


class ApiSource:
    """Files of one commit of the repository under review, through the GitHub contents API."""

    def __init__(self, gh: Any, commit: str) -> None:
        if not _SHA.fullmatch(commit):
            raise ConfigSourceError("the base commit id is missing or malformed")
        self.gh = gh
        self.commit = commit
        self.label = f"base@{commit[:7]}"

    def verify(self) -> ApiSource:
        """Confirm the commit is readable, so a 404 on a path can only mean the file is absent."""
        try:
            self.gh.commit_exists(self.commit)
        except (GitHubError, httpx.HTTPError) as e:
            raise ConfigSourceError(f"GitHub API cannot read the base commit ({_why(e)})") from e
        return self

    def read(self, path: str) -> str | None:
        rel = clean_path(path)
        if rel is None:
            return None
        try:
            return self.gh.file_at(rel, self.commit)  # type: ignore[no-any-return]
        except (GitHubError, httpx.HTTPError, ValueError) as e:
            raise ConfigSourceError(f"cannot read {rel} at the base commit ({_why(e)})") from e


class CachingSource:
    """Remembers every answer, so a file is fetched once and the answer cannot change mid-run."""

    def __init__(self, inner: FileSource) -> None:
        self.inner = inner
        self.label = inner.label
        self._seen: dict[str, str | None] = {}

    def read(self, path: str) -> str | None:
        if path not in self._seen:
            self._seen[path] = self.inner.read(path)
        return self._seen[path]


def open_base_source(gh: Any, repo_root: Path, base_sha: str) -> CachingSource:
    """The PR's base commit: the GitHub API when it works, else git when the commit is local."""
    problems: list[str] = []
    try:
        return CachingSource(ApiSource(gh, base_sha).verify())
    except ConfigSourceError as e:
        problems.append(str(e))
    try:
        return CachingSource(GitSource(repo_root, base_sha).verify())
    except ConfigSourceError as e:
        problems.append(str(e))
    raise ConfigSourceError("; ".join(problems))


def _why(e: Exception) -> str:
    return f"HTTP {e.status}" if isinstance(e, GitHubError) else type(e).__name__
