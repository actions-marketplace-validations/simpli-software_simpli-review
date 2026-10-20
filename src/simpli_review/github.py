"""The GitHub REST calls the reviewer needs, and nothing else.

Auth is the workflow's GITHUB_TOKEN (needs `pull-requests: write`,
`issues: write`, `contents: read`). Request/response bodies are never logged.
"""

from __future__ import annotations

import logging
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

API = "https://api.github.com"


class GitHubError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"GitHub API {status}: {message}")
        self.status = status


class GitHub:
    def __init__(self, token: str, repo: str, api: str = API, client: httpx.Client | None = None) -> None:
        self.repo = repo
        self.http = client or httpx.Client(
            base_url=api,
            timeout=30,
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            },
        )

    # ---------- plumbing ----------

    def _req(self, method: str, path: str, **kw: Any) -> Any:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                msg = r.json().get("message", r.reason_phrase)
            except ValueError:
                msg = r.reason_phrase
            raise GitHubError(r.status_code, str(msg))
        return r.json() if r.content else None

    def _paged(self, path: str, limit: int, **params: Any) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        page = 1
        while len(out) < limit:
            batch = self._req("GET", path, params={**params, "per_page": 100, "page": page})
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page += 1
        return out[:limit]

    # ---------- reads ----------

    def pull(self, number: int) -> dict[str, Any]:
        result: dict[str, Any] = self._req("GET", f"/repos/{self.repo}/pulls/{number}")
        return result

    def pull_files(self, number: int) -> list[dict[str, Any]]:
        return self._paged(f"/repos/{self.repo}/pulls/{number}/files", limit=3000)

    def pull_review_comments(self, number: int) -> list[dict[str, Any]]:
        return self._paged(f"/repos/{self.repo}/pulls/{number}/comments", limit=1000)

    def resolved_comment_ids(self, number: int) -> set[int]:
        """Ids of review comments whose thread is resolved (GraphQL; REST has no such field)."""
        owner, _, name = self.repo.partition("/")
        query = (
            "query($o:String!,$n:String!,$p:Int!,$c:String){repository(owner:$o,name:$n){"
            "pullRequest(number:$p){reviewThreads(first:100,after:$c){pageInfo{hasNextPage endCursor}"
            "nodes{isResolved comments(first:100){nodes{databaseId}}}}}}}"
        )
        out: set[int] = set()
        cursor: str | None = None
        for _ in range(5):  # at most 500 threads
            data = self._req(
                "POST",
                "/graphql",
                json={"query": query, "variables": {"o": owner, "n": name, "p": number, "c": cursor}},
            )
            if data.get("errors"):
                raise GitHubError(200, "GraphQL query failed")
            threads = data["data"]["repository"]["pullRequest"]["reviewThreads"]
            for t in threads["nodes"]:
                if t["isResolved"]:
                    out.update(int(c["databaseId"]) for c in t["comments"]["nodes"])
            if not threads["pageInfo"]["hasNextPage"]:
                break
            cursor = threads["pageInfo"]["endCursor"]
        return out

    def recent_repo_review_comments(self, limit: int = 300) -> list[dict[str, Any]]:
        """Inline comments across all PRs, newest first, with reaction counts -
        the raw material for learning what the team accepts or rejects."""
        return self._paged(
            f"/repos/{self.repo}/pulls/comments", limit=limit, sort="created", direction="desc"
        )

    def issue_comments(self, number: int) -> list[dict[str, Any]]:
        return self._paged(f"/repos/{self.repo}/issues/{number}/comments", limit=1000)

    def labeled_issues(self, label: str, limit: int = 1000) -> list[dict[str, Any]]:
        """Open and closed issues (not pull requests) with `label`; closed ones carry `state_reason`."""
        raw = self._paged(f"/repos/{self.repo}/issues", limit=limit, labels=label, state="all")
        return [i for i in raw if "pull_request" not in i]

    # ---------- writes ----------

    def ensure_label(self, name: str, color: str, description: str) -> None:
        """Create the label unless it exists."""
        try:
            self._req("GET", f"/repos/{self.repo}/labels/{quote(name, safe='')}")
        except GitHubError as e:
            if e.status != 404:
                raise
            try:
                self._req(
                    "POST",
                    f"/repos/{self.repo}/labels",
                    json={"name": name, "color": color, "description": description},
                )
            except GitHubError as e2:
                if e2.status != 422:  # 422: created by a concurrent run
                    raise

    def create_issue(self, title: str, body: str, labels: list[str]) -> dict[str, Any]:
        issue: dict[str, Any] = self._req(
            "POST", f"/repos/{self.repo}/issues", json={"title": title, "body": body, "labels": labels}
        )
        return issue

    def create_review(self, number: int, commit_id: str, body: str, comments: list[dict[str, Any]]) -> None:
        self._req(
            "POST",
            f"/repos/{self.repo}/pulls/{number}/reviews",
            json={"commit_id": commit_id, "body": body, "event": "COMMENT", "comments": comments},
        )

    def upsert_issue_comment(self, number: int, marker: str, body: str) -> None:
        for c in self.issue_comments(number):
            if marker in (c.get("body") or ""):
                self._req("PATCH", f"/repos/{self.repo}/issues/comments/{c['id']}", json={"body": body})
                return
        self._req("POST", f"/repos/{self.repo}/issues/{number}/comments", json={"body": body})

    def remove_label(self, number: int, label: str) -> None:
        """Remove a label from a PR; already gone is fine."""
        try:
            self._req("DELETE", f"/repos/{self.repo}/issues/{number}/labels/{quote(label, safe='')}")
        except GitHubError as e:
            if e.status != 404:
                raise

    def set_status_reaction(self, number: int, content: str, me: str | None) -> None:
        """👀 while running, then 👍 or 😕 (the status signal). Our own
        earlier status reactions are removed so only the latest shows."""
        status = {"eyes", "+1", "confused"}
        try:
            for r in self._req(
                "GET", f"/repos/{self.repo}/issues/{number}/reactions", params={"per_page": 100}
            ):
                if r.get("content") in status and (me is None or r.get("user", {}).get("login") == me):
                    self._req("DELETE", f"/repos/{self.repo}/issues/{number}/reactions/{r['id']}")
            self._req("POST", f"/repos/{self.repo}/issues/{number}/reactions", json={"content": content})
        except GitHubError as e:
            # A status emoji is cosmetic; never fail a review over it.
            log.warning("could not set status reaction (%s)", e.status)
