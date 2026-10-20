from __future__ import annotations

import json
from pathlib import Path

from simpli_review.config import Config, Rule, load_config, match_glob
from simpli_review.diff import nearest_commentable, parse_patch, split_unified_diff
from simpli_review.models import Category, CommentType, Finding, Severity

PATCH = """@@ -3,6 +3,7 @@ def charge(amount):
 def charge(amount):
-    cents = amount * 100
+    cents = to_cents(amount)
+    log(cents)
     return send(cents)


@@ -20,2 +21,2 @@ def tail():
-    old()
+    new()
     keep()"""


def test_parse_patch_tracks_new_side_lines():
    fp = parse_patch("app/billing.py", PATCH)
    assert fp.added == {4, 5, 21}
    assert fp.commentable == {3, 4, 5, 6, 7, 8, 21, 22}
    assert (fp.additions, fp.deletions) == (3, 2)


def test_annotated_gutter_shows_new_line_numbers_and_blanks_removed_lines():
    lines = parse_patch("x.py", PATCH).annotated().splitlines()
    assert lines[1] == "     3  def charge(amount):"
    assert lines[2].startswith("       -")  # removed: no number
    assert lines[3] == "     4 +    cents = to_cents(amount)"


def test_nearest_commentable_snaps_within_three_lines_only():
    fp = parse_patch("x.py", PATCH)
    assert nearest_commentable(fp, 5) == 5
    assert nearest_commentable(fp, 10) == 8
    assert nearest_commentable(fp, 15) is None


def test_split_unified_diff_handles_new_and_renamed_files():
    diff = (
        "diff --git a/a.py b/a.py\nnew file mode 100644\nindex 0000..1111\n--- /dev/null\n+++ b/a.py\n"
        "@@ -0,0 +1,2 @@\n+x = 1\n+y = 2\n"
        "diff --git a/old.py b/new.py\nsimilarity index 90%\nrename from old.py\nrename to new.py\n"
        "--- a/old.py\n+++ b/new.py\n@@ -1 +1 @@\n-a\n+b\n"
    )
    a, b = split_unified_diff(diff)
    assert (a.path, a.status, a.added) == ("a.py", "added", {1, 2})
    assert (b.path, b.status, b.previous_path, b.added) == ("new.py", "renamed", "old.py", {1})


def test_globs():
    assert match_glob("web/node_modules/x/y.js", "node_modules/**")
    assert match_glob("deep/dir/package-lock.json", "package-lock.json")
    assert match_glob("app/api/routes.py", "app/**/*.py")
    assert not match_glob("app/api/routes.py", "web/**")


def _finding(sev: Severity, typ: CommentType, conf: int = 4) -> Finding:
    return Finding(path="a.py", line=1, severity=sev, type=typ, category=Category.correctness, title="t",
                   body="b", evidence="a.py:1", confidence=conf)  # fmt: skip


def test_strictness_and_type_filters():
    default = Config()
    assert default.keeps(_finding(Severity.P2, CommentType.logic))
    assert not default.keeps(_finding(Severity.P2, CommentType.style))
    assert not default.keeps(_finding(Severity.P0, CommentType.logic, conf=2))
    strict = Config(strictness=3)
    assert strict.keeps(_finding(Severity.P0, CommentType.logic))
    assert not strict.keeps(_finding(Severity.P1, CommentType.logic))
    no_style = Config(commentTypes=[CommentType.logic])
    assert not no_style.keeps(_finding(Severity.P0, CommentType.syntax))


def test_should_review_filters():
    c = Config(excludeAuthors=["dependabot[bot]"], disabledLabels=["no-review"])
    assert c.should_review(draft=True, author="a", labels=[])[0] is False
    assert c.should_review(draft=False, author="dependabot[bot]", labels=[])[0] is False
    assert c.should_review(draft=False, author="a", labels=["no-review"])[0] is False
    assert c.should_review(draft=False, author="a", labels=[]) == (True, "")


def test_rules_scope():
    c = Config(rules=[Rule(id="cents", rule="money in cents", scope=["app/**"]),
                      Rule(id="off", rule="x", enabled=False)])  # fmt: skip
    assert [r.id for r in c.rules_for(["app/billing.py"])] == ["cents"]
    assert c.rules_for(["web/a.ts"]) == []


def test_load_config_reads_folder_and_default_context_files(repo: Path):
    folder = repo / ".simpli-review"
    folder.mkdir()
    (folder / "config.json").write_text(json.dumps({"strictness": 3, "autoReview": ["open", "push"]}))
    (folder / "rules.md").write_text("Never log emails.")
    (folder / "files.json").write_text(json.dumps({"files": [{"path": "docs/x.md", "description": "d"}]}))
    c = load_config(repo)
    assert c.strictness == 3
    assert c.auto_review == ["open", "push"]
    assert "Never log emails." in c.instructions
    assert [f.path for f in c.files] == ["docs/x.md", "CLAUDE.md"]


def test_load_config_without_folder_uses_defaults(tmp_path: Path):
    c = load_config(tmp_path)
    assert c.strictness == 2 and c.auto_review == ["open"] and c.files == []
    assert c.ignored("package-lock.json") and not c.ignored("app/x.py")
