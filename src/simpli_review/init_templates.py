"""Workflow templates written by `simpli-review init`.

They mirror examples/code-review.yml and examples/ai-scan.yml (a test keeps them in step). Three
placeholders are filled in by init.py: @@HEADING@@ (a YAML-safe scalar), @@USES@@ (the pinned
`owner/repo@sha # tag` text) and @@NOTE@@ (a comment line when the pin could not be resolved, else removed).
"""

from __future__ import annotations

CODE_REVIEW_TEMPLATE = """\
# Written by `simpli-review init`. Needs one secret: ANTHROPIC_API_KEY (repo or organization secret).
@@NOTE@@#
# A review runs when a PR is opened, reopened or marked ready for review.
# To ask for another review, add the `re-review` label to the PR. The reviewer
# removes the label when it finishes, so it can be added again. Create the
# label once per repository (any colour). Labels are used instead of comment
# triggers because an issue_comment trigger starts a (skipped) Actions run on
# every comment on every PR.
#
# Optional merge blocking: set "blockOn" in .simpli-review/config.json and make the
# job (named "review" below) a required status check. A PR with the
# `review-override` label (config "overrideLabel") is let through. The label is read
# from the PR the reviewer fetches, so the permissions below are enough. Adding that
# label does not start a review; add `re-review` afterwards to refresh the check.
name: @@HEADING@@

on:
  pull_request:
    # Add `synchronize` here AND "push" to autoReview in .simpli-review/config.json
    # to review every push; otherwise a push would only cancel a running review.
    types: [opened, reopened, ready_for_review, labeled]
    # Docs-only PRs are not reviewed (this also means a re-review label on a
    # docs-only PR does nothing).
    paths-ignore:
      - "**/*.md"
      - "docs/**"

permissions:
  contents: read
  pull-requests: write
  issues: write

jobs:
  review:
    # Skip drafts, and skip `labeled` events for any label other than
    # `re-review`. If you set "reReviewLabel" in .simpli-review/config.json,
    # use the same name here.
    if: >-
      github.event.pull_request.draft == false &&
      (github.event.action != 'labeled' || github.event.label.name == 're-review')
    # Job-level, so a skipped run never joins the group and cancels anything.
    concurrency:
      group: code-review-${{ github.event.pull_request.number }}
      cancel-in-progress: true
    runs-on: ubuntu-latest
    timeout-minutes: 30
    steps:
      # The reviewer reads the code at the PR's head commit. Its settings, rules and context files come
      # from the PR's base commit (through the GitHub API), so a PR cannot loosen its own review.
      - uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.event.pull_request.head.sha }}
          fetch-depth: 50          # some history for the git_history tool
          persist-credentials: false
      - uses: @@USES@@
        with:
          name: @@HEADING@@       # summary heading
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
"""

AI_SCAN_TEMPLATE = """\
# Monthly full-repository AI security scan (first Monday of the month), written by `simpli-review init`.
# Needs one secret: ANTHROPIC_API_KEY (repo or organization secret).
@@NOTE@@#
# What it does: reads the whole repository at the latest commit of the default branch, opens at most
# `scan.maxIssues` issues labelled `ai-scan` (created if missing) for the worst findings, and writes a
# spend report (tokens and an estimated cost in USD) to the job summary and to the `simpli-review-scan`
# artifact. Cost is capped by `scan.maxInputTokens` and `scan.maxTurns` in .simpli-review/config.json;
# run it by hand first (Actions > AI scan > Run workflow) to see the real spend.
name: AI scan

on:
  schedule:
    # Every Monday 09:17 UTC; the `when` job lets only the first Monday of the month through.
    # Cron cannot say "first Monday": "1-7 * 1" means day 1-7 OR Monday.
    - cron: "17 9 * * 1"
  workflow_dispatch:
    inputs:
      effort:
        description: "Reasoning effort: low, medium, high, xhigh or max. Empty uses scan.effort from the config (default high)."
        required: false
        default: ""
        type: string
      max-turns:
        description: "Upper bound on investigation turns. Empty uses scan.maxTurns from the config (default 60)."
        required: false
        default: ""
        type: string

permissions: {}

# Two scans never overlap: a manual run waits for the scheduled one instead of cancelling it.
concurrency:
  group: ai-scan
  cancel-in-progress: false

jobs:
  when:
    runs-on: ubuntu-latest
    timeout-minutes: 2
    outputs:
      run: ${{ steps.day.outputs.run }}
    steps:
      - name: First Monday of the month, or a manual run
        id: day
        env:
          EVENT: ${{ github.event_name }}
        run: |
          if [ "$EVENT" != "schedule" ] || [ "$(date -u +%-d)" -le 7 ]; then
            echo "run=true" >> "$GITHUB_OUTPUT"
          else
            echo "run=false" >> "$GITHUB_OUTPUT"
          fi

  scan:
    needs: when
    if: needs.when.outputs.run == 'true'
    permissions:
      contents: read
      issues: write
    runs-on: ubuntu-latest
    timeout-minutes: 45
    steps:
      - name: Check out the default branch
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          fetch-depth: 200 # history for the git_history tool
          persist-credentials: false
      - name: Scan
        uses: @@USES@@
        with:
          mode: scan
          name: AI scan # heading of the report
          anthropic-api-key: ${{ secrets.ANTHROPIC_API_KEY }}
          effort: ${{ inputs.effort }}
          max-turns: ${{ inputs.max-turns }}
      - name: Upload the scan report
        if: ${{ !cancelled() }} # a failed scan still spent money: keep its report
        uses: actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9 # v7.0.2
        with:
          name: simpli-review-scan
          path: |
            simpli-review-scan.json
            simpli-review-scan.md
          if-no-files-found: warn
          retention-days: 90
"""
