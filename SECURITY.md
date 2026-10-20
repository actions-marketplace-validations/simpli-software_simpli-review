# Security policy

## Reporting a vulnerability

Please report vulnerabilities privately through GitHub's private vulnerability reporting:
open the **Security** tab of this repository and choose **Report a vulnerability**. Please do
not open a public issue or pull request for a security problem.

Include what you found, how to reproduce it, and the version or commit you tested.

## What to expect

This project is maintained on a best-effort basis with no SLA. We will read reports and reply
when we can, and we will credit you in the fix if you want.

## Scope

In scope: this repository's code and the GitHub Action it publishes, for example a way to make
the reviewer leak a secret value or the API key into a log, comment, issue or report; to make it
run attacker-controlled code on the runner; to read files outside the checked-out repository or
secret files through its read-only tools; or to make it post to a pull request or open issues
without the permissions the workflow grants.

Out of scope: vulnerabilities in the Anthropic API or in GitHub, the content of a review (it is
model output and can be wrong), and findings the reviewer reports about your own code.
