# Paper diagnostic deployment — explicitly authorized scope

Operator approved: "Oui, diagnostic paper uniquement" on2026-09-23.
No strategy, risk, sizing, model, stage, trading schedule or order-path change.
No forced tick. This is not a profitability correction or a proof of alpha.

## Why this separate branch

Remote main was independently read via git ls-remote as
4a6d0143b6b4c5453701cbac462b06e0f384c185, matching origin/main. Research main
contained30 subsequent local commits. This branch starts from remote main,
so research modules, sizing/accounting changes and acquisitions are excluded.

Public production at2026-09-23T12:02:42Z: stage_0_paper, equity103.74060720787105,
8 closed, no open positions. Latest activity12:01:02UTC still has decision:null,
reason:null, side:'?'. This is neither proof of HOLD nor proof of a specific
failure. Read-only response+hash retained in research worktree under
data/prod_proof_kk7qdukt, SHA256
3ba00829a24301d472e2b65cea1041157851116bf582d1327870e95f27f74594.

Independent bounded reviewer verified legacy scheduler could serialize empty
state this way, including a returned critical-analysis failure. Missing history
can also abort analysis. Without server logs no root cause is established.
The public health endpoint reporting ok proves service responsiveness only.

## Change and acceptance

Port only the scheduler/API diagnostic behavior from6082660:
ABORTED for returned error before decision, INCOMPLETE for missing decision,
ERROR for an escaped tick exception; public reasons contain no exception text.
Normal approved/rejected decisions and their execution flags are preserved.
Chat context includes the last evaluation statuses. Existing dashboard activity
renderer already displays kind, decision and reason with escaping.

Add diagnostics_version=tick-status-v1 to health, so deployed code can be
checked without invoking the trading graph or waiting for its four-hour tick.
Version presence does not prove the later tick succeeded or any profit.

Tests must exercise the failure paths, verify no fake secret appears publicly,
and preserve normal decisions; mutation failures required. Run exact CI lint,
full tests and packaging build in an isolated environment before publishing.
CI image build and real health marker must then be checked separately.

## Deployment caveat

Repository secret listing returned no configured Actions secrets. Existing
workflow builds/pushes image but may rely on Dokploy's own auto-deploy; a green
workflow alone is not production proof. Never claim deployed until the actual
endpoint carries the marker. No server API credentials available locally.

Sonnet5 unavailable in current tool inventory; available-model subagent used
with explicit operator-facing disclosure. Single writer in isolated worktree;
parent handles acceptance review, commits and release.

## Local release acceptance

Exact CI lint passes. Full isolated suite:1,050passed,47skipped, one pre-existing
TestClient deprecation warning,59.34seconds. Skips are research/lake artifacts
absent from clean checkouts, not disabled diagnostic tests. Packaging build passes.
Worker reported13 mutation failures then restoration; parent independently ran
RED for missing execution-state context (four assertions failed), then final
full green suite. Separate read-only reviewer found no diagnostic-scope blocker.

Review correction: chat now explicitly distinguishes executed oui/non/inconnu,
including APPROVED with executed=False. Missing flags are not guessed false.
Tests cover these cases. All config, strategy, risk, execution, graph, agent,
dependency and workflow files remain byte-identical to origin/main.

Publishing does not itself prove deployment. Remote build and live marker
acceptance remain pending at this commit. Natural tick behavior must be inspected
after deployment; no tick is to be forced by this release procedure.
