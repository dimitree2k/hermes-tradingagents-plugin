"""tradingagents_analyze tool — runs TradingAgents (Docker or a local
checkout) so Hermes can request multi-symbol trading analyses.

TradingAgents (https://github.com/TauricResearch/TradingAgents) ships an
interactive-only CLI; there is no HTTP API to call. Instead this tool runs
a small non-interactive batch entry point (``scripts/batch_analyze.py``)
that must exist in the target checkout — see this plugin's README for the
one-file addition required on the TradingAgents side — either inside its
Docker container (``TRADINGAGENTS_EXEC_MODE=docker``, the default) or with
a plain Python interpreter against a local, non-Docker checkout
(``TRADINGAGENTS_EXEC_MODE=local``).

Config (environment variables, e.g. in ~/.hermes/.env or the process env):

    TRADINGAGENTS_DIR               Path to a TradingAgents checkout that
                                     contains scripts/batch_analyze.py (and,
                                     for docker mode, docker-compose.yml).
                                     Required.
    TRADINGAGENTS_EXEC_MODE          "docker" (default) or "local".
    TRADINGAGENTS_COMPOSE_SERVICE   docker mode only. Compose service to
                                     run. Default: "tradingagents".
    TRADINGAGENTS_PYTHON             local mode only. Python interpreter to
                                     run the batch script with — point this
                                     at the venv/conda env TradingAgents is
                                     installed in, e.g.
                                     "/path/to/.venv/bin/python". Default:
                                     "python3" (whatever that resolves to
                                     on PATH).
    TRADINGAGENTS_WATCHLIST         Comma-separated default tickers used when
                                     the tool is called without `tickers` AND
                                     no watchlist has been saved yet via the
                                     dashboard panel (see dashboard/). Once a
                                     dashboard watchlist exists, it wins.
    TRADINGAGENTS_TIMEOUT_SECONDS   Per-call subprocess timeout. Default: 3600.

``diagnose()`` is the single source of truth for "is this reachable" — the
tool's own preflight check and the dashboard's connectivity status card
(``dashboard/plugin_api.py::GET /status``) both call it, so they can never
disagree about whether a run would actually work.

Every run (success or failure, dashboard-triggered or not) is recorded via
``store.py`` so the dashboard panel can show the latest decision per ticker
and link to the full report — see ``dashboard/plugin_api.py``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, List

from tools.registry import tool_error, tool_result

from . import screener, store

logger = logging.getLogger(__name__)

_TICKER_RE = re.compile(r"^[A-Za-z0-9._\-^=]{1,32}$")
_DEFAULT_TIMEOUT_SECONDS = 3600
_VALID_EXEC_MODES = {"docker", "local"}
_SCREEN_TIMEOUT_SECONDS = 120
_CONTAINER_STOP_TIMEOUT_SECONDS = 30


def _force_stop_container(container_name: str) -> None:
    """Best-effort cleanup for a timed-out ``docker compose run``.

    ``subprocess.run(..., timeout=...)`` only kills the ``docker compose
    run`` CLI process on timeout — it does NOT stop the container that CLI
    started, since a killed client doesn't get a chance to signal it.
    Without this, a timed-out run's container keeps running (and consuming
    the LLM backend's capacity) indefinitely — observed in practice: a
    timed-out screen run's container was still making LLM calls 2+ hours
    later, starving every subsequent run of a shared local LLM backend.
    """
    try:
        subprocess.run(
            ["docker", "stop", container_name],
            capture_output=True, timeout=_CONTAINER_STOP_TIMEOUT_SECONDS,
        )
    except Exception as exc:  # noqa: BLE001 — cleanup failure shouldn't mask the timeout error
        logger.warning("failed to stop timed-out container %s: %s", container_name, exc)


@lru_cache(maxsize=1)
def _configured_settings() -> dict[str, Any]:
    """Read non-secret plugin settings from the active Hermes config."""
    try:
        from hermes_constants import get_hermes_home
        from hermes_cli.config import read_user_config_raw

        raw = read_user_config_raw(get_hermes_home() / "config.yaml") or {}
        settings = (
            raw.get("plugins", {})
            .get("entries", {})
            .get("tradingagents", {})
            .get("settings", {})
        )
        return settings if isinstance(settings, dict) else {}
    except Exception:
        return {}


def _setting(name: str, default: str = "") -> str:
    """Prefer an explicit process environment override over config.yaml."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        value = _configured_settings().get(name, default)
    return str(value).strip()


def _child_env() -> dict[str, str]:
    """Pass adapter-only source settings from Hermes config to the child."""
    env = os.environ.copy()
    for name in ("TRADINGAGENTS_SOURCE_DIR", "TRADINGAGENTS_SOURCE_PYTHON"):
        value = _setting(name)
        if value:
            env[name] = value
    return env


def _tradingagents_dir() -> Path | None:
    raw = _setting("TRADINGAGENTS_DIR")
    return Path(raw).expanduser() if raw else None


def _exec_mode() -> str:
    mode = _setting("TRADINGAGENTS_EXEC_MODE", "docker").lower()
    return mode if mode in _VALID_EXEC_MODES else "docker"


def _python_bin() -> str:
    return _setting("TRADINGAGENTS_PYTHON", "python3") or "python3"


def _python_resolvable(python_bin: str) -> bool:
    # A path (has a separator, or explicitly relative/home) must exist and
    # be executable; a bare command name is resolved against PATH instead.
    if os.sep in python_bin or python_bin.startswith("~") or python_bin.startswith("."):
        candidate = Path(python_bin).expanduser()
        return candidate.is_file() and os.access(candidate, os.X_OK)
    return shutil.which(python_bin) is not None


def diagnose() -> dict[str, Any]:
    """Report whether the configured TradingAgents target is reachable.

    Returns at least ``{"mode": ..., "directory": ..., "ready": bool,
    "detail": str}``; docker mode also returns ``compose_service``, local
    mode also returns ``python``.
    """
    mode = _exec_mode()
    directory = _tradingagents_dir()
    info: dict[str, Any] = {"mode": mode, "directory": str(directory) if directory else None}

    if directory is None:
        info.update(ready=False, detail="TRADINGAGENTS_DIR is not set.")
        return info
    if not directory.is_dir():
        info.update(ready=False, detail=f"TRADINGAGENTS_DIR does not exist: {directory}")
        return info

    script = directory / "scripts" / "batch_analyze.py"
    if not script.is_file():
        info.update(ready=False, detail=(
            f"scripts/batch_analyze.py not found under {directory}. Copy it from "
            "this plugin's reference/ directory into the TradingAgents checkout."
        ))
        return info

    if mode == "docker":
        if not (directory / "docker-compose.yml").is_file():
            info.update(ready=False, detail=f"No docker-compose.yml found in {directory}.")
            return info
        if shutil.which("docker") is None:
            info.update(ready=False, detail="The 'docker' binary is not on PATH.")
            return info
        service = _setting("TRADINGAGENTS_COMPOSE_SERVICE", "tradingagents") or "tradingagents"
        info.update(ready=True, detail="docker + docker-compose.yml + batch script found.", compose_service=service)
        return info

    # mode == "local"
    python_bin = _python_bin()
    if not _python_resolvable(python_bin):
        info.update(ready=False, detail=(
            f"Python interpreter not found: {python_bin!r}. Set TRADINGAGENTS_PYTHON "
            "to the interpreter TradingAgents is installed in."
        ))
        return info
    info.update(ready=True, detail="local python interpreter + batch script found.", python=python_bin)
    return info


def _check_tradingagents_available() -> bool:
    return bool(diagnose().get("ready"))


def diagnose_screener() -> dict[str, Any]:
    """Report which screener path (stage A of tradingagents_screen) is
    available: the native in-process path (yfinance importable directly in
    Hermes's own environment) or the fallback that shells out to
    scripts/screen_candidates.py inside TRADINGAGENTS_DIR (same reachability
    rules as tradingagents_analyze, via diagnose() above)."""
    native = screener.native_available()
    if native:
        return {"ready": True, "path": "native", "detail": "yfinance importable in Hermes's environment."}

    diag = diagnose()
    directory = _tradingagents_dir()
    fallback_script_ok = bool(
        diag.get("ready") and directory and (directory / "scripts" / "screen_candidates.py").is_file()
    )
    if fallback_script_ok:
        return {
            "ready": True, "path": "fallback",
            "detail": f"yfinance not importable here; falling back to TradingAgents ({diag.get('mode')}).",
        }
    return {
        "ready": False, "path": None,
        "detail": (
            "Neither path available: yfinance isn't importable in Hermes's environment "
            "(pip install yfinance to enable the fast native path), and the fallback needs "
            "scripts/screen_candidates.py in TRADINGAGENTS_DIR plus a reachable TradingAgents "
            f"({diag.get('detail', 'not configured')})."
        ),
    }


def _check_screener_available() -> bool:
    return bool(diagnose_screener().get("ready"))


def _parse_tickers(raw: Any) -> List[str]:
    if raw is None:
        # Dashboard-saved watchlist wins when present; otherwise fall back
        # to the env var (bootstrap path for users who haven't opened the
        # dashboard yet).
        raw = store.load_watchlist()
        if not raw:
            watchlist = _setting("TRADINGAGENTS_WATCHLIST")
            raw = [item.strip() for item in watchlist.split(",") if item.strip()]
    elif isinstance(raw, str):
        raw = [item.strip() for item in raw.split(",") if item.strip()]
    elif isinstance(raw, list):
        raw = [str(item).strip() for item in raw if str(item).strip()]
    else:
        raw = []
    return [t.upper() for t in raw]


class TradingAgentsRunError(RuntimeError):
    """Raised by run_batch() on any failure; carries structured extras
    (e.g. mode, output_tail) that callers can surface however fits them —
    tool_error() kwargs for the LLM tool, an HTTPException detail for the
    dashboard's queued /run endpoint."""

    def __init__(self, message: str, **extra: Any) -> None:
        super().__init__(message)
        self.extra = extra


def run_batch(
    tickers: List[str], trade_date: str | None = None, horizon: str | None = None, quick: bool = False,
) -> dict:
    """Run one TradingAgents batch invocation for `tickers` and persist
    results (report + history) via store.py.

    Shared by the tradingagents_analyze tool handler, tradingagents_screen's
    deep-dive stage, and the dashboard's queued run worker
    (dashboard/plugin_api.py) — one code path for "how does a batch actually
    get run", so the surfaces can't drift.

    ``horizon`` is "swing" or "position"; omitted entirely (batch_analyze.py
    defaults to "position") when not given, so existing tradingagents_analyze
    callers are unaffected.

    ``quick``, when True, passes --quick so every agent (including the
    research manager and portfolio manager, normally the deep-think model)
    uses the quick-think model for this run. Deliberately opt-in per call
    rather than a config change — a deployment where deep_think_llm is a
    much larger/slower model than quick_think_llm can make every ordinary
    ticker analysis painfully slow if the two share one model slot (e.g. a
    single local llama.cpp instance), but tradingagents_analyze / "Run all"
    should still get the deep model's full reasoning. Only the screener's
    shortlist deep-dive (a cheap pre-filter, not a final decision) sets this.

    Raises TradingAgentsRunError on any failure. Returns the trimmed
    summary dict (see _persist_and_summarize) on success.
    """
    diag = diagnose()
    if not diag.get("ready"):
        raise TradingAgentsRunError(
            diag.get("detail") or "TradingAgents is not reachable — check configuration.",
            mode=diag.get("mode"),
        )
    directory = Path(diag["directory"])
    mode = diag["mode"]

    if not tickers:
        raise TradingAgentsRunError(
            "No tickers given, and no watchlist is configured (dashboard panel "
            "or TRADINGAGENTS_WATCHLIST). Pass tickers explicitly, e.g. "
            "[\"AAPL\", \"NVDA\", \"BTC-USD\"]."
        )
    invalid = [t for t in tickers if not _TICKER_RE.match(t)]
    if invalid:
        raise TradingAgentsRunError(f"Invalid ticker symbol(s): {', '.join(invalid)}")

    timeout_seconds = int(_setting("TRADINGAGENTS_TIMEOUT_SECONDS", str(_DEFAULT_TIMEOUT_SECONDS)))
    container_name = f"tradingagents-run-{uuid.uuid4().hex[:12]}"

    if mode == "docker":
        # --entrypoint python is required: the image's ENTRYPOINT is the
        # `tradingagents` CLI itself, so without overriding it, "python
        # scripts/batch_analyze.py ..." would be appended as *arguments* to
        # that entrypoint (`tradingagents python scripts/batch_analyze.py
        # ...`) instead of replacing it — the classic Docker exec-form
        # ENTRYPOINT gotcha.
        # --name gives the container a name we control, so a timeout below
        # can target it for cleanup instead of leaving it running forever.
        cmd = [
            "docker", "compose", "run", "--rm", "-T",
            "--name", container_name,
            "--entrypoint", "python", diag["compose_service"],
            "scripts/batch_analyze.py",
            "--tickers", ",".join(tickers),
        ]
    else:
        cmd = [diag["python"], "scripts/batch_analyze.py", "--tickers", ",".join(tickers)]
    if trade_date:
        cmd += ["--date", trade_date]
    if horizon:
        cmd += ["--horizon", horizon]
    if quick:
        cmd += ["--quick"]

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(directory),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            env=_child_env(),
        )
    except subprocess.TimeoutExpired:
        if mode == "docker":
            _force_stop_container(container_name)
        raise TradingAgentsRunError(
            f"tradingagents run timed out after {timeout_seconds}s for tickers: "
            f"{', '.join(tickers)} (mode={mode})"
        )
    except Exception as exc:  # noqa: BLE001
        raise TradingAgentsRunError(f"Failed to launch tradingagents (mode={mode}): {type(exc).__name__}: {exc}")

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise TradingAgentsRunError(
            f"tradingagents process exited with code {proc.returncode} (mode={mode})", output_tail=tail
        )

    payload = None
    for line in reversed(proc.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            break

    if payload is None:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise TradingAgentsRunError("Could not find JSON output from batch_analyze.py", output_tail=tail)

    return _persist_and_summarize(payload)


def run_screen(
    asset_classes: List[str], risk: str, horizon: str, limit: int = 20, price_range: str = "all",
) -> list[dict]:
    """Stage A: cheap candidate discovery, no TradingAgents deep-dive.

    Tries the native in-process screener first (no subprocess); falls back
    to shelling out to scripts/screen_candidates.py inside TRADINGAGENTS_DIR
    when yfinance isn't importable in Hermes's own environment. Raises
    TradingAgentsRunError on any failure, same as run_batch.
    """
    diag = diagnose_screener()
    if not diag.get("ready"):
        raise TradingAgentsRunError(diag.get("detail") or "Screener is not available.")

    if diag["path"] == "native":
        try:
            return screener.discover(asset_classes, risk, horizon, limit, price_range)
        except Exception as exc:  # noqa: BLE001
            raise TradingAgentsRunError(f"Native screener failed: {type(exc).__name__}: {exc}")

    # Fallback path — mirrors run_batch's subprocess construction.
    ta_diag = diagnose()
    directory = Path(ta_diag["directory"])
    mode = ta_diag["mode"]
    args = [
        "--asset-classes", ",".join(asset_classes),
        "--risk", risk,
        "--horizon", horizon,
        "--limit", str(limit),
        "--price-range", price_range,
    ]
    screen_container_name = f"tradingagents-screen-{uuid.uuid4().hex[:12]}"
    if mode == "docker":
        cmd = [
            "docker", "compose", "run", "--rm", "-T",
            "--name", screen_container_name,
            "--entrypoint", "python", ta_diag["compose_service"],
            "scripts/screen_candidates.py",
        ] + args
    else:
        cmd = [ta_diag["python"], "scripts/screen_candidates.py"] + args

    try:
        proc = subprocess.run(
            cmd, cwd=str(directory), capture_output=True, text=True,
            timeout=_SCREEN_TIMEOUT_SECONDS,
        )
    except subprocess.TimeoutExpired:
        if mode == "docker":
            _force_stop_container(screen_container_name)
        raise TradingAgentsRunError(f"Screener fallback timed out after {_SCREEN_TIMEOUT_SECONDS}s")
    except Exception as exc:  # noqa: BLE001
        raise TradingAgentsRunError(f"Failed to launch screener fallback: {type(exc).__name__}: {exc}")

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or proc.stdout or "").splitlines()[-40:])
        raise TradingAgentsRunError(f"Screener fallback exited with code {proc.returncode}", output_tail=tail)

    payload = None
    for line in reversed(proc.stdout.splitlines()):
        stripped = line.strip()
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError:
                continue
            break
    if payload is None:
        tail = "\n".join(proc.stdout.splitlines()[-40:])
        raise TradingAgentsRunError("Could not find JSON output from screen_candidates.py", output_tail=tail)
    if payload.get("error"):
        raise TradingAgentsRunError(f"Screener fallback error: {payload['error']}")
    return payload.get("candidates", [])


def run_screen_and_analyze(
    asset_classes: List[str], risk: str, horizon: str, limit: int, trade_date: str | None = None,
    on_candidates: "Callable[[list[dict]], None] | None" = None, price_range: str = "all",
    on_result: "Callable[[list[dict]], None] | None" = None,
) -> dict:
    """Stage A (discovery) + stage B (TradingAgents deep dive on the
    shortlist), the full tradingagents_screen pipeline.

    ``on_candidates``, if given, is called once stage A resolves the
    shortlist and before stage B starts — the dashboard's queued worker uses
    it to record which tickers a screen job is about to deep-dive (job
    "tickers" starts empty since discovery hasn't run yet), so those tickers
    can be shown as busy elsewhere in the UI (e.g. the watchlist's per-ticker
    Run buttons) instead of only the screen job itself looking busy.

    Stage B deep-dives tickers **one at a time** (rather than one batched
    ``run_batch(tickers, ...)`` call) specifically so ``on_result``, if
    given, can be called after each ticker finishes with the results
    accumulated so far — the dashboard uses this to show the screener's
    table filling in row by row instead of staying empty until every
    ticker in the shortlist is done. The tradeoff is one docker/local
    invocation per ticker instead of one for the whole shortlist; negligible
    next to the minutes a single ticker's multi-agent run itself takes.
    """
    candidates = run_screen(asset_classes, risk, horizon, limit, price_range)
    tickers = [c["ticker"] for c in candidates if c.get("ticker")]
    by_ticker = {c["ticker"]: c for c in candidates if c.get("ticker")}

    if on_candidates is not None:
        on_candidates(candidates)

    if not tickers:
        return {"candidates": [], "results": []}

    enriched: list[dict] = []
    trade_date_out = trade_date
    for ticker in tickers:
        try:
            analysis = run_batch([ticker], trade_date, horizon, quick=True)
            trade_date_out = analysis.get("date") or trade_date_out
            ticker_results = analysis.get("results") or []
            result = ticker_results[0] if ticker_results else {"ticker": ticker, "error": "no result returned"}
        except TradingAgentsRunError as exc:
            result = {"ticker": ticker, "date": trade_date_out, "error": str(exc)}
        screen_info = by_ticker.get(ticker, {})
        enriched.append({
            **result,
            "screen_source": screen_info.get("source"),
            "screen_metrics": screen_info.get("metrics"),
        })
        if on_result is not None:
            on_result(list(enriched))

    try:
        store.save_screen_run({
            "asset_classes": asset_classes, "risk": risk, "horizon": horizon,
            "price_range": price_range,
            "date": trade_date_out, "results": enriched, "created_at": int(time.time()),
        })
    except Exception:
        pass  # dashboard history is best-effort; must not fail the tool call

    return {"date": trade_date_out, "candidates": candidates, "results": enriched}


def _handle_tradingagents_screen(args: dict, **kw) -> str:
    asset_classes = args.get("asset_classes") or ["stock"]
    if isinstance(asset_classes, str):
        asset_classes = [a.strip() for a in asset_classes.split(",") if a.strip()]
    risk = str(args.get("risk") or "medium").strip().lower()
    horizon = str(args.get("horizon") or "position").strip().lower()
    limit = int(args.get("limit") or 10)
    price_range = str(args.get("price_range") or "all").strip().lower()
    trade_date = str(args.get("date") or "").strip() or None
    try:
        result = run_screen_and_analyze(
            asset_classes, risk, horizon, limit, trade_date, price_range=price_range,
        )
    except TradingAgentsRunError as exc:
        return tool_error(str(exc), **exc.extra)
    except ValueError as exc:
        return tool_error(str(exc))
    return tool_result(result)


def _handle_tradingagents_analyze(args: dict, **kw) -> str:
    tickers = _parse_tickers(args.get("tickers"))
    trade_date = str(args.get("date") or "").strip() or None
    try:
        result = run_batch(tickers, trade_date)
    except TradingAgentsRunError as exc:
        return tool_error(str(exc), **exc.extra)
    return tool_result(result)


def _persist_and_summarize(payload: dict) -> dict:
    """Save each result's full report + a history entry, and return a
    trimmed summary (no report bodies) for the LLM's tool-result context.

    The full report is large (multiple analyst + debate transcripts) and
    already saved to disk for the dashboard's "open full report" link, so
    there's no reason to duplicate it into the agent's context window.
    """
    now = int(time.time())
    summarized = []
    for result in payload.get("results", []):
        ticker = result.get("ticker")
        date = result.get("date")
        entry = {
            "ticker": ticker,
            "date": date,
            "created_at": now,
            "asset_type": result.get("asset_type"),
            "horizon": result.get("horizon"),
            "decision": result.get("decision"),
            "sentiment_band": result.get("sentiment_band"),
            "sentiment_score": result.get("sentiment_score"),
            "price_target": result.get("price_target"),
            "time_horizon": result.get("time_horizon"),
            "decision_summary": result.get("decision_summary"),
            "error": result.get("error"),
        }
        report_text = result.get("report")
        if ticker and date and report_text:
            try:
                store.save_report(ticker, date, report_text)
                entry["has_report"] = True
            except Exception:
                entry["has_report"] = False
        else:
            entry["has_report"] = False
        try:
            store.append_history(entry)
        except Exception:
            pass  # dashboard history is best-effort; must not fail the tool call
        summarized.append({k: v for k, v in entry.items() if k != "created_at"})

    return {"date": payload.get("date"), "results": summarized}


TRADINGAGENTS_SCREEN_SCHEMA = {
    "name": "tradingagents_screen",
    "description": (
        "Discover new trading candidates and analyze them: first runs a cheap "
        "quantitative screen (Yahoo Finance screener for stocks, CoinGecko for "
        "crypto, a static futures list for commodities) filtered by risk level "
        "and trade horizon, then runs the full TradingAgents multi-agent pipeline "
        "on the resulting shortlist so each candidate gets a sentiment, a "
        "buy/sell/hold direction, and the underlying screen metrics. Use this "
        "instead of tradingagents_analyze when the user wants NEW ideas rather "
        "than an update on tickers they already picked."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "asset_classes": {
                "type": "array",
                "items": {"type": "string", "enum": ["stock", "crypto", "commodity"]},
                "description": "Which asset classes to screen. Defaults to [\"stock\"].",
            },
            "risk": {
                "type": "string",
                "enum": ["low", "medium", "high"],
                "description": "Risk appetite for the screen. Defaults to \"medium\".",
            },
            "horizon": {
                "type": "string",
                "enum": ["swing", "position"],
                "description": (
                    "\"swing\" for a short trade (a few days, momentum-driven), "
                    "\"position\" for a hold (multi-month trend). Defaults to \"position\"."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Max candidates per asset class to shortlist for deep-dive analysis. Defaults to 10.",
            },
            "price_range": {
                "type": "string",
                "enum": ["all", "pennies", "5_50", "51_100", "101_300", "301_plus"],
                "description": (
                    "Price filter: \"all\", \"pennies\" (under $5), \"5_50\", \"51_100\", "
                    "\"101_300\", or \"301_plus\". Defaults to \"all\"."
                ),
            },
            "date": {
                "type": "string",
                "description": "Analysis date as YYYY-MM-DD. Defaults to today.",
            },
        },
        "required": [],
    },
}


TRADINGAGENTS_ANALYZE_SCHEMA = {
    "name": "tradingagents_analyze",
    "description": (
        "Run the TradingAgents multi-agent LLM research pipeline for one or more "
        "tickers (via Docker or a local install, per TRADINGAGENTS_EXEC_MODE), and "
        "return each ticker's trade decision. "
        "Use for stocks (e.g. AAPL, 0700.HK), crypto (e.g. BTC-USD), or any other "
        "Yahoo Finance ticker. If no tickers are given, falls back to the "
        "watchlist configured in the tradingagents dashboard panel (or "
        "TRADINGAGENTS_WATCHLIST if no dashboard watchlist is saved yet). "
        "Each result is saved for the dashboard's history view and full-report link."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "tickers": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Ticker symbols to analyze, e.g. [\"AAPL\", \"NVDA\", \"BTC-USD\"].",
            },
            "date": {
                "type": "string",
                "description": "Analysis date as YYYY-MM-DD. Defaults to today.",
            },
            "output_format": {
                "type": "string",
                "enum": ["markdown", "json", "brief"],
                "description": "Requested result format; the Yeoman A2A adapter uses markdown.",
            },
            "length": {
                "type": "string",
                "enum": ["short", "long", "full"],
                "default": "short",
                "description": "Presentation hint; the A2A adapter always returns the stored full report.",
            },
        },
        "required": [],
    },
}
