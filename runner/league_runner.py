#!/usr/bin/env python3
"""
Alpha Bazaar's AI League runner: the League's open-weight models, run with llama.cpp on GitHub's free CPU runners by
this repository's workflow (.github/workflows/league-runner.yml). Python standard library only.

Every weekday evening, for each model in models.json (the workflow runs one job per model, each on its own runner):
1. `plan`: which models run, and whether this run has anything to do. A scheduled run that starts after the evening's
   hard stop, or finds every model's job already done in today's record (runs/<day>.json), does nothing.
2. `spec`: the model's pinned GGUF (repository, revision, file, sha256) and the pinned llama.cpp build, for the
   workflow's download steps, which check each file's sha256 before anything runs.
3. `serve`: with llama-server running the GGUF, a self-test, then a heartbeat to the League's API naming the GGUF's
   sha256 and the self-test's result: a model's first passing self-test starts its bot, and the League writes the
   bot's job only on a day the runner vouches for the model (else the bot's day is a "No decision"). Then the day's
   jobs: each job's messages and request options go to llama-server exactly as the League wrote them (the same ones
   every other League model gets), and the reply goes back. The first answer counts and nothing is run twice. It stops
   once its model's job for the day is closed, when no League bot plays the model today, or at the hard stop.
4. `record`: the run's record (job ids, models, sha256s, timings and decisions; never a prompt or a reply's text) into
   runs/<day>.json on the repository's `records` branch, which the workflow commits there (never to main): a public
   trail. `plan` reads the day's record from that branch.

No key anywhere: the API checks the workflow's GitHub Actions OIDC token, which this script asks GitHub for
(ACTIONS_ID_TOKEN_REQUEST_URL, with the API as the audience) for each call. The prompts carry licensed headlines and
these logs are public, so nothing here prints a job's messages, the model's reply or reasoning, or a token: only job
ids, models, timings and the decision's fields (ticker, action, size, horizon). What the League's public run logs show
from this runner (the llama.cpp build, an error) is plain text in the shapes the API accepts (plain, build_words).
"""
from __future__ import annotations

import argparse
import base64
import http.client
import json
import os
import platform
import re
import sys
import time
import traceback
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, time as dtime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
ROOT = Path(__file__).resolve().parents[1]
MODELS_FILE = ROOT / "models.json"
USER_AGENT = "AlphaBazaar-LeagueRunner/1.0 (+https://github.com/alphabazaar/ai-league-runner)"
# The League's user turn, the same for every League model (the self-test asks it too)
USER_TURN = "Analyze current market conditions and provide ONE new trading signal recommendation in JSON format."

POLL_SECONDS = 20.0  # how often the jobs are read while waiting
HEARTBEAT_EVERY_SECONDS = 15 * 60  # the runner says it is alive this often while it waits
SERVER_START_SECONDS = 15 * 60  # llama-server has this long to load the model
SELF_TEST_TIMEOUT_SECONDS = 20 * 60
MIN_LEFT_SECONDS = 30  # a job with less than this left before it closes isn't started
# A bot's job appears within seconds of its run time, or up to 15 minutes late when the persona service is busy (its
# misfire grace): one that hasn't appeared 20 minutes after the bot's time won't come today
JOB_LATEST_SECONDS = 20 * 60
MAX_MINUTES = 220  # a run polls at most this long (the job's own limit is 240 minutes)
API_REFUSALS_BEFORE_EXIT = 5
TYPICAL_PROMPT_TOKENS = 4000  # a typical day's League prompt, for the self-test's speed check
TYPICAL_COMPLETION_TOKENS = 1000
WINDOW_SHARE = 0.8  # a model must be able to answer a typical day inside this share of the League's window
DECISION_FIELDS = ("ticker", "action", "position_size", "time_horizon")
RECORD_DROP_KEYS = ("messages", "content", "reasoning", "prompt")  # never in a record, whatever happens

# The self-test's prompt: fictional candidates in the League's shape, so nothing licensed is ever sent here
SELF_TEST_PROMPT = """You are a test trader in the self-test of Alpha Bazaar's AI League runner.
The candidates below are fictional.

TODAY'S CANDIDATES:
- AAA: price $50.00, strategy stop $46.00. Three company insiders bought shares last week; the stock is up 2% since.
- BBB: price $20.00, strategy stop $18.50. Quarterly earnings beat estimates by 12% and the stock has held the gain.

Pick ONE candidate or pass. Respond with ONE JSON object and nothing else, with these keys:
ticker ("AAA" or "BBB"), action ("buy"), current_price (number), price_target (number), stop_loss (number),
position_size (5 to 15, or 0 to pass), time_horizon ("short", "medium" or "long", or null to pass),
auto_close_days (a whole number) and reasoning (one sentence)."""


class RunnerError(Exception):
    """Something this run can't do (no OIDC token, a bad config); its message is safe to print."""


# --- Config ---------------------------------------------------------------------------------------------------------

_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
_FILE = re.compile(r"^[A-Za-z0-9._-]+\.gguf$")
_SERVER_ARG = re.compile(r"^[-A-Za-z0-9_.=:]+$")
_MODEL_ID = re.compile(r"^[a-z0-9._-]{1,39}$")


def load_config(path: Path = MODELS_FILE) -> Dict[str, Any]:
    """models.json, checked: each model's id, file, sha256, repository and revision (a full commit: a branch or a tag
    could move) and server arguments (they reach a shell)."""
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    for spec in config.get("models") or []:
        model = spec.get("model", "")
        if not _MODEL_ID.match(model) or model != spec.get("file", "")[:-len(".gguf")].lower():
            raise RunnerError(f"models.json: {model!r} must be its GGUF file's name in lower case")
        if not _FILE.match(spec.get("file", "")) or not _SHA256.match(spec.get("sha256", "")):
            raise RunnerError(f"models.json: {model} needs a .gguf file name and its sha256")
        if not _REPO.match(str(spec.get("repo", ""))) or not _COMMIT.match(str(spec.get("revision", ""))):
            raise RunnerError(f"models.json: {model} needs its Hugging Face repository and a full 40-hex commit as "
                              f"its revision")
        if not all(isinstance(arg, str) and _SERVER_ARG.match(arg) for arg in spec.get("server_args") or []):
            raise RunnerError(f"models.json: {model}'s server_args may only hold flags and plain values")
    llama = config.get("llama_cpp") or {}
    if not _SHA256.match(llama.get("sha256", "")) or not str(llama.get("url", "")).startswith("https://github.com/"):
        raise RunnerError("models.json: llama_cpp needs its release asset's url and sha256")
    if not _BUILD_TAG.match(str(llama.get("tag", ""))) or not _COMMIT.match(str(llama.get("commit", ""))):
        raise RunnerError("models.json: llama_cpp needs its release tag (b12387) and its full commit")
    return config


# --- What the League's public run logs show from here -------------------------------------------------------------

_PLAIN_CHARS = re.compile(r"[^A-Za-z0-9 .,:;'()/_+=-]+")
_BUILD_TAG = re.compile(r"^b[0-9]{1,6}$")
_BUILD = re.compile(r"^b[0-9]{1,6}(-[0-9a-f]{7,40}| \([0-9a-f]{7,40}\))$")
_FINISH = re.compile(r"^[a-z_]{1,20}$")
PLAIN_CHARS = 160  # the API's limit for an error, a self-test's detail and the machine


def plain(text: Any, limit: int = PLAIN_CHARS) -> str:
    """One plain line the API accepts (its ERROR_PATTERN): letters, digits, spaces and . , : ; ' ( ) / _ + = -,
    starting with a letter or digit, at most `limit` characters; anything else becomes a space."""
    words = " ".join(_PLAIN_CHARS.sub(" ", str(text or "")).split())
    words = re.sub(r"^[^A-Za-z0-9]+", "", words)[:limit].rstrip()
    return words or "unknown"


def build_words(reported: Optional[str], config: Mapping[str, Any]) -> str:
    """The llama.cpp build for the API: llama-server's own build_info ("b12387-a956a852") when it has that shape, else
    the pinned release's tag and short commit ("b12387 (a956a85)")."""
    if isinstance(reported, str) and _BUILD.match(reported.strip()):
        return reported.strip()
    llama = config.get("llama_cpp") or {}
    return f"{llama.get('tag')} ({str(llama.get('commit') or '')[:7]})"


def finish_words(value: Any) -> Optional[str]:
    """A finish_reason the API accepts (lower-case letters and _), else None."""
    text = str(value or "").strip().lower()
    return text if _FINISH.match(text) else None


def model_spec(config: Mapping[str, Any], model: str) -> Dict[str, Any]:
    for spec in config.get("models") or []:
        if spec.get("model") == model:
            return dict(spec)
    raise RunnerError(f"{model} isn't in models.json")


def gguf_url(spec: Mapping[str, Any]) -> str:
    """The GGUF at its pinned revision, downloaded anonymously (every model here is public and ungated)."""
    return f"https://huggingface.co/{spec['repo']}/resolve/{spec['revision']}/{spec['file']}"


# --- Time -------------------------------------------------------------------------------------------------------------

def to_et(epoch: float) -> datetime:
    return datetime.fromtimestamp(epoch, timezone.utc).astimezone(EASTERN)


def iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_time(value: Any) -> Optional[float]:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


def et_time(text: str) -> dtime:
    hour, minute = (int(part) for part in str(text).split(":", 1))
    return dtime(hour, minute)


def at_et(day: date, clock_time: dtime) -> float:
    return datetime.combine(day, clock_time, EASTERN).timestamp()


# --- HTTP -----------------------------------------------------------------------------------------------------------

class HttpFailure(Exception):
    """No HTTP answer at all (refused, reset, DNS, timed out); the message names only the kind."""


class HttpResult:
    def __init__(self, status: int, body: bytes = b"", headers: Optional[Mapping[str, str]] = None):
        self.status = status
        self.body = body or b""
        self.headers = dict(headers or {})

    def json(self) -> Any:
        try:
            return json.loads(self.body.decode("utf-8") or "null")
        except (UnicodeDecodeError, ValueError):
            return None

    def detail(self) -> str:
        """The API's own short reason ({"detail": ...}), for the log."""
        data = self.json()
        detail = data.get("detail") if isinstance(data, dict) else None
        return str(detail)[:200] if isinstance(detail, str) else f"HTTP {self.status}"


Http = Callable[..., HttpResult]


def urllib_http(method: str, url: str, *, headers: Optional[Mapping[str, str]] = None, body: Any = None,
                timeout: float = 30.0) -> HttpResult:
    """One HTTP request with urllib. An error status is a result; no answer at all raises HttpFailure."""
    data = None
    all_headers = {"User-Agent": USER_AGENT, "Accept": "application/json", **(headers or {})}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        all_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, method=method, headers=all_headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return HttpResult(response.status, response.read(), dict(response.headers))
    except urllib.error.HTTPError as e:
        try:
            return HttpResult(e.code, e.read(), dict(e.headers or {}))
        finally:
            e.close()
    except (urllib.error.URLError, http.client.HTTPException, OSError) as e:
        raise HttpFailure(type(e).__name__) from None


# --- GitHub's OIDC token ----------------------------------------------------------------------------------------------

def token_expiry(token: str) -> Optional[float]:
    """A JWT's exp, read without checking it (the API checks the token; this only decides when to ask again)."""
    try:
        payload = token.split(".")[1]
        data = json.loads(base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4)))
        return float(data["exp"])
    except (IndexError, KeyError, TypeError, ValueError):
        return None


class OidcTokens:
    """The workflow's OIDC token for the API's audience, from GitHub's token endpoint (the job needs
    `permissions: id-token: write`). Kept until a minute before it expires. Never printed."""

    def __init__(self, audience: str, env: Optional[Mapping[str, str]] = None, http: Http = urllib_http,
                 clock: Callable[[], float] = time.time):
        env = os.environ if env is None else env
        self.audience = audience
        self.url = env.get("ACTIONS_ID_TOKEN_REQUEST_URL") or ""
        self._request_token = env.get("ACTIONS_ID_TOKEN_REQUEST_TOKEN") or ""
        self.http = http
        self.clock = clock
        self._token: Optional[str] = None
        self._expires = 0.0

    def get(self) -> str:
        if self._token and self._expires - 60 > self.clock():
            return self._token
        if not self.url or not self._request_token:
            raise RunnerError("no OIDC token: the job needs `permissions: id-token: write`")
        joiner = "&" if "?" in self.url else "?"
        url = f"{self.url}{joiner}audience={urllib.parse.quote(self.audience, safe='')}"
        result = self.http("GET", url, headers={"Authorization": f"bearer {self._request_token}"}, timeout=20)
        value = (result.json() or {}).get("value") if result.status == 200 and isinstance(result.json(), dict) else None
        if not isinstance(value, str) or not value:
            raise RunnerError(f"GitHub didn't issue an OIDC token ({result.status})")
        self._token, self._expires = value, token_expiry(value) or self.clock() + 240
        return value


# --- The League's API and llama-server --------------------------------------------------------------------------------

class LeagueApi:
    """GET /api/league/runner/jobs, POST /jobs/{id}/reply and /heartbeat, each with a fresh enough OIDC token."""

    def __init__(self, base_url: str, tokens: OidcTokens, http: Http = urllib_http):
        self.base_url = base_url.rstrip("/")
        self.tokens = tokens
        self.http = http

    def _call(self, method: str, path: str, body: Any = None) -> HttpResult:
        token = self.tokens.get()
        return self.http(method, f"{self.base_url}{path}", headers={"Authorization": f"Bearer {token}"}, body=body,
                         timeout=30)

    def jobs(self, model: str) -> HttpResult:
        return self._call("GET", f"/api/league/runner/jobs?model={urllib.parse.quote(model, safe='')}")

    def reply(self, job_id: str, body: Mapping[str, Any]) -> HttpResult:
        return self._call("POST", f"/api/league/runner/jobs/{urllib.parse.quote(job_id, safe='')}/reply", body)

    def heartbeat(self, body: Mapping[str, Any]) -> HttpResult:
        return self._call("POST", "/api/league/runner/heartbeat", body)


class LlamaServer:
    """llama-server's OpenAI-compatible API on this runner (127.0.0.1 only)."""

    def __init__(self, url: str, http: Http = urllib_http):
        self.url = url.rstrip("/")
        self.http = http

    def healthy(self) -> bool:
        try:
            return self.http("GET", f"{self.url}/health", timeout=10).status == 200
        except HttpFailure:
            return False

    def build(self) -> Optional[str]:
        try:
            data = self.http("GET", f"{self.url}/props", timeout=10).json()
        except HttpFailure:
            return None
        value = data.get("build_info") if isinstance(data, dict) else None
        return str(value)[:100] if isinstance(value, str) and value else None

    def chat(self, request: Mapping[str, Any], timeout: float) -> HttpResult:
        return self.http("POST", f"{self.url}/v1/chat/completions", body=dict(request), timeout=timeout)


# --- Replies ----------------------------------------------------------------------------------------------------------

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_CLOSE = re.compile(r"</think>", re.IGNORECASE)
_THINK_OPEN = re.compile(r"<think>", re.IGNORECASE)
_FENCE = re.compile(r"```[A-Za-z]*")


def extract_json_object(text: Optional[str]) -> Optional[Dict[str, Any]]:
    """The first JSON object in a reply, read as leniently as the League reads it (reasoning blocks, fences and
    chatter around it), or None. The League parses the reply itself; this is only for the log's decision fields."""
    if not isinstance(text, str) or not text.strip():
        return None
    cleaned = _THINK_BLOCK.sub("", text)
    closes = list(_THINK_CLOSE.finditer(cleaned))
    if closes:
        cleaned = cleaned[closes[-1].end():]
    opened = _THINK_OPEN.search(cleaned)
    if opened:
        cleaned = cleaned[:opened.start()]
    cleaned = _FENCE.sub("", cleaned)
    decoder = json.JSONDecoder()
    for brace in re.finditer(r"\{", cleaned):
        try:
            value, _ = decoder.raw_decode(cleaned, brace.start())
        except ValueError:
            continue
        if isinstance(value, dict):
            return value
    return None


def decision_fields(content: Optional[str]) -> Optional[Dict[str, Any]]:
    """The decision's public fields (ticker, action, position_size, time_horizon), never its reasoning."""
    data = extract_json_object(content)
    if data is None:
        return None
    fields = {}
    for key in DECISION_FIELDS:
        value = data.get(key)
        if isinstance(value, bool) or not isinstance(value, (str, int, float, type(None))):
            value = None
        fields[key] = value[:24] if isinstance(value, str) else value
    return fields


def decision_words(fields: Optional[Mapping[str, Any]]) -> str:
    if fields is None:
        return "no readable decision"
    try:
        size = float(str(fields.get("position_size") or 0).rstrip("%"))
    except ValueError:
        size = None
    if not size or fields.get("time_horizon") in (None, "", "null", "none"):
        return "pass"
    return (f"{str(fields.get('action') or 'buy').upper()} {fields.get('ticker') or '?'} {size:g}%, "
            f"{fields.get('time_horizon')} horizon")


def number(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != value:
        return None
    return float(value)


def completion_parts(result: HttpResult) -> Dict[str, Any]:
    """What a llama-server chat completion says: content, finish_reason, usage and timings (numbers only)."""
    data = result.json()
    data = data if isinstance(data, dict) else {}
    choices = data.get("choices") if isinstance(data.get("choices"), list) else []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") if isinstance(choice.get("message"), dict) else {}
    content = message.get("content")
    reasoning = message.get("reasoning_content")
    usage = {k: int(v) for k, v in (data.get("usage") or {}).items()
             if k in ("prompt_tokens", "completion_tokens", "total_tokens") and number(v) is not None}
    timings = {k: round(float(v), 2) for k, v in (data.get("timings") or {}).items()
               if k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                        "predicted_per_second", "cache_n") and number(v) is not None}
    if isinstance(reasoning, str):
        timings["reasoning_chars"] = float(len(reasoning))
    return {"content": content if isinstance(content, str) else None,
            "finish_reason": finish_words(choice.get("finish_reason")), "usage": usage, "timings": timings}


# --- The self-test ----------------------------------------------------------------------------------------------------

def run_self_test(server: LlamaServer, model: str, config: Mapping[str, Any],
                  clock: Callable[[], float] = time.time) -> Dict[str, Any]:
    """One League-shaped decision on fictional candidates, with the League's output budget: it passes when the model
    finishes with a JSON decision and is fast enough to answer a typical day's prompt inside the League's window."""
    request = {"model": model, "messages": [{"role": "system", "content": SELF_TEST_PROMPT},
                                            {"role": "user", "content": USER_TURN}],
               "max_tokens": int(config.get("max_tokens") or 4096)}
    started = clock()
    try:
        result = server.chat(request, timeout=SELF_TEST_TIMEOUT_SECONDS)
    except HttpFailure as e:
        return {"ok": False, "seconds": round(clock() - started, 1),
                "detail": plain(f"llama-server didn't answer ({e})")}
    seconds = round(clock() - started, 1)
    if result.status != 200:
        return {"ok": False, "seconds": seconds, "detail": f"llama-server answered {result.status}"}
    parts = completion_parts(result)
    timings = parts["timings"]
    test = {"seconds": seconds, "finish_reason": parts["finish_reason"],
            "prompt_tokens": int(timings.get("prompt_n") or 0) or None,
            "prompt_per_second": timings.get("prompt_per_second"),
            "completion_tokens": int(timings.get("predicted_n") or 0) or None,
            "completion_per_second": timings.get("predicted_per_second")}
    test = {k: v for k, v in test.items() if v is not None}
    decision = extract_json_object(parts["content"])
    if parts["finish_reason"] != "stop":
        return {**test, "ok": False, "detail": f"the reply didn't finish ({parts['finish_reason'] or 'no reason'})"}
    if decision is None or number(_as_number(decision.get("position_size"))) is None:
        return {**test, "ok": False, "detail": "no JSON decision in the reply"}
    pps, tps = timings.get("prompt_per_second"), timings.get("predicted_per_second")
    if not pps or not tps:
        return {**test, "ok": False, "detail": "llama-server reported no speeds"}
    estimate = TYPICAL_PROMPT_TOKENS / pps + max(TYPICAL_COMPLETION_TOKENS, test.get("completion_tokens", 0)) / tps
    window = float(config.get("window_minutes") or 30) * 60
    words = f"a typical day's prompt would take about {max(1, round(estimate / 60))} min"
    if estimate > window * WINDOW_SHARE:
        return {**test, "ok": False, "detail": f"too slow for the League's window: {words}"}
    return {**test, "ok": True, "detail": f"a JSON decision; {words}"}


def _as_number(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return float(value.strip().rstrip("%"))
        except ValueError:
            return None
    return value


# --- Serving the day's jobs -------------------------------------------------------------------------------------------

def machine_words() -> str:
    """'ubuntu 24.04, 4 vCPU, 15.6 GiB' for the heartbeat and the record."""
    name = platform.system()
    try:
        release = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
        name = f"{release.get('ID', name)} {release.get('VERSION_ID', '').strip(chr(34))}".strip()
    except OSError:
        pass
    memory = ""
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                memory = f", {int(line.split()[1]) / 1024 / 1024:.1f} GiB"
    except (OSError, ValueError, IndexError):
        pass
    return f"{name}, {os.cpu_count()} vCPU{memory}"[:200]


def run_info(env: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    env = os.environ if env is None else env
    run_id = env.get("GITHUB_RUN_ID") or ""
    url = (f"{env.get('GITHUB_SERVER_URL', 'https://github.com')}/{env.get('GITHUB_REPOSITORY', '')}/actions/runs/"
           f"{run_id}") if run_id else None
    return {"id": run_id or None, "attempt": env.get("GITHUB_RUN_ATTEMPT") or None, "url": url,
            "event": env.get("GITHUB_EVENT_NAME") or None}


class Runner:
    """One model's evening: self-test, heartbeat, then its jobs until done (see the module docstring)."""

    def __init__(self, *, model: str, config: Mapping[str, Any], api: LeagueApi, server: LlamaServer,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep,
                 out: Callable[[str], None] = print, heartbeat_only: bool = False,
                 server_log: Optional[Path] = None, machine: Optional[str] = None,
                 run: Optional[Mapping[str, Any]] = None):
        self.model = model
        self.config = config
        self.spec = model_spec(config, model)
        self.api = api
        self.server = server
        self.clock = clock
        self.sleep = sleep
        self.out = out
        self.heartbeat_only = heartbeat_only
        self.server_log = server_log
        self.machine = plain(machine or machine_words())
        self.window = float(config.get("window_minutes") or 30) * 60
        self.started = clock()
        llama = config.get("llama_cpp") or {}
        self.build = build_words(None, config)
        self.self_test: Dict[str, Any] = {"ok": False, "detail": "not run"}
        self.handled: set = set()
        self.pending: Dict[str, Dict[str, Any]] = {}  # replies the API hasn't taken yet: job id -> (body, entry)
        self.real_job_seen = False
        self.record: Dict[str, Any] = {
            "model": model, "heartbeat_only": heartbeat_only,
            "gguf": {k: self.spec.get(k) for k in ("repo", "revision", "file", "sha256", "size")},
            "llama_cpp": {"tag": llama.get("tag"), "commit": llama.get("commit"),
                          "archive_sha256": llama.get("sha256")},
            "run": dict(run or run_info()), "machine": self.machine, "started_at": iso(self.started),
            "self_test": None, "heartbeats": [], "jobs": []}

    # --- the steps --------------------------------------------------------------------------------------------

    def run(self) -> int:
        self.out(f"League runner: {self.model} = {self.spec['repo']}/{self.spec['file']} (sha256 "
                 f"{self.spec['sha256']}), llama.cpp {self.build}, {self.machine}")
        if not self._wait_for_server():
            self.self_test = {"ok": False, "detail": "llama-server didn't start"}
            self.out("llama-server didn't become ready")
            self._print_server_log()
            self._heartbeat()
            return self._finish(1)
        self.build = build_words(self.server.build(), self.config)
        self.record["llama_cpp"]["build"] = self.build
        self.self_test = run_self_test(self.server, self.model, self.config, self.clock)
        self.out(f"Self-test {'passed' if self.self_test['ok'] else 'failed'}: {self.self_test.get('detail')} "
                 f"({self._speed_words(self.self_test)})")
        self._heartbeat()
        if not self.self_test["ok"]:
            self._print_server_log()  # before any job: the self-test's fictional prompt is all it has seen
            return self._finish(0)
        if self.heartbeat_only:
            self.out("Heartbeat only: no jobs this run")
            return self._finish(0)
        return self._finish(self._poll())

    def _wait_for_server(self) -> bool:
        deadline = self.clock() + SERVER_START_SECONDS
        while self.clock() < deadline:
            if self.server.healthy():
                return True
            self.sleep(5)
        return False

    def _print_server_log(self) -> None:
        """llama-server's last lines, only while no League job has reached it (they could then quote a prompt)."""
        if self.real_job_seen or not self.server_log:
            return
        try:
            lines = Path(self.server_log).read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
        except OSError:
            return
        self.out("llama-server's last lines (before any League job):")
        for line in lines:
            self.out(f"  {line[:300]}")

    def _heartbeat(self) -> bool:
        test = {k: v for k, v in self.self_test.items()
                if k in ("ok", "seconds", "prompt_tokens", "prompt_per_second", "completion_tokens",
                         "completion_per_second", "finish_reason", "detail")}
        if "detail" in test:
            test["detail"] = plain(test["detail"])
        if "finish_reason" in test:
            test["finish_reason"] = finish_words(test["finish_reason"])
        body = {"models": [{"model": self.model, "gguf_sha256": self.spec["sha256"],
                            "self_test": {k: v for k, v in test.items() if v is not None}}],
                "llama_cpp": self.build, "machine": self.machine}
        entry = {"at": iso(self.clock()), "self_test_ok": bool(self.self_test.get("ok"))}
        self.record["heartbeats"].append(entry)
        try:
            result = self.api.heartbeat(body)
        except (HttpFailure, RunnerError) as e:
            entry["status"] = None
            self.out(f"Heartbeat not sent: {e}")
            return False
        entry["status"] = result.status
        if result.status != 200:
            self.out(f"Heartbeat refused: {result.detail()}")
            return False
        verdict = ((result.json() or {}).get("models") or {}).get(self.model) or {}
        entry["accepted"], entry["problem"] = bool(verdict.get("accepted")), verdict.get("problem")
        entry["activated"] = verdict.get("activated")
        self.out(f"Heartbeat: the League {'takes' if entry['accepted'] else 'does not take'} today's jobs from "
                 f"{self.model}" + (f" ({entry['problem']})" if entry["problem"] else "")
                 + {True: "; the model is activated", False: "; the model isn't activated"}.get(entry["activated"], ""))
        return True

    def _stop_at(self) -> float:
        today = to_et(self.started).date()
        hard_stop = at_et(today, et_time(self.config.get("hard_stop_et") or "19:20"))
        return min(hard_stop, self.started + MAX_MINUTES * 60)

    def _poll(self) -> int:
        stop = self._stop_at()
        last_heartbeat = self.clock()
        refusals = 0
        self.out(f"Waiting for {self.model}'s job until {to_et(stop):%H:%M} ET")
        while self.clock() < stop:
            if self.clock() - last_heartbeat >= HEARTBEAT_EVERY_SECONDS:
                if not self.server.healthy():
                    self.self_test = {"ok": False, "detail": "llama-server stopped"}
                    self.out("llama-server stopped answering")
                    self._heartbeat()
                    return 1
                self._heartbeat()
                last_heartbeat = self.clock()
            self._post_pending()
            try:
                result = self.api.jobs(self.model)
            except (HttpFailure, RunnerError) as e:
                self.out(f"Jobs not read: {e}")
                self.sleep(POLL_SECONDS)
                continue
            if result.status in (401, 403):
                refusals += 1
                self.out(f"The API refused the runner's token: {result.detail()}")
                if refusals >= API_REFUSALS_BEFORE_EXIT:
                    return 2
                self.sleep(POLL_SECONDS)
                continue
            if result.status != 200 or not isinstance(result.json(), dict):
                self.out(f"Jobs not read: {result.detail()}")
                self.sleep(POLL_SECONDS)
                continue
            refusals = 0
            data = result.json()
            for job in data.get("jobs") or []:
                if isinstance(job, dict) and job.get("model") == self.model and job.get("id") not in self.handled:
                    self._answer(job)
            if self._done(data):
                return 0
            self.sleep(max(1.0, min(POLL_SECONDS, stop - self.clock())))
        self.out(f"Stopped at {to_et(self.clock()):%H:%M} ET (the evening's hard stop)")
        return 0

    def _done(self, data: Mapping[str, Any]) -> bool:
        mine_open = [j for j in data.get("jobs") or [] if isinstance(j, dict) and j.get("model") == self.model]
        if mine_open or self.pending:
            return False
        if [c for c in data.get("closed") or [] if isinstance(c, dict) and c.get("model") == self.model]:
            self.out(f"Today's job for {self.model} is closed: done")
            return True
        bots = [b for b in data.get("expected") or [] if isinstance(b, dict) and b.get("model") == self.model]
        playing = [b for b in bots if b.get("plays") and b.get("at")]
        if not playing:
            self.out(f"No League bot plays {self.model} today: done")
            return True
        today = to_et(self.clock()).date()
        latest = max(at_et(today, et_time(b["at"])) for b in playing)
        if self.clock() > latest + JOB_LATEST_SECONDS:
            self.out(f"{self.model}'s job never came: done")
            return True
        return False

    def _answer(self, job: Mapping[str, Any]) -> None:
        """Send one job to llama-server exactly as the League wrote it and post the reply. Never run twice."""
        job_id = str(job.get("id"))
        self.handled.add(job_id)
        entry: Dict[str, Any] = {"id": job_id, "bot": job.get("bot"), "created_at": job.get("created_at"),
                                 "expires_at": job.get("expires_at"), "status": None}
        self.record["jobs"].append(entry)
        expires = parse_time(job.get("expires_at")) or self.clock()
        gguf_sha256 = str((job.get("gguf") or {}).get("sha256") or "")
        if gguf_sha256 != self.spec["sha256"]:
            entry["status"] = "refused"
            self.out(f"Job {job_id}: it names another GGUF (sha256 {gguf_sha256[:12] or 'none'}); not answered")
            return
        left = expires - self.clock()
        if left < MIN_LEFT_SECONDS:
            entry["status"] = "late"
            self.out(f"Job {job_id}: only {max(0, round(left))}s left before it closes; not started")
            return
        options = {k: v for k, v in (job.get("options") or {}).items() if k not in ("model", "messages")}
        request = {"model": self.model, "messages": job.get("messages") or [], **options}
        self.real_job_seen = True
        started = self.clock()
        entry["started_at"] = iso(started)
        runner = {"llama_cpp": self.build, "gguf_sha256": self.spec["sha256"]}
        try:
            result = self.server.chat(request, timeout=max(5.0, left - 5))
        except HttpFailure as e:
            result, failure = None, plain(f"llama-server didn't answer ({e})")
        else:
            failure = None if result.status == 200 else f"llama-server answered {result.status}"
        seconds = round(self.clock() - started, 1)
        entry["seconds"] = seconds
        if failure:
            body = {"model": self.model, "error": failure, "runner": {**runner, "timings": {"total_seconds": seconds}}}
            entry["error"] = failure
            self.out(f"Job {job_id} ({self.model}): failed after {seconds:.0f}s: {failure}")
        else:
            parts = completion_parts(result)
            timings = {**parts["timings"], "total_seconds": seconds}
            body = {"model": self.model, "content": parts["content"] or "", "finish_reason": parts["finish_reason"],
                    "usage": parts["usage"], "runner": {**runner, "timings": timings}}
            fields = decision_fields(parts["content"])
            entry.update(finish_reason=parts["finish_reason"], usage=parts["usage"], timings=timings,
                         decision=fields)
            self.out(f"Job {job_id} ({self.model}): {self._speed_words(timings, job=True)}, {seconds:.0f}s, finish "
                     f"{parts['finish_reason']}; decision: {decision_words(fields)}")
        self.pending[job_id] = {"body": body, "entry": entry, "expires": expires}
        self._post_pending()

    def _post_pending(self) -> None:
        """Post the replies the API hasn't taken: until it takes or refuses one, or the job's window closes."""
        for job_id, item in list(self.pending.items()):
            entry = item["entry"]
            try:
                result = self.api.reply(job_id, item["body"])
            except (HttpFailure, RunnerError) as e:
                status, words = None, str(e)
            else:
                status, words = result.status, result.detail()
            if status == 200:
                entry["status"] = "failed" if item["body"].get("error") else "answered"
                self.out(f"Job {job_id}: the League took the {'error' if entry['status'] == 'failed' else 'answer'}")
            elif status == 409:
                entry["status"] = "closed"
                self.out(f"Job {job_id}: {words}")
            elif status in (400, 404, 422):
                entry["status"] = "refused"
                self.out(f"Job {job_id}: refused: {words}")
            elif self.clock() >= item["expires"]:
                entry["status"] = "late"
                self.out(f"Job {job_id}: the reply couldn't be posted before the job closed ({words})")
            else:
                self.out(f"Job {job_id}: reply not posted yet ({words}); trying again")
                continue
            entry["http_status"] = status
            del self.pending[job_id]

    @staticmethod
    def _speed_words(values: Mapping[str, Any], job: bool = False) -> str:
        if job:
            return (f"{int(values.get('prompt_n') or 0)} prompt tokens at {values.get('prompt_per_second', '?')}/s, "
                    f"{int(values.get('predicted_n') or 0)} tokens at {values.get('predicted_per_second', '?')}/s")
        return (f"{values.get('seconds', '?')}s, prompt {values.get('prompt_per_second', '?')} tokens/s, "
                f"{values.get('completion_tokens', '?')} tokens at {values.get('completion_per_second', '?')}/s")

    def _finish(self, code: int) -> int:
        self.record.update(self_test=self.self_test, ended_at=iso(self.clock()), exit=code)
        return code


# --- The day's record -------------------------------------------------------------------------------------------------

def scrub(value: Any) -> Any:
    """A record without anything that could hold a prompt or a reply's text (RECORD_DROP_KEYS), at any depth."""
    if isinstance(value, dict):
        return {k: scrub(v) for k, v in value.items() if k not in RECORD_DROP_KEYS}
    if isinstance(value, list):
        return [scrub(v) for v in value]
    return value


def write_record(record: Mapping[str, Any], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(scrub(dict(record)), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def merge_records(day: str, records_dir: Path, runs_dir: Path, run: Mapping[str, Any]) -> Path:
    """Add this run's model records to runs/<day>.json: one entry per run (a re-run of the same attempt replaces its
    own), models in id order."""
    date.fromisoformat(day)
    path = Path(runs_dir) / f"{day}.json"
    existing = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {"date": day, "runs": []}
    models = []
    for file in sorted(Path(records_dir).glob("*.json")) if Path(records_dir).exists() else []:
        try:
            models.append(scrub(json.loads(file.read_text(encoding="utf-8"))))
        except ValueError:
            continue
    entry = {"run_id": run.get("id"), "attempt": run.get("attempt"), "url": run.get("url"), "event": run.get("event"),
             "models": sorted(models, key=lambda m: str(m.get("model")))}
    runs = [r for r in existing.get("runs") or []
            if (r.get("run_id"), r.get("attempt")) != (entry["run_id"], entry["attempt"])]
    existing["runs"] = runs + [entry]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def finished_models(day: str, runs_dir: Path) -> List[str]:
    """Models a run already finished today in the day's record: its job taken, closed or refused, or a serve that
    ended normally (its job done, no bot playing it, a job that never came, a self-test that failed: none of which a
    second run would change). A heartbeat-only run finishes nothing; a run that failed early (llama-server didn't
    start, the API refused the token) leaves the model to a backup run."""
    path = Path(runs_dir) / f"{day}.json"
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except ValueError:
        return []
    done = set()
    for run in data.get("runs") or []:
        for record in run.get("models") or []:
            jobs_over = any(job.get("status") in ("answered", "failed", "closed", "refused")
                            for job in record.get("jobs") or [])
            if jobs_over or (record.get("exit") == 0 and not record.get("heartbeat_only")):
                done.add(record.get("model"))
    return sorted(m for m in done if m)


# --- The plan ---------------------------------------------------------------------------------------------------------

def plan(config: Mapping[str, Any], *, models: str = "", heartbeat_only: bool = False, now: Optional[float] = None,
         runs_dir: Path = ROOT / "records" / "runs") -> Dict[str, Any]:
    """Which models this run serves: every enabled one (or the ones named, enabled or not), less the ones whose job
    is done today, and none after the evening's hard stop. A heartbeat-only run serves them whatever the time."""
    now = time.time() if now is None else now
    local = to_et(now)
    day = local.date().isoformat()
    known = [m["model"] for m in config.get("models") or []]
    if models.strip():
        wanted = [m.strip().lower() for m in models.split(",") if m.strip()]
        unknown = [m for m in wanted if m not in known]
        if unknown:
            raise RunnerError(f"not in models.json: {', '.join(unknown)}")
        chosen = list(dict.fromkeys(wanted))
    else:
        chosen = [m["model"] for m in config.get("models") or [] if m.get("enabled")]
    reason = ""
    if not heartbeat_only:
        if local.time() >= et_time(config.get("hard_stop_et") or "19:20"):
            chosen, reason = [], "after the evening's hard stop"
        else:
            done = finished_models(day, runs_dir)
            chosen = [m for m in chosen if m not in done]
            reason = "every model's job is done today" if not chosen else ""
    if not chosen and not reason:
        reason = "no model enabled"
    return {"matrix": json.dumps(chosen), "skip": "true" if not chosen else "false", "day": day, "reason": reason}


# --- The command line -------------------------------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None, out: Callable[[str], None] = print) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p_plan = sub.add_parser("plan", help="which models run (prints GitHub step outputs)")
    p_plan.add_argument("--models", default="")
    p_plan.add_argument("--heartbeat-only", action="store_true")
    p_plan.add_argument("--runs", default=str(ROOT / "records" / "runs"), help="the records branch's runs/")
    p_spec = sub.add_parser("spec", help="a model's pinned files (prints GitHub step outputs)")
    p_spec.add_argument("--model", required=True)
    p_serve = sub.add_parser("serve", help="self-test, heartbeat, then answer the day's jobs")
    p_serve.add_argument("--model", required=True)
    p_serve.add_argument("--server", default="http://127.0.0.1:8080")
    p_serve.add_argument("--server-log")
    p_serve.add_argument("--record", required=True)
    p_serve.add_argument("--heartbeat-only", action="store_true")
    p_record = sub.add_parser("record", help="add this run's records to runs/<day>.json (the records branch's)")
    p_record.add_argument("--day", required=True)
    p_record.add_argument("--records", required=True)
    p_record.add_argument("--runs", default=str(ROOT / "records" / "runs"))
    args = parser.parse_args(argv)
    try:
        config = load_config()
        if args.command == "plan":
            result = plan(config, models=args.models, heartbeat_only=args.heartbeat_only, runs_dir=Path(args.runs))
            for key in ("matrix", "skip", "day"):
                out(f"{key}={result[key]}")
            print(f"Plan for {result['day']}: {result['matrix']}" + (f" (skip: {result['reason']})"
                                                                     if result["skip"] == "true" else ""),
                  file=sys.stderr)
            return 0
        if args.command == "spec":
            spec, llama = model_spec(config, args.model), config["llama_cpp"]
            for key, value in (("file", spec["file"]), ("url", gguf_url(spec)), ("sha256", spec["sha256"]),
                               ("size", spec["size"]), ("llama_url", llama["url"]), ("llama_sha256", llama["sha256"]),
                               ("llama_tag", llama["tag"]), ("ctx", config.get("context_size") or 16384),
                               ("server_args", " ".join(spec.get("server_args") or []))):
                out(f"{key}={value}")
            return 0
        if args.command == "serve":
            tokens = OidcTokens(config.get("audience") or "https://api.alphabazaar.ai")
            runner = Runner(model=args.model, config=config, api=LeagueApi(config.get("api"), tokens),
                            server=LlamaServer(args.server), heartbeat_only=args.heartbeat_only,
                            server_log=Path(args.server_log) if args.server_log else None,
                            out=lambda line: print(line, flush=True))
            try:
                code = runner.run()
            except Exception as e:  # never a traceback: its message could quote what a job or a reply held
                frame = traceback.extract_tb(e.__traceback__)[-1]
                print(f"League runner: unexpected {type(e).__name__} in {frame.name} (line {frame.lineno})",
                      file=sys.stderr)
                code = 1
            finally:
                write_record(runner.record, Path(args.record))
            return code
        if args.command == "record":
            path = merge_records(args.day, Path(args.records), Path(args.runs), run_info())
            out(f"Record: {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}")
            return 0
    except RunnerError as e:
        print(f"League runner: {e}", file=sys.stderr)
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
