"""The League runner (runner/league_runner.py) against a fake League API and a fake llama-server: one in-memory
`http` function plays both, on a fake clock whose sleeps move it. One test talks real HTTP to a local server.

Run from the repository's root: python -m unittest
"""
import base64
import contextlib
import io
import json
import tempfile
import threading
import unittest
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

from runner import league_runner as lr

CONFIG = lr.load_config()
GRANITE = "granite-4.2-8b-q4_k_m"
QWEN = "qwen3.5-4b-q4_k_m"
SPEC = lr.model_spec(CONFIG, GRANITE)
API = "https://api.alphabazaar.ai"
TOKEN_URL = "https://pipelines.actions.example/idtoken?api-version=2.0"
REQUEST_TOKEN = "request-token-never-printed"
PROMPT = "THE LEAGUE PROMPT: licensed headline ACME beats estimates"
REASONING = "TOP has every Big Money source lined up and is not chased."
DECISION = {"ticker": "TOP", "action": "buy", "position_size": 10, "time_horizon": "medium", "current_price": 100.0,
            "price_target": 116.0, "stop_loss": 92.0, "auto_close_days": 30, "reasoning": REASONING}


def at(hour, minute, day=16):
    return datetime(2026, 11, day, hour, minute, tzinfo=lr.EASTERN).timestamp()


def jwt_with_exp(exp):
    def part(data):
        return base64.urlsafe_b64encode(json.dumps(data).encode()).decode().rstrip("=")
    return f"{part({'alg': 'RS256', 'kid': 'k1'})}.{part({'exp': exp, 'aud': API})}.signature"


class Clock:
    def __init__(self, start):
        self.now = start
        self.slept = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.slept.append(seconds)
        self.now += seconds


def a_job(job_id="20261116-ai_granite_42_8b", model=GRANITE, sha256=None, created=None, wait=1800):
    created = at(18, 42) if created is None else created
    return {"id": job_id, "bot": "ai_granite_42_8b", "model": model,
            "gguf": {"repo": SPEC["repo"], "revision": SPEC["revision"], "file": SPEC["file"],
                     "sha256": sha256 or SPEC["sha256"], "size": SPEC["size"]},
            "messages": [{"role": "system", "content": PROMPT}, {"role": "user", "content": lr.USER_TURN}],
            "options": {"max_tokens": 4096}, "created_at": lr.iso(created), "expires_at": lr.iso(created + wait)}


class World:
    """The League's API, GitHub's token endpoint and llama-server, as one fake `http` function."""

    def __init__(self, clock, model=GRANITE):
        self.clock = clock
        self.model = model
        self.scheduled = []  # (appears at, job)
        self.open = {}
        self.closed = []
        self.expected = [{"bot": "ai_granite_42_8b", "model": GRANITE, "at": "18:42", "plays": True}]
        self.replies, self.heartbeats, self.chats, self.tokens, self.calls = [], [], [], [], []
        self.healthy = True
        self.chat_seconds = 240.0
        self.chat_content = json.dumps(DECISION)
        self.chat_finish = "stop"
        self.chat_failure = None
        self.self_test_content = json.dumps(DECISION)
        self.reply_statuses = []  # statuses to answer replies with first (then 200)
        self.jobs_status = 200
        self.heartbeat_verdict = {"accepted": True, "problem": None}

    def issued(self):
        return self.tokens[-1] if self.tokens else None

    def __call__(self, method, url, *, headers=None, body=None, timeout=30.0):
        headers = headers or {}
        self.calls.append((method, url))
        if url.startswith(TOKEN_URL):
            assert headers.get("Authorization") == f"bearer {REQUEST_TOKEN}"
            assert "audience=https%3A%2F%2Fapi.alphabazaar.ai" in url
            token = jwt_with_exp(self.clock() + 300)
            self.tokens.append(token)
            return lr.HttpResult(200, json.dumps({"value": token}).encode())
        if url.startswith(API):
            assert headers.get("Authorization") == f"Bearer {self.issued()}", "a fresh OIDC token on every API call"
            return self.api(method, url[len(API):], body)
        if url.startswith("http://127.0.0.1:8080"):
            return self.llama(method, url[len("http://127.0.0.1:8080"):], body, timeout)
        raise AssertionError(f"unexpected call {method} {url}")

    def api(self, method, path, body):
        if path.startswith("/api/league/runner/jobs?") and method == "GET":
            if self.jobs_status != 200:
                return lr.HttpResult(self.jobs_status, json.dumps({"detail": "wrong repository"}).encode())
            for appears, job in list(self.scheduled):
                if self.clock() >= appears:
                    self.scheduled.remove((appears, job))
                    self.open[job["id"]] = job
            for job_id, job in list(self.open.items()):
                if self.clock() >= lr.parse_time(job["expires_at"]):
                    del self.open[job_id]
                    self.closed.append({"id": job_id, "model": job["model"], "status": "expired"})
            return lr.HttpResult(200, json.dumps({"jobs": list(self.open.values()), "closed": self.closed,
                                                  "expected": self.expected}).encode())
        if path.endswith("/reply") and method == "POST":
            job_id = path.split("/")[-2]
            self.replies.append((job_id, body))
            if self.reply_statuses:
                status = self.reply_statuses.pop(0)
                return lr.HttpResult(status, json.dumps({"detail": f"status {status}"}).encode())
            if job_id not in self.open:
                return lr.HttpResult(409, json.dumps({"detail": "This job already has its answer."}).encode())
            job = self.open.pop(job_id)
            self.closed.append({"id": job_id, "model": job["model"], "status": "answered"})
            return lr.HttpResult(200, json.dumps({"status": "accepted", "id": job_id}).encode())
        if path == "/api/league/runner/heartbeat" and method == "POST":
            self.heartbeats.append(body)
            return lr.HttpResult(200, json.dumps({"models": {self.model: self.heartbeat_verdict}}).encode())
        raise AssertionError(f"unexpected API call {method} {path}")

    def llama(self, method, path, body, timeout):
        if path == "/health":
            return lr.HttpResult(200 if self.healthy else 503, b'{"status": "ok"}')
        if path == "/props":
            return lr.HttpResult(200, json.dumps({"build_info": "b12387-a956a852"}).encode())
        if path == "/v1/chat/completions":
            self.chats.append((body, timeout))
            if self.chat_failure:
                raise lr.HttpFailure(self.chat_failure)
            self_test = body["messages"][0]["content"] == lr.SELF_TEST_PROMPT
            self.clock.now += 40.0 if self_test else self.chat_seconds
            content = self.self_test_content if self_test else self.chat_content
            return lr.HttpResult(200, json.dumps({
                "model": body["model"], "system_fingerprint": "b12387-a956a852",
                "choices": [{"index": 0, "finish_reason": self.chat_finish,
                             "message": {"role": "assistant", "content": content,
                                         "reasoning_content": "Comparing TOP and DRFT on the evidence..."}}],
                "usage": {"prompt_tokens": 3600, "completion_tokens": 410, "total_tokens": 4010},
                "timings": {"prompt_n": 3600, "prompt_ms": 150000.0, "prompt_per_second": 24.0, "predicted_n": 410,
                            "predicted_ms": 90000.0, "predicted_per_second": 4.56}}).encode())
        raise AssertionError(f"unexpected llama-server call {path}")


def runner_for(world, clock, model=GRANITE, heartbeat_only=False, server_log=None):
    lines = []
    tokens = lr.OidcTokens(API, env={"ACTIONS_ID_TOKEN_REQUEST_URL": TOKEN_URL,
                                     "ACTIONS_ID_TOKEN_REQUEST_TOKEN": REQUEST_TOKEN}, http=world, clock=clock)
    runner = lr.Runner(model=model, config=CONFIG, api=lr.LeagueApi(API, tokens, http=world),
                       server=lr.LlamaServer("http://127.0.0.1:8080", http=world), clock=clock, sleep=clock.sleep,
                       out=lines.append, heartbeat_only=heartbeat_only, server_log=server_log,
                       machine="ubuntu 24.04, 4 vCPU, 15.6 GiB",
                       run={"id": "987654321", "attempt": "1", "event": "schedule",
                            "url": "https://github.com/alphabazaar/ai-league-runner/actions/runs/987654321"})
    return runner, lines


class TheEvening(unittest.TestCase):
    def setUp(self):
        self.clock = Clock(at(17, 10))
        self.world = World(self.clock)

    def assert_nothing_secret(self, lines, runner=None):
        text = "\n".join(lines) + json.dumps(runner.record if runner else {})
        for secret in (PROMPT, "ACME", REASONING, "Comparing TOP", REQUEST_TOKEN, *self.world.tokens):
            self.assertNotIn(secret, text)

    def test_it_self_tests_vouches_answers_the_job_exactly_as_written_and_stops(self):
        self.world.scheduled.append((at(18, 42), a_job()))
        runner, lines = runner_for(self.world, self.clock)
        self.assertEqual(runner.run(), 0)
        (self_test, _), (job_request, timeout) = self.world.chats
        self.assertEqual(self_test["messages"][1]["content"], lr.USER_TURN)  # the League's own user turn
        self.assertEqual(self_test["max_tokens"], 4096)
        job = a_job()
        self.assertEqual(job_request, {"model": GRANITE, "messages": job["messages"], "max_tokens": 4096})
        self.assertLessEqual(timeout, 1800)
        (job_id, reply), = self.world.replies
        self.assertEqual(job_id, job["id"])
        self.assertEqual(json.loads(reply["content"]), DECISION)  # the raw reply text, for the League to parse
        self.assertEqual(reply["model"], GRANITE)
        self.assertEqual(reply["finish_reason"], "stop")
        self.assertEqual(reply["runner"]["gguf_sha256"], SPEC["sha256"])
        self.assertEqual(reply["runner"]["llama_cpp"], "b12387-a956a852")
        self.assertEqual(reply["runner"]["timings"]["predicted_n"], 410)
        self.assertEqual(reply["usage"], {"prompt_tokens": 3600, "completion_tokens": 410, "total_tokens": 4010})
        first = self.world.heartbeats[0]
        self.assertEqual(first["models"][0]["model"], GRANITE)
        self.assertEqual(first["models"][0]["gguf_sha256"], SPEC["sha256"])
        self.assertTrue(first["models"][0]["self_test"]["ok"])
        self.assertGreaterEqual(len(self.world.heartbeats), 6)  # one at the start, then every 15 min while waiting
        self.assertIn("Job 20261116-ai_granite_42_8b (granite-4.2-8b-q4_k_m): 3600 prompt tokens at 24.0/s, 410 tokens "
                      "at 4.56/s, 240s, finish stop; decision: BUY TOP 10%, medium horizon", lines)
        self.assertIn("Today's job for granite-4.2-8b-q4_k_m is closed: done", lines)
        record = runner.record
        (entry,) = record["jobs"]
        self.assertEqual((entry["status"], entry["http_status"]), ("answered", 200))
        self.assertEqual(entry["decision"], {"ticker": "TOP", "action": "buy", "position_size": 10,
                                             "time_horizon": "medium"})
        self.assertEqual(record["gguf"]["sha256"], SPEC["sha256"])
        self.assertEqual(record["exit"], 0)
        self.assert_nothing_secret(lines, runner)

    def test_other_models_jobs_and_another_gguf_are_never_answered(self):
        self.world.scheduled += [(at(18, 33), a_job("20261116-ai_qwen_35_4b", model=QWEN)),
                                 (at(18, 42), a_job(sha256="0" * 64))]
        self.world.expected = []  # nothing more to wait for once those are seen
        runner, lines = runner_for(self.world, self.clock)
        self.clock.now = at(18, 43)
        runner.run()
        self.assertEqual([body["messages"][0]["content"] for body, _ in self.world.chats], [lr.SELF_TEST_PROMPT])
        self.assertEqual(self.world.replies, [])
        self.assertEqual([j["status"] for j in runner.record["jobs"]], ["refused"])
        self.assertTrue(any("names another GGUF" in line for line in lines))

    def test_a_job_about_to_close_isnt_started(self):
        self.world.scheduled.append((at(18, 42), a_job(created=at(18, 42), wait=1800)))
        self.clock.now = at(19, 11, ) + 45  # 15 s before it closes
        runner, lines = runner_for(self.world, self.clock)
        runner._poll()
        self.assertEqual(len(self.world.chats), 0)
        self.assertEqual(runner.record["jobs"][0]["status"], "late")

    def test_a_model_that_fails_mid_job_sends_an_error_not_a_second_try(self):
        self.world.scheduled.append((at(18, 42), a_job()))
        runner, lines = runner_for(self.world, self.clock)
        runner.self_test = {"ok": True}
        self.clock.now = at(18, 42)
        self.world.chat_failure = "ConnectionResetError"
        runner._poll()
        (job_id, reply), = self.world.replies
        self.assertEqual(reply["error"], "llama-server didn't answer (ConnectionResetError)")
        self.assertNotIn("content", reply)
        self.assertEqual(len(self.world.chats), 1)  # the job ran once
        self.assertEqual(runner.record["jobs"][0]["status"], "failed")

    def test_a_reply_the_api_couldnt_take_is_posted_again_and_a_closed_job_is_left(self):
        self.world.scheduled.append((at(18, 42), a_job()))
        self.world.reply_statuses = [503, 503]
        runner, lines = runner_for(self.world, self.clock)
        runner.self_test = {"ok": True}
        self.clock.now = at(18, 42)
        runner._poll()
        self.assertEqual(len(self.world.replies), 3)  # the same reply, until the API took it
        self.assertEqual(len({json.dumps(body, sort_keys=True) for _, body in self.world.replies}), 1)
        self.assertEqual(len(self.world.chats), 1)
        self.assertEqual(runner.record["jobs"][0]["status"], "answered")
        world2 = World(Clock(at(18, 42)))
        world2.scheduled.append((at(18, 42), a_job()))
        world2.reply_statuses = [409]
        runner2, lines2 = runner_for(world2, world2.clock)
        runner2.self_test = {"ok": True}
        runner2._poll()
        self.assertEqual(runner2.record["jobs"][0]["status"], "closed")
        self.assertIn("Job 20261116-ai_granite_42_8b: status 409", lines2)

    def test_a_failed_self_test_vouches_for_nothing_and_answers_nothing(self):
        self.world.self_test_content = "TOP looks good."
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "llama-server.log"
            log.write_text("main: loading model\nllama_model_load: loaded 5.3 GB\nmain: server is listening\n")
            runner, lines = runner_for(self.world, self.clock, server_log=log)
            self.assertEqual(runner.run(), 0)
        (heartbeat,) = self.world.heartbeats
        self.assertEqual(heartbeat["models"][0]["self_test"]["ok"], False)
        self.assertEqual(heartbeat["models"][0]["self_test"]["detail"], "no JSON decision in the reply")
        self.assertFalse([url for _, url in self.world.calls if "/jobs" in url])  # never polled
        self.assertIn("  llama_model_load: loaded 5.3 GB", lines)  # before any job: safe to show

    def test_a_model_too_slow_for_the_window_doesnt_vouch_for_itself(self):
        slow = lr.HttpResult(200, json.dumps({"choices": [{"finish_reason": "stop", "message": {
            "content": json.dumps(DECISION)}}], "timings": {"prompt_n": 600, "prompt_per_second": 2.0,
                                                           "predicted_n": 300, "predicted_per_second": 0.6}}).encode())
        server = mock.Mock(chat=mock.Mock(return_value=slow))
        test = lr.run_self_test(server, GRANITE, CONFIG, self.clock)
        self.assertFalse(test["ok"])
        self.assertTrue(test["detail"].startswith("too slow for the League's window: a typical day's prompt would "
                                                  "take about"))
        fast = lr.run_self_test(lr.LlamaServer("http://127.0.0.1:8080", http=self.world), GRANITE, CONFIG,
                                self.clock)
        self.assertTrue(fast["ok"])
        self.assertEqual(fast["detail"], "a JSON decision; a typical day's prompt would take about 6 min")

    def test_a_server_that_never_starts_is_reported(self):
        self.world.healthy = False
        runner, lines = runner_for(self.world, self.clock)
        self.assertEqual(runner.run(), 1)
        self.assertEqual(self.world.heartbeats[0]["models"][0]["self_test"],
                         {"ok": False, "detail": "llama-server didn't start"})

    def test_a_self_test_reply_that_didnt_finish_doesnt_pass(self):
        """A JSON decision cut off at the output limit (or stopped for any other reason) isn't a pass: the League's
        reply would be cut off too."""
        for finish in ("length", "error", None):
            self.world.chat_finish = finish
            test = lr.run_self_test(lr.LlamaServer("http://127.0.0.1:8080", http=self.world), GRANITE, CONFIG,
                                    self.clock)
            self.assertFalse(test["ok"], finish)
            self.assertEqual(test["detail"], f"the reply didn't finish ({finish or 'no reason'})")
        self.world.chat_finish = "Stop <b>"  # not a finish_reason the API accepts
        test = lr.run_self_test(lr.LlamaServer("http://127.0.0.1:8080", http=self.world), GRANITE, CONFIG, self.clock)
        self.assertEqual((test["ok"], test.get("finish_reason")), (False, None))

    def test_llama_servers_log_is_never_shown_once_a_league_job_reached_it(self):
        """Its log could quote a prompt (licensed headlines) once a job ran: from then on it stays on the machine."""
        with tempfile.TemporaryDirectory() as tmp:
            log = Path(tmp) / "llama-server.log"
            log.write_text(f"main: loading model\nslot launch: prompt '{PROMPT}'\n")
            runner, lines = runner_for(self.world, self.clock, server_log=log)
            runner._print_server_log()
            self.assertIn("  main: loading model", lines)  # before any job: safe to show
            lines.clear()
            self.world.scheduled.append((at(18, 42), a_job()))
            runner.self_test = {"ok": True}
            self.clock.now = at(18, 42)
            self.world.healthy = False  # llama-server stops after the job: the run ends without its log
            runner._poll()
            self.assertTrue(runner.real_job_seen)
            runner._print_server_log()
            self.assertFalse([line for line in lines if "llama-server's last lines" in line or PROMPT in line])
            self.assert_nothing_secret(lines, runner)

    def test_a_reply_the_api_refuses_with_422_is_dropped_never_posted_again(self):
        self.world.scheduled.append((at(18, 42), a_job()))
        self.world.reply_statuses = [422]
        runner, lines = runner_for(self.world, self.clock)
        runner.self_test = {"ok": True}
        self.clock.now = at(18, 42)
        runner._poll()
        self.assertEqual(len(self.world.replies), 1)  # refused: the League stores nothing, nothing to retry
        (entry,) = runner.record["jobs"]
        self.assertEqual((entry["status"], entry["http_status"]), ("refused", 422))
        self.assertEqual(runner.pending, {})
        self.assertIn("Job 20261116-ai_granite_42_8b: refused: status 422", lines)

    def test_what_the_public_run_log_shows_has_the_shapes_the_api_accepts(self):
        self.world.scheduled.append((at(18, 42), a_job()))
        runner, lines = runner_for(self.world, self.clock)
        runner.self_test = {"ok": True}
        self.clock.now = at(18, 42)
        self.world.chat_failure = "[Errno 104] Connection reset by peer <html>"
        runner._poll()
        (_, reply), = self.world.replies
        self.assertEqual(reply["error"], "llama-server didn't answer ( Errno 104 Connection reset by peer html )")
        self.assertRegex(reply["error"], r"^[A-Za-z0-9][A-Za-z0-9 .,:;'()/_+=-]{0,159}$")
        self.assertRegex(reply["runner"]["llama_cpp"], r"^b[0-9]{1,6}(-[0-9a-f]{7,40}| \([0-9a-f]{7,40}\))$")
        self.assertEqual(lr.plain("  -- llama-server: \"oops\"\n<b>x</b>  "), "llama-server: oops b x /b")
        self.assertEqual(lr.plain("\u202e\u0000"), "unknown")
        self.assertEqual(len(lr.plain("y" * 500)), lr.PLAIN_CHARS)
        self.assertEqual(lr.build_words("b12387-a956a852", CONFIG), "b12387-a956a852")
        self.assertEqual(lr.build_words("version: 12387 (a956a85) built with cc", CONFIG), "b12387 (a956a85)")
        self.assertEqual(lr.build_words(None, CONFIG), "b12387 (a956a85)")
        self.assertEqual((lr.finish_words("STOP"), lr.finish_words("tool_calls"), lr.finish_words("stop!")),
                         ("stop", "tool_calls", None))
        runner2, _ = runner_for(World(Clock(at(17, 10))), Clock(at(17, 10)))
        self.assertEqual(lr.plain(runner2.machine), runner2.machine)

    def test_the_heartbeat_says_whether_the_league_takes_todays_jobs_and_whether_the_model_is_activated(self):
        runner, lines = runner_for(self.world, self.clock)
        runner.self_test = {"ok": True, "detail": "a JSON decision;\nsecond line", "finish_reason": "STOP"}
        self.world.heartbeat_verdict = {"accepted": False, "activated": True,
                                        "problem": "the model's self-test on the runner didn't pass"}
        self.assertTrue(runner._heartbeat())
        test = self.world.heartbeats[-1]["models"][0]["self_test"]
        self.assertEqual((test["detail"], test["finish_reason"]), ("a JSON decision; second line", "stop"))
        self.assertEqual(lines[-1], "Heartbeat: the League does not take today's jobs from granite-4.2-8b-q4_k_m "
                                    "(the model's self-test on the runner didn't pass); the model is activated")
        self.world.heartbeat_verdict = {"accepted": True, "problem": None}  # an older API: no word on activation
        runner._heartbeat()
        self.assertEqual(lines[-1], "Heartbeat: the League takes today's jobs from granite-4.2-8b-q4_k_m")

    def test_heartbeat_only_answers_nothing(self):
        self.world.scheduled.append((at(17, 10), a_job()))
        runner, lines = runner_for(self.world, self.clock, heartbeat_only=True)
        self.assertEqual(runner.run(), 0)
        self.assertEqual(len(self.world.heartbeats), 1)
        self.assertEqual(self.world.replies, [])
        self.assertIn("Heartbeat: the League takes today's jobs from granite-4.2-8b-q4_k_m", lines)

    def test_it_stops_when_no_bot_plays_the_model_or_at_the_hard_stop(self):
        self.world.expected = [{"bot": "ai_granite_42_8b", "model": GRANITE, "at": "18:42", "plays": False}]
        runner, lines = runner_for(self.world, self.clock)
        self.assertEqual(runner.run(), 0)
        self.assertIn("No League bot plays granite-4.2-8b-q4_k_m today: done", lines)
        clock = Clock(at(18, 0))
        world = World(clock)
        world.expected = [{"bot": "ai_granite_42_8b", "model": GRANITE, "at": "19:05", "plays": True}]
        runner, lines = runner_for(world, clock)
        self.assertEqual(runner.run(), 0)
        self.assertEqual(lines[-1], "Stopped at 19:20 ET (the evening's hard stop)")
        self.assertLessEqual(clock.now, at(19, 20) + lr.POLL_SECONDS)

    def test_a_job_that_never_comes_ends_the_wait(self):
        runner, lines = runner_for(self.world, self.clock)
        self.assertEqual(runner.run(), 0)
        self.assertIn("granite-4.2-8b-q4_k_m's job never came: done", lines)
        self.assertLess(self.clock.now, at(19, 20))

    def test_a_refused_token_ends_the_run_after_a_few_tries(self):
        self.world.jobs_status = 401
        runner, lines = runner_for(self.world, self.clock)
        self.assertEqual(runner.run(), 2)
        self.assertEqual(lines.count("The API refused the runner's token: wrong repository"),
                         lr.API_REFUSALS_BEFORE_EXIT)


class TheToken(unittest.TestCase):
    def test_it_asks_github_for_the_apis_audience_and_keeps_it_until_near_expiry(self):
        clock = Clock(at(17, 0))
        world = World(clock)
        tokens = lr.OidcTokens(API, env={"ACTIONS_ID_TOKEN_REQUEST_URL": TOKEN_URL,
                                         "ACTIONS_ID_TOKEN_REQUEST_TOKEN": REQUEST_TOKEN}, http=world, clock=clock)
        first = tokens.get()
        self.assertEqual(tokens.get(), first)
        self.assertEqual(len(world.tokens), 1)
        clock.now += 250  # within a minute of its exp
        self.assertNotEqual(tokens.get(), first)
        self.assertEqual(lr.token_expiry(first), at(17, 0) + 300)

    def test_without_id_token_permission_there_is_no_token(self):
        with self.assertRaisesRegex(lr.RunnerError, "id-token: write"):
            lr.OidcTokens(API, env={}).get()
        refusing = lr.OidcTokens(API, env={"ACTIONS_ID_TOKEN_REQUEST_URL": TOKEN_URL,
                                           "ACTIONS_ID_TOKEN_REQUEST_TOKEN": REQUEST_TOKEN},
                                 http=lambda *a, **k: lr.HttpResult(403, b'{"message": "no"}'))
        with self.assertRaisesRegex(lr.RunnerError, r"GitHub didn't issue an OIDC token \(403\)"):
            refusing.get()


class ThePlan(unittest.TestCase):
    def test_every_enabled_model_runs_until_the_hard_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = lr.plan(CONFIG, now=at(17, 7), runs_dir=Path(tmp))
            self.assertEqual(json.loads(result["matrix"]), [m["model"] for m in CONFIG["models"]])
            self.assertEqual((result["skip"], result["day"]), ("false", "2026-11-16"))
            late = lr.plan(CONFIG, now=at(19, 25), runs_dir=Path(tmp))
            self.assertEqual((late["skip"], late["reason"]), ("true", "after the evening's hard stop"))
            self.assertEqual(lr.plan(CONFIG, now=at(19, 25), runs_dir=Path(tmp), heartbeat_only=True)["skip"],
                             "false")  # a heartbeat-only run by hand works any time

    def test_named_models_and_models_done_today(self):
        config = json.loads(json.dumps(CONFIG))
        config["models"] = config["models"][:5]  # the first five models' day
        config["models"][0]["enabled"] = False  # gpt-oss-20b dropped without a code change
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp)
            self.assertNotIn("gpt-oss-20b-mxfp4", json.loads(lr.plan(config, now=at(17, 7), runs_dir=runs)["matrix"]))
            named = lr.plan(config, models="gpt-oss-20b-mxfp4, GRANITE-4.2-8B-Q4_K_M", now=at(17, 7), runs_dir=runs)
            self.assertEqual(json.loads(named["matrix"]), ["gpt-oss-20b-mxfp4", GRANITE])
            with self.assertRaisesRegex(lr.RunnerError, "not in models.json: llama-9"):
                lr.plan(config, models="llama-9", now=at(17, 7), runs_dir=runs)
            (runs / "2026-11-16.json").write_text(json.dumps({"date": "2026-11-16", "runs": [{"models": [
                {"model": GRANITE, "jobs": [{"status": "answered"}], "exit": 0},
                {"model": QWEN, "jobs": [{"status": "late"}], "exit": 1},  # failed early: a backup tries it
                {"model": "ministral-3-8b-instruct-2512-q4_k_m", "jobs": [], "exit": 0},  # e.g. its self-test failed
                {"model": "nvidia-nemotron-3.5-bolt-e4b-q8_0", "jobs": [], "exit": 0, "heartbeat_only": True}]}]}))
            later = json.loads(lr.plan(config, now=at(18, 50), runs_dir=runs)["matrix"])
            self.assertEqual(later, [QWEN, "nvidia-nemotron-3.5-bolt-e4b-q8_0"])  # the backup run's share

    def test_spec_prints_the_pinned_files(self):
        lines = []
        self.assertEqual(lr.main(["spec", "--model", GRANITE], out=lines.append), 0)
        outputs = dict(line.split("=", 1) for line in lines)
        self.assertEqual(outputs["url"], f"https://huggingface.co/ibm-granite/granite-4.2-8b-GGUF/resolve/"
                                         f"{SPEC['revision']}/granite-4.2-8b-Q4_K_M.gguf")
        self.assertEqual(outputs["sha256"], "16a9369d0805f80b7377d25d87f937a90c05dc04ad79173a52001e42c9aab311")
        self.assertEqual(outputs["llama_tag"], "b12387")
        self.assertEqual(outputs["llama_sha256"], "9c58ebcd5a704e879cfadf2bbcb1fb1ec8ad4a5bec761fc58b3515864dd5e37a")
        self.assertEqual((outputs["ctx"], outputs["server_args"]), ("16384", ""))

    def test_the_config_is_checked_before_it_reaches_a_shell(self):
        with tempfile.TemporaryDirectory() as tmp:
            for change, words in (({"server_args": ["--temp", "0.1; curl evil"]}, "server_args"),
                                  ({"model": "granite"}, "GGUF file's name"),
                                  ({"sha256": "abc"}, "sha256"),
                                  ({"revision": "main"}, "full 40-hex commit"),  # a branch can move
                                  ({"revision": SPEC["revision"][:7]}, "full 40-hex commit"),
                                  ({"revision": SPEC["revision"].upper()}, "full 40-hex commit"),
                                  ({"repo": "ibm-granite/../x"}, "Hugging Face repository"),
                                  ({"repo": "ibm-granite"}, "Hugging Face repository")):
                config = json.loads(json.dumps(CONFIG))
                config["models"][3].update(change)
                path = Path(tmp) / "models.json"
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(lr.RunnerError, words):
                    lr.load_config(path)
            for change in ({"tag": "latest"}, {"commit": "a956a85"}):
                config = json.loads(json.dumps(CONFIG))
                config["llama_cpp"].update(change)
                path.write_text(json.dumps(config))
                with self.assertRaisesRegex(lr.RunnerError, "release tag"):
                    lr.load_config(path)
        for spec in CONFIG["models"]:  # every pin is a full commit
            self.assertRegex(spec["revision"], r"^[0-9a-f]{40}$")

    def test_every_enabled_model_gets_a_runner_at_once_and_is_in_the_readme(self):
        # GitHub's free plan runs 20 jobs at a time per organization, and every model's job must be up at once (no
        # max-parallel: a job that queued would miss its window), so at most 18, leaving room for another job
        enabled = [m for m in CONFIG["models"] if m.get("enabled")]
        self.assertLessEqual(len(enabled), 18)
        self.assertEqual(len({m["model"] for m in CONFIG["models"]}), len(CONFIG["models"]))
        readme = (lr.ROOT / "README.md").read_text(encoding="utf-8")
        fields = {"model", "enabled", "lab", "name", "quant", "repo", "revision", "file", "sha256", "size",
                  "server_args"}
        for spec in CONFIG["models"]:
            self.assertEqual(set(spec), fields, spec["model"])
            self.assertLess(spec["size"], 13 * 10 ** 9, spec["model"])  # fits a 16 GB runner, or never activates
            self.assertIn(spec["quant"].lower(), spec["file"].lower())
            for words in (spec["file"], spec["sha256"], spec["revision"]):
                self.assertIn(words, readme, spec["model"])

    def test_plan_reads_todays_record_from_the_records_branch(self):
        lines = []
        with tempfile.TemporaryDirectory() as tmp:
            runs = Path(tmp) / "runs"
            runs.mkdir()
            day = lr.to_et(datetime.now().timestamp()).date().isoformat()
            (runs / f"{day}.json").write_text(json.dumps({"date": day, "runs": [{"models": [
                {"model": m["model"], "jobs": [{"status": "answered"}], "exit": 0} for m in CONFIG["models"]]}]}))
            with mock.patch.object(lr, "et_time", return_value=lr.dtime(23, 59)):
                self.assertEqual(lr.main(["plan", "--runs", str(runs)], out=lines.append), 0)
        self.assertIn("skip=true", lines)  # every model's job is done today: nothing to run
        self.assertEqual(lr.plan.__kwdefaults__["runs_dir"], lr.ROOT / "records" / "runs")


class TheRecord(unittest.TestCase):
    def test_a_runs_records_go_into_the_days_file_without_any_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            records, runs = Path(tmp) / "record", Path(tmp) / "runs"
            lr.write_record({"model": GRANITE, "jobs": [{"id": "j1", "status": "answered", "content": PROMPT,
                                                         "decision": {"ticker": "TOP"}}],
                             "messages": [PROMPT]}, records / f"{GRANITE}.json")
            lr.write_record({"model": QWEN, "jobs": []}, records / f"{QWEN}.json")
            run = {"id": "1", "attempt": "1", "url": "https://github.com/x/y/actions/runs/1", "event": "schedule"}
            path = lr.merge_records("2026-11-16", records, runs, run)
            day = json.loads(path.read_text())
            self.assertEqual(day["date"], "2026-11-16")
            (entry,) = day["runs"]
            self.assertEqual([m["model"] for m in entry["models"]], [GRANITE, QWEN])
            self.assertNotIn(PROMPT, path.read_text())
            lr.merge_records("2026-11-16", records, runs, run)  # the same attempt again: replaced, not doubled
            lr.merge_records("2026-11-16", records, runs, dict(run, id="2"))
            self.assertEqual([r["run_id"] for r in json.loads(path.read_text())["runs"]], ["1", "2"])
            self.assertEqual(lr.finished_models("2026-11-16", runs), [GRANITE])

    def test_serve_writes_its_record_and_never_a_traceback(self):
        with tempfile.TemporaryDirectory() as tmp:
            record = Path(tmp) / "r.json"
            err = io.StringIO()
            with mock.patch.object(lr.Runner, "run", side_effect=ValueError(f"bad reply {PROMPT}")), \
                    contextlib.redirect_stderr(err):
                code = lr.main(["serve", "--model", GRANITE, "--record", str(record)], out=lambda line: None)
            self.assertEqual(code, 1)
            self.assertTrue(record.exists())
            self.assertIn("unexpected ValueError", err.getvalue())
            self.assertNotIn(PROMPT, err.getvalue())


class TheReply(unittest.TestCase):
    def test_decisions_are_read_leniently_and_never_with_their_reasoning(self):
        text = "<think>TOP or DRFT? " + "x" * 50 + "</think>\n```json\n" + json.dumps(DECISION) + "\n```"
        self.assertEqual(lr.extract_json_object(text), DECISION)
        self.assertEqual(lr.decision_fields(text), {"ticker": "TOP", "action": "buy", "position_size": 10,
                                                    "time_horizon": "medium"})
        self.assertIsNone(lr.decision_fields("no JSON at all"))
        self.assertEqual(lr.decision_words({"position_size": 0, "time_horizon": None}), "pass")
        self.assertEqual(lr.decision_words(None), "no readable decision")


class RealHttp(unittest.TestCase):
    """The urllib path once, against a local server playing GitHub's token endpoint and the League's API."""

    def test_tokens_and_api_calls_over_http(self):
        seen = []

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _send(self, status, data):
                body = json.dumps(data).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                seen.append((self.path, self.headers.get("Authorization"), self.headers.get("User-Agent")))
                if self.path.startswith("/idtoken"):
                    self._send(200, {"value": jwt_with_exp(9_999_999_999)})
                else:
                    self._send(200, {"jobs": [], "closed": [], "expected": []})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                seen.append((self.path, json.loads(self.rfile.read(length))))
                self._send(409, {"detail": "This job already has its answer."})

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_address[1]}"
            tokens = lr.OidcTokens(API, env={"ACTIONS_ID_TOKEN_REQUEST_URL": f"{base}/idtoken?api-version=2.0",
                                             "ACTIONS_ID_TOKEN_REQUEST_TOKEN": REQUEST_TOKEN})
            api = lr.LeagueApi(base, tokens)
            self.assertEqual(api.jobs(GRANITE).json(), {"jobs": [], "closed": [], "expected": []})
            result = api.reply("20261116-ai_granite_42_8b", {"model": GRANITE, "content": "{}"})
            self.assertEqual((result.status, result.detail()), (409, "This job already has its answer."))
        finally:
            server.shutdown()
            server.server_close()
        token_call, jobs_call, reply_call = seen
        self.assertEqual(token_call[0], "/idtoken?api-version=2.0&audience=https%3A%2F%2Fapi.alphabazaar.ai")
        self.assertEqual(token_call[1], f"bearer {REQUEST_TOKEN}")
        self.assertEqual(jobs_call[0], f"/api/league/runner/jobs?model={GRANITE}")
        self.assertTrue(jobs_call[1].startswith("Bearer ") and jobs_call[2].startswith("AlphaBazaar-LeagueRunner/"))
        self.assertEqual(reply_call, ("/api/league/runner/jobs/20261116-ai_granite_42_8b/reply",
                                      {"model": GRANITE, "content": "{}"}))
        with self.assertRaises(lr.HttpFailure):
            lr.urllib_http("GET", "http://127.0.0.1:9/", timeout=2)  # nothing listening: no answer at all


if __name__ == "__main__":
    unittest.main()
