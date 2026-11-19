#!/usr/bin/env python3
"""End-to-end test of hyperparse_server's HTTP API (server/routes.h). Standard library only.

Starts its own hyperparse_server on a free port (--max-queue 2, --max-pages 500,
--shutdown-grace 3) and checks: the three upload forms, input errors, the NDJSON
stream, queue back-pressure (503 + Retry-After, FIFO order), the model
lifecycle endpoints including idle unload, client-disconnect abort, the page
cap, metrics, and graceful shutdown.

  python3 tests/server_test.py [--server build/hyperparse_server] [--data testdata]
      [--small a.pdf] [--medium b.pdf] [--long c.pdf] [--big d.pdf]

Needs four PDFs (paths relative to --data, or absolute): --small, a few pages;
--medium, ~60 pages; --long, ~100 pages (several seconds: long enough to
interrupt); --big, more than 500 pages (over the page cap).
"""
import argparse
import base64
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid

HP = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
failures = 0


def check(cond, what):
    global failures
    print(("ok   " if cond else "FAIL ") + what)
    failures += 0 if cond else 1


def request(port, method, path, body=None, headers=None, timeout=600):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    c.request(method, path, body=body, headers=headers or {})
    r = c.getresponse()
    data = r.read()
    c.close()
    return r.status, dict((k.lower(), v) for k, v in r.getheaders()), data


def post_pdf(port, path, pdf, query=""):
    return request(port, "POST", path + query, pdf, {"Content-Type": "application/pdf"})


def as_json(data):
    try:
        return json.loads(data)
    except ValueError:
        return {}


def error_code(data):
    return as_json(data).get("error", {}).get("code")


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(binary, port, log):
    p = subprocess.Popen([binary, "--port", str(port), "--host", "127.0.0.1", "--max-queue", "2",
                          "--max-pages", "500", "--shutdown-grace", "3"],
                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    deadline = time.time() + 120
    while time.time() < deadline:
        try:
            if request(port, "GET", "/health", timeout=2)[0] == 200:
                return p
        except OSError:
            time.sleep(0.3)
        if p.poll() is not None:
            break
    raise RuntimeError("hyperparse_server did not come up; see the log")


def multipart(field, filename, data):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"{field}\"; filename=\"{filename}\"\r\n"
            f"Content-Type: application/pdf\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}


def test_uploads(port, small):
    st, h, d = post_pdf(port, "/parse", small)
    j = as_json(d)
    pages = [p["page"] for p in j.get("pages", [])]
    check(st == 200 and pages == list(range(len(pages))) and len(pages) > 0, f"/parse raw body: {len(pages)} pages in order")
    check(all("markdown" in p and "lines" in p for p in j.get("pages", [])), "pages carry lines + markdown")
    check(h.get("content-type") == "application/json" and "x-request-id" in h, "JSON content type + request id")
    body, hdr = multipart("file", "a.pdf", small)
    st, _, d = request(port, "POST", "/parse", body, hdr)
    check(st == 200 and len(as_json(d).get("pages", [])) == len(pages), "multipart upload")
    st, _, d = request(port, "POST", "/parse", json.dumps({"pdf": base64.b64encode(small).decode()}),
                       {"Content-Type": "application/json"})
    check(st == 200 and len(as_json(d).get("pages", [])) == len(pages), "base64 JSON upload")
    st, h, d = post_pdf(port, "/parse", small, "?format=markdown")
    check(st == 200 and d.startswith(b"<!-- page 1 -->") and h.get("content-type", "").startswith("text/markdown"),
          "format=markdown")
    st, _, d = post_pdf(port, "/parse", small, "?ocr=none&formulas=0&markdown=0")
    check(st == 200 and all("markdown" not in p for p in as_json(d).get("pages", [])), "ocr=none, markdown=0")


def test_errors(port, small, big):
    cases = [
        ("not a PDF", lambda: post_pdf(port, "/parse", b"hello world"), 400, "INVALID_PDF"),
        ("PDF header, unreadable", lambda: post_pdf(port, "/parse", b"%PDF-1.7\n" + os.urandom(2000)), 400, "INVALID_PDF"),
        ("empty body", lambda: request(port, "POST", "/parse", b""), 400, "EMPTY_BODY"),
        ("unknown parameter", lambda: post_pdf(port, "/parse", small, "?ocr_mode=all"), 400, "INVALID_PARAMETER"),
        ("format on /parse/stream", lambda: post_pdf(port, "/parse/stream", small, "?format=markdown"), 400, "INVALID_PARAMETER"),
        ("dpi out of range", lambda: post_pdf(port, "/parse", small, "?dpi=5000"), 400, "INVALID_DPI"),
        ("bad ocr mode", lambda: post_pdf(port, "/parse", small, "?ocr=maybe"), 400, "INVALID_PARAMETER"),
        ("multipart without file field", lambda: request(port, "POST", "/parse", *multipart("doc", "a.pdf", small)), 400, "MISSING_FILE"),
        ("bad base64", lambda: request(port, "POST", "/parse", b'{"pdf":"!!"}', {"Content-Type": "application/json"}), 400, "BASE64_DECODE_FAILED"),
        ("over --max-pages 500", lambda: post_pdf(port, "/parse", big), 400, "PDF_TOO_LARGE"),
    ]
    for name, call, status, code in cases:
        st, _, d = call()
        check(st == status and error_code(d) == code, f"{name}: {st} {error_code(d)}")
    st, _, _ = request(port, "GET", "/parse")
    check(st == 405, "GET /parse: 405")


def test_stream(port, medium):
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    c.request("POST", "/parse/stream", medium, {"Content-Type": "application/pdf"})
    r = c.getresponse()
    events = [json.loads(line) for line in r.read().splitlines() if line.strip()]
    c.close()
    kinds = [e["event"] for e in events]
    pages = [e["page"] for e in events if e["event"] == "page"]
    n = events[0].get("pages") if events else None
    check(r.status == 200 and r.getheader("Content-Type") == "application/x-ndjson", "stream: 200 application/x-ndjson")
    check(kinds[:1] == ["meta"] and kinds[-1:] == ["end"], "stream: meta first, end last")
    check(sorted(pages) == list(range(n or 0)) and events[-1].get("failed") == 0, f"stream: each of {n} pages once")


def test_queue(port, medium):
    results = [None] * 6

    def one(i):
        st, h, d = post_pdf(port, "/parse", medium)
        results[i] = (st, h, as_json(d))

    threads = []
    for i in range(6):  # 1 running + 2 waiting (--max-queue 2) -> 3 rejected
        t = threading.Thread(target=one, args=(i,))
        t.start()
        threads.append(t)
        time.sleep(0.2)
    for t in threads:
        t.join()
    ok = [r for r in results if r[0] == 200]
    busy = [r for r in results if r[0] == 503]
    check(len(ok) == 3 and len(busy) == 3, f"queue: {len(ok)} served, {len(busy)} rejected")
    check(all(r[2]["error"]["code"] == "SERVER_BUSY" and r[1].get("retry-after") == "1" for r in busy),
          "rejections: SERVER_BUSY + Retry-After")
    waits = [r[2]["stats"]["queue_ms"] for r in results if r[0] == 200]
    check(waits == sorted(waits) and waits[-1] > 1000, f"served in arrival order (queue_ms {[round(w) for w in waits]})")
    return len(busy)


def models(port, method="GET", path="/models"):
    return as_json(request(port, method, path)[2])


def test_models(port, small):
    check(models(port)["loaded"] is True, "warmed up at start")
    m = models(port, "POST", "/models/unload")
    check(m.get("loaded") is False and m["lifecycle"]["last_unload_ms"] > 0, "POST /models/unload")
    st, _, d = post_pdf(port, "/parse", small)
    check(st == 200 and as_json(d)["stats"]["load_ms"] > 0, "parse while unloaded loads on demand")
    m = models(port, "POST", "/models/warmup")
    check(m.get("loaded") is True and m["lifecycle"]["last_warmup_ms"] > 0, "POST /models/warmup")
    check(request(port, "PUT", "/models/idle_unload?seconds=-1")[0] == 400, "idle_unload rejects -1")
    m = models(port, "PUT", "/models/idle_unload?seconds=1")
    check(m.get("idle_unload_s") == 1, "PUT /models/idle_unload?seconds=1")
    time.sleep(2.0)
    m = models(port)
    check(m["loaded"] is False and m["lifecycle"]["idle_unloads"] == 1, "idle unload after 1 s")
    models(port, "PUT", "/models/idle_unload?seconds=0")
    check(models(port, "POST", "/models/load").get("loaded") is True, "POST /models/load")


def test_disconnect(port, big_ok, small):
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(b"POST /parse/stream HTTP/1.1\r\nHost: x\r\nContent-Type: application/pdf\r\n"
              + f"Content-Length: {len(big_ok)}\r\n\r\n".encode() + big_ok)
    got = b""
    while got.count(b'"event":"page"') < 3:
        got += s.recv(65536)
    s.close()  # client gone mid-document
    t = time.time()
    st, _, _ = post_pdf(port, "/parse", small)
    took = time.time() - t
    check(st == 200 and took < 5, f"disconnect aborts the parse: next request served in {took:.2f}s")


def test_metrics(port, rejected):
    st, h, d = request(port, "GET", "/metrics")
    text = d.decode()
    vals = {line.split(" ")[0]: float(line.split(" ")[1]) for line in text.splitlines() if line and not line.startswith("#")}
    check(st == 200 and h.get("content-type", "").startswith("text/plain"), "/metrics text exposition")
    check(vals.get("hp_rejected_total", 0) >= rejected and vals.get("hp_documents_total", 0) > 0, "metrics: counters")
    check('hp_model_unloads_total{cause="idle"}' in vals and "hp_queue_max_waiting" in vals, "metrics: engine + queue gauges")


def test_shutdown(p, port, medium, big_ok):
    long_result = {}

    def long_parse():
        st, _, d = post_pdf(port, "/parse", big_ok)
        long_result["st"], long_result["code"] = st, error_code(d)

    t = threading.Thread(target=long_parse)
    t.start()
    time.sleep(1.5)
    p.send_signal(signal.SIGTERM)
    time.sleep(0.3)
    check(request(port, "GET", "/health/ready")[0] == 503, "draining: /health/ready 503")
    st, _, d = post_pdf(port, "/parse", medium)
    check(st == 503 and error_code(d) == "SHUTTING_DOWN", "draining: new parse 503 SHUTTING_DOWN")
    t.join()
    check(long_result.get("st") == 503 and long_result.get("code") == "SHUTTING_DOWN",
          "running parse cancelled after the 3 s grace")
    try:
        p.wait(timeout=20)
        exited = True
    except subprocess.TimeoutExpired:
        exited = False
    check(exited and p.returncode == 0, "server exits cleanly")
    leftovers = subprocess.run(["pgrep", "-s", str(p.pid)], capture_output=True, text=True).stdout.split()
    check(not leftovers, "no page workers left behind")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default=os.path.join(HP, "build", "hyperparse_server"))
    ap.add_argument("--data", default=os.path.join(HP, "testdata"))
    ap.add_argument("--small", default="idl_scans/fgbv0051.pdf", help="a few pages")
    ap.add_argument("--medium", default="papers60.pdf", help="~60 pages")
    ap.add_argument("--long", default="mix100.pdf", help="~100 pages, several seconds")
    ap.add_argument("--big", default="idl751.pdf", help="more than 500 pages")
    a = ap.parse_args()
    read = lambda name: open(os.path.join(a.data, name), "rb").read()
    small, medium, big_ok, big = read(a.small), read(a.medium), read(a.long), read(a.big)

    port = free_port()
    log_path = os.path.join(os.environ.get("TMPDIR", "/tmp"), f"hyperparse_server_test_{port}.log")
    with open(log_path, "w") as log:
        p = start_server(a.server, port, log)
        try:
            test_uploads(port, small)
            test_errors(port, small, big)
            test_stream(port, medium)
            rejected = test_queue(port, medium)
            test_models(port, small)
            test_disconnect(port, big_ok, small)
            test_metrics(port, rejected)
            test_shutdown(p, port, medium, big_ok)
        finally:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGKILL)
    print(f"server log: {log_path}")
    print("FAILED (%d)" % failures if failures else "PASS")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
