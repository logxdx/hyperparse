#!/usr/bin/env python3
"""Real-workload test of a running hyperparse_server: accuracy and throughput over HTTP.

Uploads every PDF in a directory as its own /parse request from several
concurrent clients (honouring 503 + Retry-After, like a real client), writes
the pages in merged-document order as JSON lines (the hyperparse --out format, so
tools/score_idl.py can score them), and reports throughput and latency. With
--stream, larger documents are streamed through /parse/stream at the same
time, so the queue mixes request kinds as it would in production.

  python3 tests/server_eval.py --url http://127.0.0.1:8080 --dir testdata/idl_scans \
      --out idl_server.jsonl [--clients 4] [--stream testdata/papers60.pdf ...]
"""
import argparse
import glob
import http.client
import json
import os
import statistics
import sys
import threading
import time
import urllib.parse


def post(url, path, pdf, stream=False):
    u = urllib.parse.urlparse(url)
    c = http.client.HTTPConnection(u.hostname, u.port, timeout=3600)
    c.request("POST", path, pdf, {"Content-Type": "application/pdf"})
    r = c.getresponse()
    if stream and r.status == 200:
        events = [json.loads(l) for l in r.read().splitlines() if l.strip()]
        c.close()
        return r.status, r.getheader("Retry-After"), events
    body = r.read()
    c.close()
    return r.status, r.getheader("Retry-After"), body


def with_retry(fn, counters):
    """Retries 503 SERVER_BUSY after Retry-After, as a client of a full server should."""
    while True:
        st, retry_after, body = fn()
        if st != 503:
            return st, body
        counters["busy"] += 1
        time.sleep(float(retry_after or 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--dir", required=True, help="PDFs sent one request each, in sorted order")
    ap.add_argument("--out", required=True, help="merged-order JSON lines")
    ap.add_argument("--clients", type=int, default=4)
    ap.add_argument("--stream", nargs="*", default=[], help="PDFs streamed concurrently via /parse/stream")
    a = ap.parse_args()

    pdfs = sorted(glob.glob(os.path.join(a.dir, "*.pdf")))
    results = [None] * len(pdfs)
    latencies, counters, lock = [], {"busy": 0, "errors": 0}, threading.Lock()
    next_doc = [0]

    def client():
        while True:
            with lock:
                i = next_doc[0]
                next_doc[0] += 1
            if i >= len(pdfs):
                return
            pdf = open(pdfs[i], "rb").read()
            t = time.time()
            st, body = with_retry(lambda: post(a.url, "/parse", pdf), counters)
            with lock:
                latencies.append(time.time() - t)
                if st == 200:
                    results[i] = json.loads(body)
                else:
                    counters["errors"] += 1
                    print(f"{os.path.basename(pdfs[i])}: HTTP {st} {body[:200]!r}", file=sys.stderr)

    streams = {}

    def streamer(path):
        pdf = open(path, "rb").read()
        t = time.time()
        st, events = with_retry(lambda: post(a.url, "/parse/stream", pdf, stream=True), counters)
        end = events[-1] if st == 200 and events else {}
        pages = [e["page"] for e in events if e.get("event") == "page"] if st == 200 else []
        streams[os.path.basename(path)] = {
            "status": st, "seconds": time.time() - t, "pages": len(pages), "distinct": len(set(pages)),
            "declared": events[0].get("pages") if st == 200 else None,
            "end": end.get("event"), "failed": end.get("failed"),
            "formulas": end.get("stats", {}).get("formulas"), "lines": end.get("stats", {}).get("lines")}

    t0 = time.time()
    threads = [threading.Thread(target=client) for _ in range(a.clients)]
    threads += [threading.Thread(target=streamer, args=(p,)) for p in a.stream]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    # Merged order: page offsets follow the sorted file list, as in idl751.pdf.
    offset, n_pages, failed = 0, 0, 0
    with open(a.out, "w") as f:
        for r in results:
            if r is None:
                continue
            for p in r["pages"]:
                if "error" in p:
                    failed += 1
                    continue
                p = dict(p, page=offset + p["page"])
                p.pop("markdown", None)
                f.write(json.dumps(p) + "\n")
            offset += r["stats"]["pages"]
            n_pages += r["stats"]["pages"]
    stream_pages = sum(s["pages"] for s in streams.values())
    lat = sorted(latencies)
    print(f"{len(pdfs)} documents, {n_pages} pages via /parse ({failed} failed pages, {counters['errors']} failed "
          f"requests, {counters['busy']} 503 retries), {stream_pages} pages via /parse/stream")
    print(f"wall {wall:.1f}s -> {(n_pages + stream_pages) / wall:.1f} pages/s overall with {a.clients} clients"
          + (f" + {len(a.stream)} streams" if a.stream else ""))
    if lat:
        print(f"/parse latency per document: p50 {statistics.median(lat):.2f}s, "
              f"p95 {lat[int(0.95 * (len(lat) - 1))]:.2f}s, max {lat[-1]:.2f}s")
    for name, s in streams.items():
        print(f"stream {name}: {json.dumps(s)}")
    return 1 if counters["errors"] or failed else 0


if __name__ == "__main__":
    sys.exit(main())
