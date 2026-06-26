#!/usr/bin/env python3
import argparse
import concurrent.futures
import os
import sys
import time
import urllib.request
from pathlib import Path


def request(url, headers=None, timeout=60):
    return urllib.request.Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            **(headers or {}),
        },
    )


def probe(url):
    req = request(url, {"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        content_range = resp.headers.get("Content-Range")
        if resp.status != 206 or not content_range or "/" not in content_range:
            raise RuntimeError("Server does not support HTTP Range requests")
        return int(content_range.rsplit("/", 1)[1])


def download_part(url, part_path, start, end):
    part_path.parent.mkdir(parents=True, exist_ok=True)
    pos = start + (part_path.stat().st_size if part_path.exists() else 0)
    if pos > end:
        return

    while pos <= end:
        req = request(url, {"Range": f"bytes={pos}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=90) as resp:
                if resp.status not in (200, 206):
                    raise RuntimeError(f"Unexpected HTTP status {resp.status}")
                with part_path.open("ab") as f:
                    while True:
                        chunk = resp.read(1024 * 1024)
                        if not chunk:
                            break
                        f.write(chunk)
                        pos += len(chunk)
        except Exception as exc:
            print(f"retry {part_path.name}: {exc}", file=sys.stderr)
            time.sleep(5)


def combine(parts, out_path, total):
    tmp = out_path.with_suffix(out_path.suffix + ".assembling")
    with tmp.open("wb") as out:
        for part in parts:
            with part.open("rb") as f:
                while True:
                    chunk = f.read(8 * 1024 * 1024)
                    if not chunk:
                        break
                    out.write(chunk)
    if tmp.stat().st_size != total:
        raise RuntimeError(f"Assembled size mismatch: {tmp.stat().st_size} != {total}")
    tmp.replace(out_path)


def main():
    parser = argparse.ArgumentParser(description="Resumable multi-connection HTTP Range downloader")
    parser.add_argument("url")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--part-mb", type=int, default=256)
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = probe(args.url)
    if out_path.exists() and out_path.stat().st_size == total:
        print(f"Already complete: {out_path} ({total} bytes)")
        return

    part_dir = out_path.parent / (out_path.name + ".parts")
    part_size = args.part_mb * 1024 * 1024
    ranges = []
    start = 0
    idx = 0
    while start < total:
        end = min(start + part_size - 1, total - 1)
        ranges.append((idx, start, end))
        start = end + 1
        idx += 1

    parts = [part_dir / f"part-{idx:05d}" for idx, _, _ in ranges]
    print(f"Target: {out_path}", flush=True)
    print(f"Size: {total / 1024 / 1024 / 1024:.2f} GiB, parts: {len(parts)}, workers: {args.workers}", flush=True)

    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_part, args.url, parts[idx], start, end)
            for idx, start, end in ranges
        ]
        while True:
            done = sum(1 for f in futures if f.done())
            downloaded = sum(p.stat().st_size for p in parts if p.exists())
            pct = downloaded * 100 / total
            elapsed = max(time.time() - start_time, 1)
            speed = downloaded / elapsed / 1024 / 1024
            print(f"progress {pct:5.2f}%  {downloaded / 1024 / 1024:.1f} MiB / {total / 1024 / 1024:.1f} MiB  {speed:.2f} MiB/s  done {done}/{len(futures)}", flush=True)
            if done == len(futures):
                break
            time.sleep(15)
        for f in futures:
            f.result()

    downloaded = sum(p.stat().st_size for p in parts if p.exists())
    if downloaded != total:
        raise RuntimeError(f"Downloaded size mismatch across parts: {downloaded} != {total}")

    print("Combining parts...", flush=True)
    combine(parts, out_path, total)
    print(f"Complete: {out_path}", flush=True)


if __name__ == "__main__":
    main()
