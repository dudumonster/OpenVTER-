#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Dependency-light local backend for the OpenVTER trajectory visualizer."""
from __future__ import annotations

import argparse
import csv
import json
import mimetypes
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from converter import DEFAULT_ADJUSTED_ROOT, DEFAULT_INITIAL_ROOT, convert_all


APP_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = APP_ROOT / "static"
VIS_ROOT = APP_ROOT.parent


def _safe_dataset_path(dataset_id: str, version: str) -> Path:
    if (
        not dataset_id
        or not version
        or "/" in dataset_id
        or "\\" in dataset_id
        or ".." in dataset_id
        or "/" in version
        or "\\" in version
        or ".." in version
    ):
        raise ValueError("Invalid dataset id.")
    path = (DEFAULT_ADJUSTED_ROOT / dataset_id / version).resolve()
    if DEFAULT_ADJUSTED_ROOT.resolve() not in path.parents and path != DEFAULT_ADJUSTED_ROOT.resolve():
        raise ValueError("Dataset path escapes Adjusted results.")
    return path


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _csv_records(path: Path):
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


class VisualizerHandler(BaseHTTPRequestHandler):
    server_version = "OpenVTERVisualizer/1.0"

    def log_message(self, fmt, *args):  # noqa: D401 - keep BaseHTTPRequestHandler signature.
        print("%s - - %s" % (self.client_address[0], fmt % args))

    def _send_bytes(self, data: bytes, status=HTTPStatus.OK, content_type="application/octet-stream") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload, status=HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(data, status, "application/json; charset=utf-8")

    def _send_error(self, status, message: str) -> None:
        self._send_json({"error": message}, status)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "File not found.")
            return
        content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self._send_bytes(path.read_bytes(), HTTPStatus.OK, content_type)

    def do_GET(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        try:
            if path == "/" or path == "/index.html":
                self._serve_file(STATIC_ROOT / "index.html")
                return
            if path.startswith("/static/"):
                rel = path[len("/static/") :]
                file_path = (STATIC_ROOT / rel).resolve()
                if STATIC_ROOT.resolve() not in file_path.parents:
                    self._send_error(HTTPStatus.BAD_REQUEST, "Invalid static path.")
                    return
                self._serve_file(file_path)
                return
            if path == "/api/datasets":
                self._send_json(self._datasets())
                return
            if path.startswith("/api/datasets/"):
                self._dataset_endpoint(path)
                return
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler.
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path != "/api/scan":
            self._send_error(HTTPStatus.NOT_FOUND, "Unknown endpoint.")
            return
        try:
            query = parse_qs(parsed.query)
            force = query.get("force", ["false"])[0].lower() in {"1", "true", "yes"}
            result = convert_all(DEFAULT_INITIAL_ROOT, DEFAULT_ADJUSTED_ROOT, force=force)
            self._send_json(result)
        except Exception as exc:
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def _datasets(self):
        DEFAULT_INITIAL_ROOT.mkdir(parents=True, exist_ok=True)
        DEFAULT_ADJUSTED_ROOT.mkdir(parents=True, exist_ok=True)
        converted = []
        for dataset_dir in sorted(DEFAULT_ADJUSTED_ROOT.iterdir()):
            if not dataset_dir.is_dir():
                continue
            for version_dir in sorted(dataset_dir.iterdir()):
                if not version_dir.is_dir():
                    continue
                metadata_path = version_dir / "metadata.json"
                if not metadata_path.exists():
                    continue
                metadata = _read_json(metadata_path)
                converted.append(
                    {
                        "dataset_id": dataset_dir.name,
                        "version": version_dir.name,
                        "display_name": metadata.get("display_name", f"{dataset_dir.name} / {version_dir.name}"),
                        "row_count": metadata.get("row_count"),
                        "object_count": metadata.get("object_count"),
                        "full_object_count": metadata.get("full_object_count"),
                        "filtered_object_count": metadata.get("filtered_object_count"),
                        "total_frames": metadata.get("total_frames"),
                        "fps": metadata.get("fps"),
                        "class_names": metadata.get("class_names", []),
                        "converted_time": metadata.get("converted_time"),
                        "warnings": metadata.get("warnings", []),
                    }
                )

        initial = []
        for source_dir in sorted(DEFAULT_INITIAL_ROOT.iterdir()):
            if source_dir.is_dir():
                initial.append(
                    {
                        "dataset_id": source_dir.name,
                        "has_pkl": any(source_dir.glob("*.pkl")),
                        "converted": (
                            (DEFAULT_ADJUSTED_ROOT / source_dir.name / "full" / "metadata.json").exists()
                            and (DEFAULT_ADJUSTED_ROOT / source_dir.name / "moving_filtered" / "metadata.json").exists()
                        ),
                    }
                )
        return {"converted": converted, "initial": initial}

    def _dataset_endpoint(self, path: str) -> None:
        parts = path.strip("/").split("/")
        if len(parts) < 4:
            self._send_error(HTTPStatus.BAD_REQUEST, "Dataset endpoint missing dataset id.")
            return
        dataset_id = parts[2]
        version = parts[3]
        action = parts[4] if len(parts) > 4 else "metadata"
        dataset_dir = _safe_dataset_path(dataset_id, version)
        if not dataset_dir.exists():
            self._send_error(HTTPStatus.NOT_FOUND, f"Dataset {dataset_id} not found.")
            return

        if action == "metadata":
            self._send_json(_read_json(dataset_dir / "metadata.json"))
            return
        if action in {"tracks", "objects", "frames"}:
            csv_path = dataset_dir / f"{action}.csv"
            self._send_json(_csv_records(csv_path))
            return
        if action == "background":
            metadata = _read_json(dataset_dir / "metadata.json")
            image_name = metadata.get("background_image")
            if not image_name:
                self._send_error(HTTPStatus.NOT_FOUND, "Dataset has no background image.")
                return
            self._serve_file(dataset_dir / image_name)
            return
        self._send_error(HTTPStatus.NOT_FOUND, "Unknown dataset action.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the local OpenVTER trajectory visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), VisualizerHandler)
    url = f"http://{args.host}:{args.port}"
    print(f"OpenVTER trajectory visualizer running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
