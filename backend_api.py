"""Minimal HTTP API exposing the existing GeoTIFF detection pipeline.

Run with:

    ./.venv/bin/python backend_api.py

The server intentionally uses only Python's standard library.  It is a local
development bridge for the React frontend, not a replacement for Streamlit.
"""

import json
import os
import tempfile
from email.parser import BytesParser
from email.policy import default
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from detection.pipeline import DetectionPipelineError, process_geotiff
from drift.pipeline import DriftPipelineError, build_drift_analysis


HOST = os.environ.get("API_HOST", "127.0.0.1")
PORT = int(os.environ.get("API_PORT", "8000"))
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
ALLOWED_ORIGINS = {
    "http://127.0.0.1:5173",
    "http://localhost:5173",
}


def _origin(headers) -> str | None:
    value = headers.get("Origin")
    return value if value in ALLOWED_ORIGINS else None


class ApiHandler(BaseHTTPRequestHandler):
    server_version = "OceanTraceAPI/1.0"

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(status)
        origin = _origin(self.headers)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        origin = _origin(self.headers)
        self.send_response(204)
        if origin:
            self.send_header("Access-Control-Allow-Origin", origin)
            self.send_header("Vary", "Origin")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        if urlparse(self.path).path == "/health":
            self._send_json(200, {"status": "ok", "service": "OceanTrace detection API"})
            return
        self._send_json(404, {"detail": "Endpoint not found."})

    def do_POST(self):  # noqa: N802 - required by BaseHTTPRequestHandler
        route = urlparse(self.path).path
        if route not in {"/api/detect", "/api/drift", "/drift"}:
            self._send_json(404, {"detail": "Endpoint not found."})
            return

        if route in {"/api/drift", "/drift"}:
            self._handle_drift_request()
            return

        self._handle_detection_request()

    def _handle_detection_request(self) -> None:

        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length) if content_length is not None else 0
        except ValueError:
            self._send_json(400, {"detail": "Invalid Content-Length header."})
            return
        if body_length <= 0 or body_length > MAX_UPLOAD_BYTES:
            self._send_json(413, {"detail": "GeoTIFF upload must be between 1 byte and 500 MB."})
            return

        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("multipart/form-data"):
            self._send_json(400, {"detail": "Expected multipart/form-data with a file field."})
            return

        request_body = self.rfile.read(body_length)
        message = BytesParser(policy=default).parsebytes(
            b"MIME-Version: 1.0\r\nContent-Type: "
            + content_type.encode("utf-8")
            + b"\r\n\r\n"
            + request_body
        )
        uploaded_bytes = None
        uploaded_name = "upload.tif"
        for part in message.iter_parts():
            disposition = part.get_content_disposition()
            if disposition != "form-data" or part.get_param("name", header="content-disposition") != "file":
                continue
            uploaded_bytes = part.get_payload(decode=True)
            uploaded_name = part.get_filename() or uploaded_name
            break

        if not uploaded_bytes:
            self._send_json(400, {"detail": "The multipart request is missing a non-empty file field."})
            return
        if not Path(uploaded_name).suffix.lower() in {".tif", ".tiff"}:
            self._send_json(415, {"detail": "Only .tif and .tiff GeoTIFF files are supported."})
            return

        suffix = Path(uploaded_name).suffix.lower() or ".tif"
        temp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary_file:
                temporary_file.write(uploaded_bytes)
                temp_path = temporary_file.name
            self._send_json(200, process_geotiff(temp_path))
        except DetectionPipelineError as exc:
            self._send_json(422, {"detail": str(exc)})
        except Exception:
            # Keep implementation details and stack traces out of the browser.
            self._send_json(500, {"detail": "The GeoTIFF could not be processed by the detection pipeline."})
        finally:
            if temp_path:
                try:
                    os.remove(temp_path)
                except OSError:
                    pass

    def _handle_drift_request(self) -> None:
        content_length = self.headers.get("Content-Length")
        try:
            body_length = int(content_length) if content_length is not None else 0
        except ValueError:
            self._send_json(400, {"detail": "Invalid Content-Length header."})
            return
        if body_length <= 0 or body_length > 1_000_000:
            self._send_json(413, {"detail": "The drift request body is invalid or too large."})
            return
        if "application/json" not in self.headers.get("Content-Type", "").lower():
            self._send_json(400, {"detail": "Expected an application/json drift request."})
            return

        try:
            request = json.loads(self.rfile.read(body_length).decode("utf-8"))
            if not isinstance(request, dict):
                raise ValueError("Request body must be a JSON object.")
            origin = request.get("origin")
            if not isinstance(origin, dict):
                raise ValueError("Request must include an origin object.")
            response = build_drift_analysis(
                origin_x=origin.get("x"),
                origin_y=origin.get("y"),
                crs=request.get("crs"),
                hours=request.get("hours"),
                mode=request.get("mode", "forecast"),
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, TypeError) as exc:
            self._send_json(400, {"detail": str(exc)})
            return
        except DriftPipelineError as exc:
            self._send_json(422, {"detail": str(exc)})
            return
        except Exception:
            self._send_json(500, {"detail": "The drift forecast could not be calculated."})
            return

        self._send_json(200, response)

    def log_message(self, format, *args):
        print(f"[{self.log_date_time_string()}] {format % args}")


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), ApiHandler)
    print(f"OceanTrace detection API listening at http://{HOST}:{PORT}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping OceanTrace detection API.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
