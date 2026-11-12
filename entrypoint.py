#!/usr/bin/env python3
"""Container entrypoint: serve the Kronos web UI without Flask's debug reloader
(debug=True would spawn a second process and confuse container health checks)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app  # noqa: E402

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "7070"))
    print(f"Kronos Web UI on http://0.0.0.0:{port}  (HF_ENDPOINT={os.environ.get('HF_ENDPOINT')})", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
