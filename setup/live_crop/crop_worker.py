"""
Persistent live-crop worker.

Reads newline-delimited JSON requests on stdin, writes one JSON response
line per request on stdout. Keeps decoded WSI pages warm in memory so
per-tile crops stay fast enough for lazy per-request loading.

Request  : {"id": int, "op": "tiles", "pair": int, "level": int}
           {"id": int, "op": "crop",  "pair": int, "level": int,
            "x": int, "y": int, "side": "he"|"ihc", "dx"?: float, "dy"?: float}
           {"id": int, "op": "whole", "pair": int, "level": int, "side": "he"|"ihc"}
Response : {"id": int, "ok": true,  ...}              (op-specific payload)
           {"id": int, "ok": false, "error": str}

dx/dy are tile-pixel displacements (see crop_core); omitted means a plain crop.
"""

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import crop_core


def _handle(req: dict) -> dict:
    op = req.get("op")
    if op == "tiles":
        result = crop_core.tissue_tiles(int(req["pair"]), int(req["level"]))
        return {"ok": True, **result}
    if op == "crop":
        png = crop_core.crop_png(
            int(req["pair"]),
            int(req["level"]),
            int(req["x"]),
            int(req["y"]),
            str(req["side"]),
            float(req.get("dx", 0.0)),
            float(req.get("dy", 0.0)),
        )
        return {"ok": True, "png": base64.b64encode(png).decode("ascii")}
    if op == "whole":
        png = crop_core.whole_png(int(req["pair"]), str(req["side"]), int(req["level"]))
        if png is None:
            return {"ok": False, "error": "no pyramid page for whole-image preview"}
        return {"ok": True, "png": base64.b64encode(png).decode("ascii")}
    return {"ok": False, "error": f"unknown op: {op!r}"}


def main() -> None:
    out = sys.stdout
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as exc:
            out.write(json.dumps({"id": None, "ok": False, "error": f"bad json: {exc}"}) + "\n")
            out.flush()
            continue

        req_id = req.get("id")
        try:
            payload = _handle(req)
        except Exception as exc:  # noqa: BLE001 - report any failure back to caller
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        payload["id"] = req_id
        out.write(json.dumps(payload) + "\n")
        out.flush()


if __name__ == "__main__":
    main()
