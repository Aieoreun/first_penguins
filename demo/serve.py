"""시연용 중계 서버. 프로토타입 화면을 내주고, 화면의 요청을 대회 서버(/tutor, /guard)로 넘긴다.

실행 (대회 서버를 먼저 띄운 뒤, .env를 읽어 들인 셸에서):
    python demo/serve.py

화면은 질문을 하나씩 따로 보낸다(앞 대화를 붙이지 않는다). 대회 서버와 프로토타입 파일은 고치지 않는다. 화면에는 bridge.js 한 줄만 끼워 넣어 내준다.
토큰은 이 서버만 알고 화면에는 내려가지 않는다. 요청과 답변의 본문은 로그에 남기지 않는다.
"""
import csv
import json
import os
import re
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
BACKEND = os.environ.get("DEMO_BACKEND", "http://127.0.0.1:8000")
KB_DIR = Path(os.environ.get("KB_DIR", HERE.parent / "kb"))
PORT = int(os.environ.get("DEMO_PORT", "8765"))
TOKEN = os.environ["TEAM_TOKEN"]
SECTION_ID = re.compile(r"^[A-Z]+-\d+-\d+$")
TAG = '<script src="/bridge.js"></script>'


def page() -> bytes:
    html = (HERE / "prototype.html").read_text(encoding="utf-8")
    cut = html.rfind("</body>")
    return (html[:cut] + TAG + html[cut:]).encode("utf-8")


def _title(path: Path) -> str:
    with path.open(encoding="utf-8") as f:
        return f.readline().lstrip("# ").replace(path.stem, "", 1).strip()


def toc():
    """과목별 섹션 목록. 교재 패널의 목차로 쓴다."""
    subjects = []
    for folder in sorted(p for p in KB_DIR.iterdir() if p.is_dir()):
        sections = [{"id": f.stem, "title": _title(f)} for f in sorted(folder.glob("*.md"))]
        subjects.append({"subject": folder.name, "sections": sections})
    return subjects


def section(section_id: str):
    if not SECTION_ID.match(section_id):
        return None
    found = list(KB_DIR.glob(f"*/{section_id}.md"))
    if not found:
        return None
    text = found[0].read_text(encoding="utf-8")
    tables = {}
    for table in sorted(found[0].parent.glob(f"{section_id}-table*.csv")):
        with table.open(encoding="utf-8-sig", newline="") as f:
            tables[table.name] = list(csv.reader(f))
    return {"id": section_id, "subject": found[0].parent.name, "title": _title(found[0]),
            "text": text.partition("\n")[2].strip(), "tables": tables}


class Handler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, kind: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, status: int, data) -> None:
        self._send(status, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):
        if self.path in ("/", "/index.html", "/prototype.html"):
            self._send(200, page(), "text/html; charset=utf-8")
        elif self.path == "/bridge.js":
            self._send(200, (HERE / "bridge.js").read_bytes(), "text/javascript; charset=utf-8")
        elif self.path == "/api/toc":
            self._json(200, toc())
        elif self.path.startswith("/api/section/"):
            data = section(self.path.rsplit("/", 1)[1])
            self._json(200 if data else 404, data or {"error": "not found"})
        else:
            self._json(404, {"error": "not found"})

    def do_POST(self):
        if self.path not in ("/api/tutor", "/api/guard"):
            self._json(404, {"error": "not found"})
            return
        body = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        request = urllib.request.Request(
            BACKEND + self.path[len("/api"):], data=body, method="POST",
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {TOKEN}"})
        try:
            with urllib.request.urlopen(request, timeout=35) as response:
                self._send(response.status, response.read(), "application/json; charset=utf-8")
        except urllib.error.HTTPError as e:
            self._json(e.code, {"error": "backend error"})
        except Exception as e:
            self._json(502, {"error": type(e).__name__})

    def log_message(self, fmt, *args):
        print(self.command, self.path.split("?")[0], args[1] if len(args) > 1 else "", flush=True)


if __name__ == "__main__":
    print(f"시연 화면: http://127.0.0.1:{PORT}  (대회 서버: {BACKEND})", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
