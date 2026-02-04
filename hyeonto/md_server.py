#!/usr/bin/env python3
"""
Hyeonto 대시보드 & 마크다운 로컬 서버

마크다운 렌더링과 대시보드 HTML을 로컬에서 서빙합니다.
"""

import argparse
import http.server
import os
import socketserver
import sys
import webbrowser
from pathlib import Path


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    """경로 처리 개선된 HTTP 요청 핸들러"""

    def do_GET(self):
        """GET 요청 처리"""
        # 경로 정규화
        if self.path == "/":
            self.path = "/dashboard.html"
        elif self.path.endswith("/"):
            self.path = self.path.rstrip("/") + "/dashboard.html"

        # 마크다운 파일 직접 접근 허용
        file_path = self.path.lstrip("/")

        # 파일이 존재하지 않으면 404 반환
        if not Path(file_path).exists() and file_path not in ["dashboard.html", "md_viewer.html"]:
            self.send_response(404)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(
                f"""<html><head><meta charset="utf-8"><title>404 Not Found</title></head>
                <body style="font-family:sans-serif; padding:40px;">
                <h1>404 - 파일을 찾을 수 없습니다</h1>
                <p>요청한 경로: <code>{file_path}</code></p>
                <p><a href="/">대시보드로 돌아가기</a></p>
                </body></html>""".encode("utf-8")
            )
            return

        return super().do_GET()

    def end_headers(self):
        """CORS 헤더 추가"""
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
        return super().end_headers()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hyeonto 대시보드 & 마크다운 로컬 서버"
    )
    parser.add_argument("--host", default="127.0.0.1", help="서버 호스트 (기본: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="서버 포트 (기본: 8080)")
    parser.add_argument(
        "--no-browser", action="store_true", help="브라우저 자동 열기 안 함"
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    root = Path(__file__).resolve().parent
    
    # 현재 디렉토리를 hyeonto로 설정
    os.chdir(str(root))

    try:
        with socketserver.TCPServer((args.host, args.port), MyHTTPRequestHandler) as httpd:
            url = f"http://{args.host}:{args.port}/dashboard.html"
            print(f"\n{'='*60}")
            print(f"📂 서빙 경로: {root}")
            print(f"🌐 대시보드: {url}")
            print(f"{'='*60}")
            print(f"📝 마크다운 보기: {url.replace('dashboard.html', 'md_viewer.html?file=...')}")
            print(f"⏹️  종료: Ctrl+C")
            print(f"{'='*60}\n")

            if not args.no_browser:
                webbrowser.open(url)

            httpd.serve_forever()

    except OSError as exc:
        print(f"\n❌ 서버 실행 실패: {exc}")
        if "Address already in use" in str(exc):
            print(f"   포트 {args.port}가 이미 사용 중입니다.")
            print(f"   다른 포트를 사용하세요: python md_server.py --port 9000")
        return 1
    except KeyboardInterrupt:
        print("\n\n✓ 마크다운 서버 종료됨")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
