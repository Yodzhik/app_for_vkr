"""Локальный HTTP-интерфейс. Запуск: python app.py [--port 8000]."""
from __future__ import annotations

import argparse
import logging
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

BASE_DIR = Path(__file__).resolve().parent
HOST = "127.0.0.1"
MAX_BODY_BYTES = 16_384
LOG = logging.getLogger(__name__)

# Импорт не загружает модели и не открывает сокет.
try:
    from prediction import InputError, ModelLoadError, PredictionService
    from web_ui import TEMPLATE_PATH, render_page
except ImportError:
    if __name__ == "__main__":
        print("Не установлены зависимости. Выполните: python -m pip install -r requirements-app.txt", file=sys.stderr)
        raise SystemExit(1)
    raise


def create_handler(service: PredictionService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        server_version = "CompositeLocal/1.1"
        sys_version = ""
        timeout = 10

        def send_content(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; form-action 'self'; frame-ancestors 'none'; base-uri 'none'")
            self.end_headers()
            self.wfile.write(data)

        def send_html(self, html: str, status: int = 200) -> None:
            self.send_content(html.encode("utf-8"), "text/html; charset=utf-8", status)

        def trusted_request(self) -> bool:
            port = self.server.server_address[1]
            allowed = {f"127.0.0.1:{port}", f"localhost:{port}"}
            if self.headers.get("Host") not in allowed:
                self.send_error(403, "Local requests only")
                return False
            origin = self.headers.get("Origin")
            if origin and origin not in {f"http://{host}" for host in allowed}:
                self.send_error(403, "Cross-origin requests are not allowed")
                return False
            return True

        def do_GET(self) -> None:
            if not self.trusted_request():
                return
            parsed = urlparse(self.path)
            assets = {"/static/app.css": ("app.css", "text/css; charset=utf-8"), "/static/app.js": ("app.js", "text/javascript; charset=utf-8")}
            if parsed.path in assets:
                filename, mime = assets[parsed.path]
                self.send_content((BASE_DIR / "static" / filename).read_bytes(), mime)
                return
            if parsed.path == "/favicon.ico":
                self.send_content(b"", "image/x-icon", 204)
                return
            if parsed.path != "/":
                self.send_error(404)
                return
            try:
                mode = parse_qs(parsed.query, max_num_fields=10).get("mode", ["properties"])[0]
                self.send_html(render_page(service, mode))
            except ValueError as exc:
                self.send_html(render_page(service, error=str(exc)), 400)

        def do_POST(self) -> None:
            if not self.trusted_request():
                return
            if urlparse(self.path).path != "/predict":
                self.send_error(404)
                return
            if self.headers.get_content_type() != "application/x-www-form-urlencoded":
                self.send_error(415)
                return
            try:
                length = int(self.headers.get("Content-Length", "-1"))
            except ValueError:
                self.send_error(400, "Invalid Content-Length")
                return
            if length < 0 or length > MAX_BODY_BYTES:
                self.send_error(413)
                return
            mode, values = "properties", {}
            try:
                body = self.rfile.read(length).decode("utf-8", errors="strict")
                form = parse_qs(body, keep_blank_values=True, encoding="utf-8", errors="strict", max_num_fields=30)
                if any(len(items) != 1 for items in form.values()):
                    raise InputError("Поля запроса не должны повторяться.")
                mode = form.get("mode", ["properties"])[0]
                task = service.task(mode)
                unknown = set(form) - set(task.features) - {"mode"}
                if unknown:
                    raise InputError("В запросе присутствуют неизвестные поля.")
                values = {name: form.get(name, [""])[0] for name in task.features}
                result = service.predict(mode, values)
                self.send_html(render_page(service, mode, result.values, result))
            except (InputError, UnicodeError, ValueError) as exc:
                if mode not in service.tasks:
                    mode = "properties"
                self.send_html(render_page(service, mode, values, error=str(exc)), 400)
            except Exception:
                LOG.exception("Ошибка прогнозирования")
                self.send_html(render_page(service, mode, values, error="Не удалось выполнить расчёт. Перезапустите приложение; подробности в терминале."), 500)

        def log_message(self, format: str, *args: object) -> None:
            LOG.debug(format, *args)

    return Handler


def create_server(service: PredictionService, port: int = 8000) -> ThreadingHTTPServer:
    # Адрес нельзя изменить аргументом командной строки: только этот компьютер.
    return ThreadingHTTPServer((HOST, port), create_handler(service))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Локальный прогноз свойств композитов")
    parser.add_argument("--port", type=int, default=8000, help="Порт от 1024 до 65535 (по умолчанию 8000)")
    args = parser.parse_args(argv)
    if not 1024 <= args.port <= 65535:
        parser.error("Порт должен быть от 1024 до 65535.")
    try:
        service = PredictionService.load(BASE_DIR / "models.joblib", BASE_DIR / "examples.json")
        if not TEMPLATE_PATH.is_file():
            raise ModelLoadError("Не найден templates/page.html. Распакуйте полный комплект приложения.")
        for filename in ("app.css", "app.js"):
            if not (BASE_DIR / "static" / filename).is_file():
                raise ModelLoadError(f"Не найден static/{filename}. Распакуйте полный комплект приложения.")
        server = create_server(service, args.port)
    except ModelLoadError as exc:
        print(f"Ошибка запуска: {exc}", file=sys.stderr)
        return 1
    except OSError:
        print(f"Не удалось открыть порт {args.port}. Закройте другую копию приложения или выполните: python app.py --port {args.port + 1 if args.port < 65535 else 8000}", file=sys.stderr)
        return 1
    print(f"Приложение запущено: http://{HOST}:{args.port}", flush=True)
    print("Откройте адрес в браузере. Остановка сервера: Ctrl+C.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nПриложение остановлено.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
