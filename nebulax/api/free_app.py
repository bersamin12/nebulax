"""Single-process demo entry point: keep concurrent predictions within a small RAM budget."""
import threading

from starlette.responses import JSONResponse


class PredictionGate:
    def __init__(self, application):
        self.application = application
        self.lock = threading.Lock()

    async def __call__(self, scope, receive, send):
        path = scope.get("path", "")
        costly = (scope["type"] == "http" and scope.get("method") == "POST"
                  and path.startswith("/api/ps3/")
                  and (path.endswith(("/predict", "/stream", "/validate", "/save")) or "/local/" in path))
        if not costly:
            return await self.application(scope, receive, send)
        if not self.lock.acquire(blocking=False):
            response = JSONResponse({"detail": "Another prediction is running. Please retry after it finishes."},
                                    status_code=503, headers={"Retry-After": "5"})
            return await response(scope, receive, send)
        try:
            await self.application(scope, receive, send)
        finally:
            self.lock.release()


def create_app():
    from .main import app
    return PredictionGate(app)
