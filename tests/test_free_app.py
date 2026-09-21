import asyncio
from nebulax.api.free_app import PredictionGate


def test_only_one_prediction_but_health_stays_available():
    async def run():
        entered, release = asyncio.Event(), asyncio.Event()
        async def app(scope, receive, send):
            if scope['path'].endswith('/predict'):
                entered.set()
                await release.wait()
            await send({'type': 'http.response.start', 'status': 200, 'headers': []})
            await send({'type': 'http.response.body', 'body': b'ok'})
        gate = PredictionGate(app)
        async def request(path, method='POST'):
            messages = []
            async def send(message): messages.append(message)
            async def receive(): return {'type': 'http.request', 'body': b''}
            await gate({'type': 'http', 'method': method, 'path': path}, receive, send)
            return messages[0]['status']
        first = asyncio.create_task(request('/api/ps3/rail/predict'))
        await entered.wait()
        assert await request('/api/ps3/shm/predict') == 503
        assert await request('/api/health', 'GET') == 200
        release.set()
        assert await first == 200
        assert await request('/api/ps3/shm/predict') == 200
    asyncio.run(run())


def test_prediction_gate_releases_after_failure():
    async def run():
        async def app(scope, receive, send): raise RuntimeError('failed')
        gate = PredictionGate(app)
        try:
            await gate({'type': 'http', 'method': 'POST', 'path': '/api/ps3/door/predict'}, None, None)
        except RuntimeError:
            pass
        assert not gate.lock.locked()
    asyncio.run(run())
