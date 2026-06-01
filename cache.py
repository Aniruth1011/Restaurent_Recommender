"""Tiny cache: Redis if available, else in-memory dict. JSON values only here."""
import json

try:
    import redis
    _redis_available = True
except ImportError:
    _redis_available = False

_mem = {}


class Cache:
    def __init__(self, host="localhost", port=6379, ttl=1800):
        self.ttl = ttl
        self.client = None
        if _redis_available:
            try:
                self.client = redis.Redis(host=host, port=port, db=0, socket_timeout=2)
                self.client.ping()
                print("Redis connected")
            except Exception:
                self.client = None
        if self.client is None:
            print("Using in-memory cache")

    def _key(self, ns, key):
        return f"{ns}:{key}"

    def set_json(self, ns, key, obj):
        k = self._key(ns, key)
        data = json.dumps(obj)
        if self.client:
            self.client.setex(k, self.ttl, data)
        else:
            _mem[k] = data

    def get_json(self, ns, key):
        k = self._key(ns, key)
        if self.client:
            data = self.client.get(k)
            return json.loads(data) if data else None
        return json.loads(_mem[k]) if k in _mem else None

    def delete_prefix(self, ns, prefix):
        if self.client:
            for k in self.client.scan_iter(f"{ns}:{prefix}*"):
                self.client.delete(k)
        else:
            for k in [k for k in _mem if k.startswith(f"{ns}:{prefix}")]:
                _mem.pop(k, None)


cache = Cache()
