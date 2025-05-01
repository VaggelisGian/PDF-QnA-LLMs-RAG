from redis import Redis
import os
from dotenv import load_dotenv

load_dotenv()

class RedisClient:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
        connect_timeout = int(os.getenv('REDIS_CONNECT_TIMEOUT', '5'))
        socket_timeout = int(os.getenv('REDIS_SOCKET_TIMEOUT', '5'))
        try:
            self.client = Redis.from_url(
                redis_url,
                socket_connect_timeout=connect_timeout,
                socket_timeout=socket_timeout,
                decode_responses=True
            )
            self.client.ping()
            print(f"Redis client connected to: {redis_url}")
        except Exception as e:
            print(f"FATAL: Failed to connect Redis client to {redis_url}: {e}")
            self.client = None

    def set_value(self, key, value, expiry_seconds=None):
        if not self.client:
            print("ERROR: Redis client not initialized.")
            return
        try:
            if expiry_seconds:
                self.client.setex(key, expiry_seconds, value)
            else:
                self.client.set(key, value)
        except Exception as e:
            print(f"ERROR setting value for key '{key}' in Redis: {e}")

    def get_value(self, key):
        if not self.client:
            print("ERROR: Redis client not initialized.")
            return None
        try:
            return self.client.get(key)
        except Exception as e:
            print(f"ERROR getting value for key '{key}' from Redis: {e}")
            return None

    def delete_value(self, key):
        if not self.client:
            print("ERROR: Redis client not initialized.")
            return
        try:
            self.client.delete(key)
        except Exception as e:
            print(f"ERROR deleting key '{key}' from Redis: {e}")

    def exists(self, key):
        if not self.client:
            print("ERROR: Redis client not initialized.")
            return False
        try:
            return self.client.exists(key) > 0
        except Exception as e:
            print(f"ERROR checking existence of key '{key}' in Redis: {e}")
            return False

    def close(self):
        if self.client:
            try:
                self.client.close()
                print("Redis client connection closed.")
            except Exception as e:
                print(f"ERROR closing Redis client connection: {e}")
            finally:
                self.client = None
