import hashlib
import hmac
import json
import os
import time
from typing import Union

import aiohttp
import six


class PusherStreamer:
    def __init__(self, channel: str | None = None) -> None:
        self.auth_key = os.environ.get("PUSHER_AUTH_KEY")
        self.auth_secret = os.environ.get("PUSHER_AUTH_SECRET", "").encode("utf8")
        self.auth_version = os.environ.get("PUSHER_AUTH_VERSION")

        self.base = f"https://api-{os.environ.get('PUSHER_APP_CLUSTER')}.pusher.com"
        self.path = f"/apps/{os.environ.get('PUSHER_APP_ID')}/events"
        self.headers = {
            "X-Pusher-Library": f"pusher-http-python {self.auth_version}",
            "Content-Type": "application/json",
        }

        self.channel = channel
        self.session: aiohttp.ClientSession | None = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        if self.session:
            await self.session.close()

    def _generate_query_string(self, params: dict) -> str:
        body_md5 = hashlib.md5(json.dumps(params).encode("utf-8")).hexdigest()
        query_params: dict = {
            "auth_key": self.auth_key,
            "auth_timestamp": str(int(time.time())),
            "auth_version": self.auth_version,
            "body_md5": six.text_type(body_md5),
        }
        query_string = "&".join(map("=".join, sorted(query_params.items(), key=lambda x: x[0])))
        auth_string = "\n".join(["POST", self.path, query_string]).encode("utf8")
        signature_encoded = hmac.new(self.auth_secret, auth_string, hashlib.sha256).hexdigest()
        query_params["auth_signature"] = six.text_type(signature_encoded)
        query_string += "&auth_signature=" + query_params["auth_signature"]

        return query_string

    async def push_event(self, name: str, data: str) -> Union[dict, str]:
        params = {"name": name, "data": data, "channels": [self.channel]}
        query_string = self._generate_query_string(params)
        url = f"{self.base}{self.path}?{query_string}"
        body = json.dumps(params)

        if not self.session:
            raise RuntimeError("Streamer session not initialized. Use 'async with Streamer(...)' context.")

        try:
            async with self.session.post(url=url, data=body, headers=self.headers) as response:
                if response.headers.get("Content-Type") == "application/json":
                    return await response.json()
                return await response.text()
        except aiohttp.ClientError as exc:
            return str(exc)


__all__ = ["PusherStreamer"]
