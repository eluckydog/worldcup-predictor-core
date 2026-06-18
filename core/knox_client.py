"""
Knox.chat LLM 客户端 — OpenAI 风格 Chat Completions 的轻量 HTTP 代理。

支持：
- 流式/非流式对话
- 环境变量或配置文件读取 API Key
- 自定义模型和 base URL
"""

import json
import os
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Callable

# ── 配置路径 ──

_CONFIG_PATH = os.path.expanduser("~/.worldcup/config.yaml")


def _load_api_key() -> Optional[str]:
    """
    从环境变量或配置文件读取 Knox.chat API Key。

    优先级：
    1. 环境变量 KNOX_API_KEY
    2. ~/.worldcup/config.yaml 中的 knox_api_key 字段

    Returns:
        Optional[str]: API Key，未找到返回 None
    """
    # 1) 环境变量
    env_key = os.environ.get("KNOX_API_KEY")
    if env_key:
        return env_key

    # 2) 配置文件
    try:
        with open(_CONFIG_PATH, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("knox_api_key:"):
                    parts = line.split(":", 1)
                    if len(parts) == 2:
                        return parts[1].strip().strip("\"'")
    except (OSError, FileNotFoundError):
        pass

    return None


# =============================================================================
# KnoxClient
# =============================================================================


class KnoxClient:
    """
    Knox.chat Chat Completions 客户端。

    Usage:
        client = KnoxClient()
        response = client.chat([
            {"role": "system", "content": "You are a football expert."},
            {"role": "user", "content": "Analyze this match..."},
        ])
        print(response)
    """

    DEFAULT_MODEL = "anthropic/claude-sonnet-4.6"
    DEFAULT_BASE_URL = "https://api.knox.chat"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 30,
    ):
        """
        Args:
            api_key: Knox.chat API Key（默认从环境变量/配置文件读取）
            model: 模型名称
            base_url: API 基础 URL
            timeout: 请求超时时间（秒）
        """
        self.api_key = api_key or _load_api_key()
        if not self.api_key:
            raise ValueError(
                "KNOX_API_KEY 未设置。请设置环境变量或 "
                f"创建 {_CONFIG_PATH} 文件包含 knox_api_key: YOUR_KEY"
            )

        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    @property
    def _chat_url(self) -> str:
        """Chat Completions API URL"""
        return f"{self.base_url}/v1/chat/completions"

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        stream: bool = False,
        on_chunk: Optional[Callable[[str], None]] = None,
    ) -> str:
        """
        发送对话请求。

        Args:
            messages: 消息列表，格式为 [{"role": "user", "content": "..."}]
            temperature: 温度参数 (0-2)
            max_tokens: 最大生成 token 数
            stream: 是否流式输出
            on_chunk: 流式模式下每收到一个 chunk 的回调

        Returns:
            str: 模型回复内容
        """
        payload: Dict = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }

        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self._chat_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        if stream:
            return self._stream_chat(req, on_chunk)
        else:
            return self._non_stream_chat(req)

    def _non_stream_chat(self, request: urllib.request.Request) -> str:
        """非流式对话"""
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Knox API HTTP {e.code}: {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Knox API 连接失败: {e.reason}") from e

        choices = data.get("choices", [])
        if not choices:
            return ""

        message = choices[0].get("message", {})
        return message.get("content", "")

    def _stream_chat(
        self,
        request: urllib.request.Request,
        on_chunk: Optional[Callable[[str], None]],
    ) -> str:
        """流式对话"""
        full_content: List[str] = []

        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as resp:
                buffer = ""
                while True:
                    chunk = resp.read(4096)
                    if not chunk:
                        break
                    buffer += chunk.decode("utf-8", errors="replace")
                    # 解析 SSE 格式：data: {...}
                    while "\n" in buffer:
                        line, buffer = buffer.split("\n", 1)
                        line = line.strip()
                        if line.startswith("data: "):
                            json_str = line[6:]
                            if json_str == "[DONE]":
                                break
                            try:
                                data = json.loads(json_str)
                                delta = (
                                    data.get("choices", [{}])[0]
                                    .get("delta", {})
                                    .get("content", "")
                                )
                                if delta:
                                    full_content.append(delta)
                                    if on_chunk:
                                        on_chunk(delta)
                            except json.JSONDecodeError:
                                pass
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Knox API HTTP {e.code} (stream): {error_body[:500]}"
            ) from e
        except urllib.error.URLError as e:
            raise RuntimeError(f"Knox API 连接失败 (stream): {e.reason}") from e

        return "".join(full_content)

    def is_available(self) -> bool:
        """
        检查 Knox API 是否可用。

        Returns:
            bool: API Key 已配置且可达
        """
        if not self.api_key:
            return False
        try:
            self.chat(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
                temperature=0.0,
            )
            return True
        except (RuntimeError, ValueError):
            return False
