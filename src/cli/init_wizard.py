"""Interactive setup wizard for mul-agent.

Usage: mulagent init
"""

from __future__ import annotations

import shutil
from pathlib import Path

# ANSI colors
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_BOLD = "\033[1m"
_NC = "\033[0m"

_PROVIDERS = {
    "1": {
        "id": "qwen",
        "name": "Qwen (通义千问)",
        "model": "qwen3.5-plus",
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "max_tokens": 65536,
    },
    "2": {
        "id": "deepseek",
        "name": "DeepSeek",
        "model": "deepseek-chat",
        "base_url": "https://api.deepseek.com/v1",
        "max_tokens": 8192,
    },
    "3": {
        "id": "openai",
        "name": "OpenAI",
        "model": "gpt-4o",
        "base_url": "",
        "max_tokens": 4096,
    },
    "4": {
        "id": "ollama",
        "name": "Ollama (本地)",
        "model": "qwen2.5:14b",
        "base_url": "http://localhost:11434/v1",
        "max_tokens": 4096,
        "api_key": "ollama",
    },
}


def _ask(prompt: str, default: str = "") -> str:
    suffix = f" [{default}]" if default else ""
    val = input(f"  {prompt}{suffix}: ").strip()
    return val or default


def _ask_yn(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    val = input(f"  {prompt} ({hint}): ").strip().lower()
    if not val:
        return default
    return val in ("y", "yes")


def run_init() -> None:
    """Interactive setup wizard."""
    from cli import ensure_src_path
    ensure_src_path()
    from common.config import CONFIG_DIR

    config_path = CONFIG_DIR / "settings.yaml"
    example_path = CONFIG_DIR / "settings.yaml.example"

    print()
    print(f"{_CYAN}{_BOLD}  mul-agent 初始化向导{_NC}")
    print(f"  {'─' * 40}")
    print()

    # Check existing config
    if config_path.exists():
        if not _ask_yn(f"{_YELLOW}config/settings.yaml 已存在，覆盖？{_NC}", default=False):
            print(f"\n  {_DIM}跳过，保留现有配置{_NC}\n")
            return

    # Ensure config dir exists
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Choose LLM provider
    print(f"  {_BOLD}1. 选择 LLM 提供商{_NC}")
    print()
    for k, v in _PROVIDERS.items():
        print(f"     {k}) {v['name']}  {_DIM}({v['model']}){_NC}")
    print(f"     5) 自定义")
    print()

    choice = _ask("选择", "1")

    if choice in _PROVIDERS:
        prov = _PROVIDERS[choice]
        model_id = prov["id"]
        model_name = prov["name"]
        model_model = prov["model"]
        base_url = prov["base_url"]
        max_tokens = prov["max_tokens"]
        api_key = prov.get("api_key", "")
    else:
        # Custom
        model_id = _ask("模型 ID (如 my-llm)", "custom")
        model_name = _ask("显示名称", model_id)
        model_model = _ask("模型名 (如 gpt-4o)")
        base_url = _ask("API Base URL")
        max_tokens = int(_ask("Max tokens", "4096"))
        api_key = ""

    # Step 2: API Key
    if not api_key:
        print()
        print(f"  {_BOLD}2. API Key{_NC}")
        api_key = _ask(f"{model_name} API Key")
        if not api_key:
            print(f"  {_YELLOW}未填写 API Key，启动时需要手动配置{_NC}")

    # Step 3: Feishu (optional)
    print()
    print(f"  {_BOLD}3. 飞书机器人 (可选){_NC}")
    feishu_id = ""
    feishu_secret = ""
    if _ask_yn("配置飞书 Bot？", default=False):
        feishu_id = _ask("App ID")
        feishu_secret = _ask("App Secret")

    # Step 4: Write config
    print()
    print(f"  {_BOLD}4. 生成配置文件{_NC}")

    # Build YAML content
    base_url_line = f'      base_url: "{base_url}"' if base_url else '      base_url: ""'
    yaml_content = f"""app:
  name: "mul-agent"
  version: "0.6.0"
  debug: true

server:
  host: "0.0.0.0"
  port: 8000

database:
  url: "postgresql+asyncpg://mulagent:mulagent@localhost:5432/mulagent"
  echo: false

redis:
  url: "redis://localhost:6379/0"

qdrant:
  url: "http://localhost:6333"
  collection_name: "case_library"

llm:
  default: "{model_id}"
  models:
    {model_id}:
      name: "{model_name}"
      model: "{model_model}"
      api_key: "{api_key}"
{base_url_line}
      max_tokens: {max_tokens}

feishu:
  app_id: "{feishu_id}"
  app_secret: "{feishu_secret}"
  autostart: false

security:
  max_sandbox_timeout_seconds: 300
  max_sandbox_memory_mb: 512
"""

    config_path.write_text(yaml_content)
    print(f"  {_GREEN}✓{_NC} 已写入 {config_path}")

    # Copy example if not exists
    if example_path.exists() and not config_path.exists():
        shutil.copy2(example_path, config_path)

    print()
    print(f"  {_GREEN}{_BOLD}初始化完成！{_NC}")
    print()
    print(f"  启动方式:")
    print(f"    mulagent              {_DIM}# TUI 模式{_NC}")
    print(f"    mulagent --headless   {_DIM}# 纯文本模式{_NC}")
    print(f"    mulagent -c \"你好\"    {_DIM}# 单次执行{_NC}")
    print()
    if not api_key:
        print(f"  {_YELLOW}提醒: 请先编辑 config/settings.yaml 填入 API Key{_NC}")
        print()
