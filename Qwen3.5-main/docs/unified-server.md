# Unified Server Notes

The app now runs API and GUI in one process/port:

- API routes: `/v1/*`
- GUI mount path: `/ui` (configurable via `APP_UI_PATH`)

## Start

```bash
uv run python api_server.py
```

or:

```bash
uv run python gui.py
```

Both start the same unified server.

## Fallback precedence

For optional API request fields (model/base_url/tokens/thinking/api_key):

1. request value
2. env value
3. hardcoded default

Prompt routing precedence:

1. `prompt_text`
2. `prompt_id`
3. `prompt_name`
4. messages text

## Security controls

- `API_AUTH_TOKEN`: when set, `/v1/*` requires `Authorization: Bearer <token>`.
- `GUI_LOCAL_ONLY=1`: only loopback clients can open GUI path.
- `API_BLOCK_PRIVATE_URLS=1`: blocks private/loopback hosts for HTTP(S) `video_url`.
- `API_VIDEO_URL_TIMEOUT_SECONDS`, `API_VIDEO_URL_MAX_MB`: remote download guardrails.

## Key modules

- `src/qwen_image/config.py`
- `src/qwen_image/app.py`
- `src/qwen_image/api/routes.py`
- `src/qwen_image/inference/service.py`
- `src/qwen_image/ui/adapter.py`
