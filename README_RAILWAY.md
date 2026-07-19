# UBUZIMA AI — Railway deployment

## Repo layout this expects

```
your-repo/
  app.py
  requirements.txt
  Dockerfile
  .dockerignore
  adapter/                  <- your 15MB LoRA adapter, committed directly
    adapter_config.json
    adapter_model.safetensors
    preprocessor_config.json
    tokenizer_config.json
    vocab.json
    ...whatever else Wav2Vec2BertProcessor.from_pretrained() needs
```

Commit the `adapter/` folder into git as-is — at 15MB it's well within
normal git limits, no Git LFS needed.

## Deploy steps

1. Push this repo to GitHub.
2. In Railway: **New Project → Deploy from GitHub repo** → select it.
3. Railway will detect the `Dockerfile` automatically and build from it —
   you don't need to set a build/start command manually.
4. Go to your service → **Variables** and add:
   - `OPENROUTER_API_KEY` = your key from openrouter.ai/keys
   - `HF_TOKEN` = only if badrex's base model is gated/private (it likely
     isn't, so this is optional)
5. Go to **Settings → Networking → Generate Domain** to get a public URL.
   Railway auto-injects `$PORT`; `app.py` already reads it, so you don't
   need to set it yourself.
6. Watch the **Deploy Logs**. You should see:
   ```
   Loading ASR...
   ASR ready (badrex + your LoRA adapter)
   Loading MMS-TTS...
   MMS-TTS ready
   Pipeline ready
   ```
   before Gradio starts. If it crashes before that, the traceback will
   point at exactly which model failed to load.

## What's different from the Colab notebook

- No `google.colab` / Drive mount — the adapter is bundled in the repo
  instead of read from Drive.
- No interactive `getpass()` — `OPENROUTER_API_KEY` and `HF_TOKEN` come
  from Railway's environment variables.
- TTS is **MMS-TTS only** — the notebook's Coqui YourTTS-first-with-fallback
  logic was dropped. Coqui TTS needs `espeak-ng` and downloads an extra
  ~100MB+ model at startup; on a metered, CPU-only platform that's a
  reliability and cost risk for not much quality gain over MMS-TTS. If you
  want YourTTS back later, say so and I'll wire it back in behind an env
  var flag.
- `demo.launch(share=True, debug=True)` → `demo.launch(server_name="0.0.0.0",
  server_port=port)`. `share=True` creates a temporary Gradio tunnel URL,
  which is redundant (and unreliable long-term) once you have a real
  Railway domain; `debug=True` is a local dev convenience.

## Cost note

Railway bills per-second on CPU/RAM usage, not a flat fee. This pipeline
loads three models into memory (ASR base + adapter + TTS) and keeps them
resident for as long as the service runs — expect noticeably more than
Railway's $5 trial credit if you leave it running continuously rather than
only during active testing/demo windows.
