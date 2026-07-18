# Deployment guide

## A GPU is not required

This surprises people, so here is the reasoning.

**Every local model in this pipeline is non-autoregressive.**

| Stage | Architecture | Compute shape |
|---|---|---|
| ASR — badrex W2V-BERT 2.0 | Transformer encoder + **CTC** head | ONE forward pass, then argmax |
| TTS — YourTTS | **VITS** (parallel flow-based) | ONE forward pass |
| LLM — Gemini 2.5 Flash | network call | no local compute at all |

Nothing loops. There is no token-by-token decoder anywhere on your hardware. That
is the whole reason this runs acceptably on 2 vCPU.

Contrast the obvious alternatives. Whisper has an autoregressive decoder — it emits
one token at a time, so a 200-word transcript is 200 sequential forward passes.
Tacotron 2 is autoregressive over mel frames. Had this study picked Whisper + Tacotron,
CPU deployment would be miserable. CTC + VITS is what makes it cheap.

This is a design finding, not an accident of luck. It belongs in the report.

## Memory, which IS the real constraint

```
badrex W2V-BERT 2.0    583.7M params    2.33 GB fp32
YourTTS (VITS + SE)    ~290M  params    1.16 GB fp32
weights total                           3.49 GB
+ activations / runtime                ~2.0  GB
                                       -------
                                       ~5.5  GB
```

- HF Spaces CPU basic: **16 GB** → fits comfortably.
- Render / Vercel standard tiers: ~4 GB → does not fit.

So §3.4's rejection of Render and Vercel stands, but the reason is **RAM, not GPU**.
Fix that sentence in the report.

## Cold start

First request after a restart loads both models from disk. On CPU expect longer than
the 22–28 s the report records for a T4. It happens at most once per restart, and
§5.3 excludes it.

## Order of operations

1. Push the LoRA adapter to the Hub (`scripts/push_adapter.py`).
2. Create the Space on **CPU basic**.
3. Set `GEMINI_API_KEY` as a Secret, `ADAPTER_ID` and `SPEAKER_WAV` as Variables.
4. Push the code.
5. Telemetry tab → confirm the ASR line reads `... + LoRA (<your-id>) [cpu]`.
   If it says **NO ADAPTER**, step 1 or 3 failed.
6. Ask three questions, then run `benchmark.py --latency 50` for Table 5.8.

## Do you need LOW_RESOURCE=1?

Probably not — measure first.

`LOW_RESOURCE=1` applies int8 dynamic quantisation. It roughly halves CPU inference
time, but it **changes the model**, so the Chapter 5 WER would no longer describe
what you serve and you would have to re-benchmark.

At fp32 on CPU you are running numerically the same model your notebook evaluated
(cell 26 loads `torch_dtype=torch.float32`), so **Chapter 5's 6.76% / 31.29% describes
your deployment exactly**. That is worth real marks. Do not throw it away to save a
few seconds unless the latency is genuinely unacceptable — and if you do quantise,
report both numbers.

## Checks before the defence

- [ ] Telemetry tab shows the adapter loaded, not the base model
- [ ] A diagnostic question ("ndwaye iki?") triggers the referral, not an LLM answer
- [ ] English speech returns "UBUZIMA AI yumva Ikinyarwanda gusa"
- [ ] The disclaimer appears on every response above ~120 characters
- [ ] Privacy & Terms tab renders and is reachable without leaving the app
- [ ] Space is public and reachable from a phone (NFR5)
- [ ] No key anywhere in git history
