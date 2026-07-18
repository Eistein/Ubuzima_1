# The tone problem

## What is actually broken

Kinyarwanda tone is contrastive. Kimenyi documents pairs distinguished by tone alone:

| tone-marked | gloss | tone-marked | gloss |
|---|---|---|---|
| indá | louse | inda | stomach |
| umuryáango | door | umuryaango | family |

Standard Kinyarwanda orthography does not mark tone. So both members of each pair
are **the same string** once written normally. `python tone_probe.py collapse` prints
this in one second:

```
indá        louse     ->  inda
inda        stomach   ->  inda        IDENTICAL
umuryáango  door      ->  umuryaango
umuryaango  family    ->  umuryaango  IDENTICAL
```

The synthesiser is handed one string and must return one waveform. It cannot
distinguish "louse" from "stomach". Not *does not distinguish them well* — **cannot**.
The information is absent from the input before the model ever runs.

This is why voice cloning does not fix it. Speaker conditioning changes *who* is
talking. It cannot recover *what was never written down*.

## Why the field hasn't fixed it either

- The mbazaNLP FastPitch model card states the model "does not always capture the
  Kinyarwanda tones" and recommends training future models against a tonal dictionary.
- `Kinyarwanda_YourTTS_v1` sets `use_phonemes: False` — it is a grapheme model.
- No public tone-marked Kinyarwanda lexicon exists that I could find. **That gap is
  the bottleneck for the whole language, not just for you.**

## The one fact that makes this tractable

Kimenyi's **One Tone Rule**: *a word has only one phonemic tone.* Simple nouns may
carry a secondary tone, compounds a tertiary, both derived from the underlying one.

That matters enormously. You do not need to annotate every syllable. You need **one
tone position per word**. A problem that looks like per-mora annotation of a whole
language collapses into a lexicon of roots plus a rule system — and the rule system
is already written down (Kimenyi, 2002).

Two further wrinkles, so you don't oversell it: Kinyarwanda tone rules apply
**right-to-left**, unlike most Bantu languages, and verbs carry morphological and
grammatical tone on top of lexical tone. Nouns are the tractable end. Verbs are the
research.

## What to do, by how much time you have

### Now (2 hours) — measure it, don't fix it

This is the move. You are a final-year student with a deadline, not a tonologist with
a grant.

```bash
python tone_probe.py collapse          # the proof, instant, no model needed
python tone_probe.py measure --human recordings/ --speaker-wav voices/ganza/reference.wav
```

For `measure`: record yourself saying each pair member in a carrier phrase, as
`recordings/pair1_a.wav`, `pair1_b.wav`, and so on. The script extracts F0, normalises
to semitones over each utterance's own median (so human and synthetic are comparable),
and reports mean human tonal contrast against mean synthesised contrast.

The synthesised number will be approximately zero, because the inputs were identical.
That zero, next to your measured human contrast in semitones, is the finding.

**Add eight more pairs first.** `MINIMAL_PAIRS` in `tone_probe.py` ships Kimenyi's two.
Ten pairs is a probe; two is an anecdote. You are the native speaker — health-domain
pairs would be ideal, since that is the register your system speaks in.

### Next project (months) — automatic tone annotation

You are unusually well placed for this, and you should say so in Chapter 6.

1. **Force-align** a Kinyarwanda speech corpus. You already have the aligner: your own
   LoRA-fine-tuned W2V-BERT CTC model emits frame-level alignments for free.
2. **Extract F0** per mora (pyin, CREPE, or REAPER).
3. **Normalise** register per speaker and per utterance.
4. **Classify** each mora H or L against the local register.
5. **Constrain with the One Tone Rule** — one phonemic tone per word kills most of the
   classification noise. This is the step that makes it work.
6. **Publish the tone-marked lexicon.** This is the actual contribution. Nobody has one.
7. **Retrain TTS** on tone-marked graphemes (add á to the character set).

Step 6 is worth more than step 7. A tone-marked Kinyarwanda lexicon would be used by
every Kinyarwanda TTS, ASR and MT system after you — and Muhirwe (2010) already showed
that morphological analysis of tone-marked Kinyarwanda text is measurably less
ambiguous than of unmarked text, so the benefit is not limited to synthesis.

### Alternative — rule-based, if a lexicon appears

Nzeyimana's KinLP morphological analyser (the one underneath KinyaBERT) plus a
tone-marked root lexicon plus Kimenyi's rules would generate tone marks without any
audio. The analyser exists. The lexicon does not. That is the whole blocker.

## What NOT to do

- **Don't post-process F0** with a vocoder to "add" tone. You would be guessing the
  contour from text you cannot read tone from. Artefacts, and no linguistic basis.
- **Don't record more data.** More untoned text trains a better untoned model.
- **Don't quantise, cache, or optimise.** None of this is a compute problem.

## For the report

§5.4, after the tone probe gives you numbers:

> Speaker-conditioned synthesis reproduced speaker timbre but not lexical tone. This
> is a property of the input representation rather than of the model's capacity:
> standard Kinyarwanda orthography does not mark tone, and the synthesiser was
> trained on graphemes with phonemisation disabled, so tonally contrastive pairs such
> as indá 'louse' and inda 'stomach' (Kimenyi, 2002) are presented to the model as an
> identical input string and are necessarily realised identically. A probe over N
> minimal pairs measured a mean tonal contrast of X semitones in natural speech
> against Y semitones in synthesis, confirming that the contrast is not attenuated but
> absent.

Chapter 6, replacing whichever of your fifteen recommendations is weakest:

> **R1. Build and release a tone-marked Kinyarwanda lexicon.** Tonal prosody is the
> principal quality gap in Kinyarwanda speech synthesis, and it is a data problem
> rather than a modelling one. Kimenyi's One Tone Rule — that a word carries a single
> phonemic tone — bounds the annotation task to one tone position per lexical root
> rather than per mora. A forced-alignment pipeline built on the CTC model fine-tuned
> in this study could derive candidate tone assignments from F0 over an aligned
> corpus, constrained by that rule and verified against Kimenyi (2002). The resulting
> lexicon would benefit Kinyarwanda synthesis, morphological analysis (Muhirwe, 2010)
> and machine translation alike.

That recommendation is worth more than the other fourteen combined, because it is
specific, it is grounded in your own measurement, and you are the person with the
aligner already trained.

## References

Kimenyi, A. (2002). *A Tonal Grammar of Kinyarwanda: An Autosegmental and Metrical
Analysis.* Lewiston, NY: Edwin Mellen Press.

Kimenyi, A. (n.d.). *Kinyarwanda Tones Made Easy.* California State University,
Sacramento.

Muhirwe, J. (2010). Morphological analysis of tone marked Kinyarwanda text. In
A. Yli-Jyrä, A. Kornai, J. Sakarovitch, & B. Watson (Eds.), *Finite-State Methods and
Natural Language Processing* (pp. 48-55). Berlin: Springer.

Myers, S. (2003). F0 timing in Kinyarwanda. *Phonetica, 60*(2), 71-97.
