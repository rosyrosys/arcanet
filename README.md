# ArcaNet

**A dependency-free reference implementation of Phrase-Level Combinatorial AI (PLCA).**

ArcaNet is the companion prototype to the paper
*"Phrase-Level Combinatorial Generation: A Historically Grounded Alternative
to End-to-End Generative Music AI"* (submitted to *Computer Music Journal*, 2026).

It is a deliberate homage to Athanasius Kircher's **Arca Musarithmica** (1650):
a bank of pre-composed phrases, a neural-style selector over them,
a hard constraint layer enforcing grammar and voice-leading, and a
human-readable selection trace. The whole system is ~550 lines of pure
Python with no third-party dependencies.

## Listen

Pre-rendered audio previews are available on the project page:
**https://<your-github-username>.github.io/arcanet/**

Or play the MP3s directly:

| Sample | Form | Bars | MIDI | Audio |
|--------|------|------|------|-------|
| 1 | AABA_16 | 16 | [`examples/arcanet_sample_01.mid`](examples/arcanet_sample_01.mid) | [`docs/audio/arcanet_sample_01.mp3`](docs/audio/arcanet_sample_01.mp3) |
| 2 | sentence_8 | 8 | [`examples/arcanet_sample_02.mid`](examples/arcanet_sample_02.mid) | [`docs/audio/arcanet_sample_02.mp3`](docs/audio/arcanet_sample_02.mp3) |
| 3 | free_14 | 14 | [`examples/arcanet_sample_03.mid`](examples/arcanet_sample_03.mid) | [`docs/audio/arcanet_sample_03.mp3`](docs/audio/arcanet_sample_03.mp3) |

All three were produced by a single run of `python arcanet.py` with the
fixed seed `2026`. Running that command again should reproduce the MIDI
files byte-for-byte.

## Quick start

ArcaNet requires only Python 3.9+. No `pip install` step is necessary
to run the composer itself.

```bash
git clone https://github.com/<your-github-username>/arcanet.git
cd arcanet
python arcanet.py
```

This produces, in `output/`:

```
arcanet_sample_01.mid   # 16-bar A-A'-B-A' form
arcanet_sample_02.mid   # 8-bar antecedent-consequent
arcanet_sample_03.mid   # 14-bar free combination
arcanet_log.txt         # per-phrase selection trace (top-3 candidates)
arcanet_stats.json      # phrase-reuse and constraint-hit rates
```

The selection trace records, for every slot in every composition, the
top three candidates the selector considered and their softmax
probabilities. This is the key affordance of the PLCA approach: every
compositional decision is inspectable and editable.

## Regenerating the audio (optional)

The repository ships with pre-rendered MP3 files so no synthesis is
required to hear the output. If you would like to re-render them
yourself, install `numpy` and `ffmpeg` and run:

```bash
pip install numpy
python render_audio.py
```

This uses a minimal in-repo SMF parser and a small additive-synthesis
"piano" voice to produce `docs/audio/*.mp3` from `examples/*.mid`.
It is not a high-fidelity rendering (no soundfont, no pedal, no
stereo) — it exists purely to produce shareable audio previews. For
a faithful rendering, open the MIDI files in any DAW (Logic Pro,
Ableton Live, REAPER, MuseScore) with a real piano instrument.

## Repository layout

```
arcanet.py              # the composer (~550 LOC, stdlib only)
render_audio.py         # MP3 preview renderer (numpy + ffmpeg)
examples/               # deterministic MIDI + selection traces (seed 2026)
docs/                   # GitHub Pages static site
  index.html            # landing page
  samples.html          # audio gallery with embedded players
  code.html             # annotated code walkthrough
  audio/                # pre-rendered MP3 previews
  assets/style.css
CITATION.cff            # machine-readable citation metadata
LICENSE                 # MIT
README.md               # this file
```

## How it works (one-paragraph version)

ArcaNet treats the **phrase** (a fully formed one- or two-bar musical
figure with a definite harmonic and rhythmic identity) as the smallest
unit of generation, not the token or event. Composition proceeds by
(1) selecting a phrase-form template such as AABA, antecedent–consequent,
or a free chain; (2) filling each slot by scoring every phrase in the
bank against a feature vector derived from the slot role and from the
last-emitted phrase, applying a temperature-softmax, sampling, and
logging the top-3 candidates; (3) rejecting any draw that violates the
harmonic grammar or produces a voice-leading jump > 5 semitones.
The resulting `Composition` object is then serialised to a Format-0
Standard MIDI File with no third-party library. Every step is
deterministic under a fixed random seed.

## Citing ArcaNet

If you use ArcaNet in research, please cite the forthcoming paper. A
machine-readable citation is provided in [`CITATION.cff`](CITATION.cff):

> Paper citation will be updated here once the submission receives a
> decision; DOI will be minted at publication.

## License

MIT. See [LICENSE](LICENSE). The paper text is © the author and is
distributed under a separate agreement with *Computer Music Journal*.
