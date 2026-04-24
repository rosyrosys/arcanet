"""
ArcaNet: A Phrase-Level Combinatorial Engine for Algorithmic Composition
========================================================================

A minimal, dependency-free reference implementation of the Phrase-Level
Combinatorial AI (PLCA) methodology proposed in:

    "Phrase-Level Combinatorial Generation: A Historically Grounded
     Alternative to End-to-End Generative Music AI" (Computer Music
     Journal, submitted 2026).

The system is a direct descendant of Athanasius Kircher's Arca
Musarithmica (1650), re-cast with a neural-style phrase selector on top
of a curated phrase bank and a hard constraint layer.

Design goals
------------
1. The smallest compositional unit is a **phrase** (one or two bars),
   not a token.  Every candidate phrase is fully formed musical material
   with an explicit harmonic and rhythmic identity.
2. All selection decisions are inspectable, reversible, and editable.
3. The system is written in pure-Python stdlib so that it can be cloned,
   read, and modified by any reader of the journal in one afternoon.

Run:
    python arcanet.py

Output:
    output/arcanet_sample_01.mid   (16-bar A-A'-B-A' form)
    output/arcanet_sample_02.mid   (8-bar antecedent-consequent)
    output/arcanet_sample_03.mid   (12-bar free combination)
    output/arcanet_log.txt         (selection trace for each sample)
    output/arcanet_stats.json      (phrase-reuse, constraint-hit rates)
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
from dataclasses import dataclass, field, asdict
from typing import Callable, Iterable

# ---------------------------------------------------------------------------
# 1.  Musical primitives
# ---------------------------------------------------------------------------

PITCH_NAMES = ["C", "C#", "D", "D#", "E", "F",
               "F#", "G", "G#", "A", "A#", "B"]

MAJOR_SCALE = [0, 2, 4, 5, 7, 9, 11]   # C major degrees
# Triads for I, ii, iii, IV, V, vi in C major
TRIADS_C = {
    "I":  [60, 64, 67],
    "ii": [62, 65, 69],
    "iii":[64, 67, 71],
    "IV": [65, 69, 72],
    "V":  [67, 71, 74],
    "vi": [69, 72, 76],
    "V/V":[62, 66, 69],   # secondary dominant
}

# Harmonic function taxonomy used by the grammar
FUNCTION = {
    "I":   "T",   # tonic
    "vi":  "T",
    "iii": "T",
    "ii":  "PD",  # pre-dominant
    "IV":  "PD",
    "V/V": "PD",
    "V":   "D",   # dominant
}

# Allowed successions inside a phrase group (standard common-practice grammar)
FUNCTION_TRANSITIONS = {
    "T":  ["T", "PD", "D"],
    "PD": ["PD", "D"],
    "D":  ["T", "D"],
}


# ---------------------------------------------------------------------------
# 2.  Phrase object
# ---------------------------------------------------------------------------

@dataclass
class Phrase:
    """A single musical unit: one or two bars, completely composed."""

    pid: str
    # [(pitch_midi, onset_beats, duration_beats, velocity)]
    notes: list[tuple[int, float, float, int]]
    bars: int
    beats_per_bar: int
    chord: str                 # e.g. "I", "V", "IV"
    function: str              # "T" / "PD" / "D"
    role: str                  # "open", "mid", "cad", "final"
    density: float             # notes-per-beat
    embedding: list[float] = field(default_factory=list)

    # --- derived helpers ---------------------------------------------------

    @property
    def total_beats(self) -> float:
        return self.bars * self.beats_per_bar

    @property
    def first_pitch(self) -> int:
        return self.notes[0][0]

    @property
    def last_pitch(self) -> int:
        return self.notes[-1][0]

    def transpose(self, semitones: int) -> "Phrase":
        tnotes = [(p + semitones, o, d, v) for (p, o, d, v) in self.notes]
        return Phrase(
            pid=f"{self.pid}_t{semitones:+d}",
            notes=tnotes, bars=self.bars, beats_per_bar=self.beats_per_bar,
            chord=self.chord, function=self.function, role=self.role,
            density=self.density, embedding=self.embedding,
        )

    def pitch_class_profile(self) -> list[float]:
        """Duration-weighted PCP in R^12."""
        pcp = [0.0] * 12
        for (p, _o, d, _v) in self.notes:
            pcp[p % 12] += d
        total = sum(pcp) or 1.0
        return [x / total for x in pcp]


# ---------------------------------------------------------------------------
# 3.  Phrase bank
# ---------------------------------------------------------------------------
#
# Each phrase is entered by hand (two bars of 4/4, C-major) in the spirit
# of Kircher's printed numerical tables.  An AI system in deployment
# would harvest these from a licensed corpus, but for the reference
# implementation we use 18 curated phrases covering the basic functions,
# plus two cadential finals.

def _mkphrase(pid, chord, role, notes, bars=2, bpb=4):
    func = FUNCTION[chord]
    density = len(notes) / (bars * bpb)
    return Phrase(pid=pid, notes=notes, bars=bars, beats_per_bar=bpb,
                  chord=chord, function=func, role=role, density=density)


BANK: list[Phrase] = [
    # ------------- Tonic openings -------------
    _mkphrase("T_open_01", "I", "open",
        [(60, 0.0, 1.0, 80), (64, 1.0, 1.0, 78), (67, 2.0, 1.0, 82),
         (64, 3.0, 1.0, 78), (60, 4.0, 2.0, 84), (67, 6.0, 2.0, 80)]),
    _mkphrase("T_open_02", "I", "open",
        [(60, 0.0, 0.5, 80), (62, 0.5, 0.5, 76), (64, 1.0, 1.0, 82),
         (67, 2.0, 1.0, 80), (72, 3.0, 1.0, 85), (71, 4.0, 1.0, 78),
         (69, 5.0, 1.0, 76), (67, 6.0, 2.0, 80)]),
    _mkphrase("T_open_03", "I", "open",
        [(67, 0.0, 1.0, 78), (64, 1.0, 1.0, 76), (60, 2.0, 2.0, 82),
         (64, 4.0, 1.0, 78), (67, 5.0, 1.0, 80), (72, 6.0, 2.0, 84)]),
    _mkphrase("T_open_04", "vi", "open",
        [(57, 0.0, 1.0, 76), (60, 1.0, 1.0, 78), (64, 2.0, 2.0, 80),
         (69, 4.0, 1.0, 80), (67, 5.0, 1.0, 76), (64, 6.0, 2.0, 78)]),

    # ------------- Pre-dominant middles -------------
    _mkphrase("PD_mid_01", "IV", "mid",
        [(65, 0.0, 1.0, 78), (69, 1.0, 1.0, 78), (72, 2.0, 1.0, 82),
         (69, 3.0, 1.0, 78), (65, 4.0, 2.0, 80), (74, 6.0, 2.0, 82)]),
    _mkphrase("PD_mid_02", "ii", "mid",
        [(62, 0.0, 1.0, 76), (65, 1.0, 1.0, 78), (69, 2.0, 2.0, 80),
         (65, 4.0, 1.0, 76), (62, 5.0, 1.0, 74), (69, 6.0, 2.0, 80)]),
    _mkphrase("PD_mid_03", "IV", "mid",
        [(72, 0.0, 0.5, 82), (71, 0.5, 0.5, 78), (69, 1.0, 1.0, 80),
         (65, 2.0, 1.0, 78), (69, 3.0, 1.0, 78), (72, 4.0, 2.0, 82),
         (74, 6.0, 1.0, 80), (72, 7.0, 1.0, 80)]),
    _mkphrase("PD_mid_04", "V/V", "mid",
        [(62, 0.0, 1.0, 76), (66, 1.0, 1.0, 78), (69, 2.0, 2.0, 80),
         (74, 4.0, 2.0, 82), (72, 6.0, 2.0, 80)]),

    # ------------- Dominant approaches -------------
    _mkphrase("D_mid_01", "V", "mid",
        [(67, 0.0, 1.0, 82), (71, 1.0, 1.0, 80), (74, 2.0, 2.0, 84),
         (72, 4.0, 1.0, 82), (71, 5.0, 1.0, 80), (67, 6.0, 2.0, 82)]),
    _mkphrase("D_mid_02", "V", "mid",
        [(71, 0.0, 0.5, 80), (74, 0.5, 0.5, 82), (77, 1.0, 1.0, 84),
         (74, 2.0, 1.0, 82), (71, 3.0, 1.0, 80), (67, 4.0, 2.0, 82),
         (74, 6.0, 2.0, 84)]),

    # ------------- Cadential phrases -------------
    _mkphrase("cad_auth_01", "V", "cad",
        [(67, 0.0, 1.0, 82), (71, 1.0, 1.0, 84), (74, 2.0, 1.0, 86),
         (72, 3.0, 1.0, 80), (71, 4.0, 1.0, 78), (67, 5.0, 3.0, 84)]),
    _mkphrase("cad_auth_02", "V", "cad",
        [(74, 0.0, 1.0, 84), (72, 1.0, 1.0, 80), (71, 2.0, 1.0, 82),
         (72, 3.0, 1.0, 80), (74, 4.0, 1.0, 82), (71, 5.0, 1.0, 78),
         (67, 6.0, 2.0, 84)]),

    # ------------- Tonic finals -------------
    _mkphrase("T_final_01", "I", "final",
        [(64, 0.0, 1.0, 78), (67, 1.0, 1.0, 80), (72, 2.0, 2.0, 86),
         (71, 4.0, 1.0, 78), (69, 5.0, 1.0, 76), (67, 6.0, 0.5, 74),
         (64, 6.5, 0.5, 72), (60, 7.0, 1.0, 90)]),
    _mkphrase("T_final_02", "I", "final",
        [(67, 0.0, 1.0, 78), (64, 1.0, 1.0, 76), (60, 2.0, 2.0, 82),
         (67, 4.0, 2.0, 78), (72, 6.0, 1.0, 80), (60, 7.0, 1.0, 90)]),

    # ------------- Additional tonic middles -------------
    _mkphrase("T_mid_01", "I", "mid",
        [(72, 0.0, 1.0, 82), (71, 1.0, 1.0, 78), (69, 2.0, 1.0, 78),
         (67, 3.0, 1.0, 76), (64, 4.0, 2.0, 78), (67, 6.0, 2.0, 80)]),
    _mkphrase("T_mid_02", "vi", "mid",
        [(69, 0.0, 1.0, 78), (72, 1.0, 1.0, 80), (76, 2.0, 2.0, 82),
         (72, 4.0, 1.0, 78), (69, 5.0, 1.0, 76), (65, 6.0, 2.0, 78)]),

    # ------------- Extra dominant motion -------------
    _mkphrase("D_mid_03", "V", "mid",
        [(62, 0.0, 1.0, 76), (67, 1.0, 1.0, 80), (71, 2.0, 1.0, 82),
         (74, 3.0, 1.0, 84), (74, 4.0, 2.0, 82), (67, 6.0, 2.0, 80)]),

    # ------------- Half-cadence -------------
    _mkphrase("cad_half_01", "V", "cad",
        [(65, 0.0, 1.0, 78), (64, 1.0, 1.0, 76), (62, 2.0, 1.0, 78),
         (67, 3.0, 1.0, 80), (71, 4.0, 1.0, 80), (74, 5.0, 1.0, 82),
         (74, 6.0, 2.0, 84)]),
]


# ---------------------------------------------------------------------------
# 4.  Embeddings
# ---------------------------------------------------------------------------
#
# A real system would use a pre-trained symbolic-music encoder (e.g. a
# small transformer over MIDI).  For the reference implementation we use
# a deterministic hand-crafted embedding that concatenates the 12-dim
# pitch-class profile with a small set of musical descriptors.  This is
# enough to demonstrate that *semantic* nearest-neighbour retrieval, not
# autoregressive token prediction, drives generation.

def embed(phrase: Phrase) -> list[float]:
    pcp = phrase.pitch_class_profile()
    role_onehot = {"open":[1,0,0,0], "mid":[0,1,0,0],
                   "cad":[0,0,1,0], "final":[0,0,0,1]}[phrase.role]
    func_onehot = {"T":[1,0,0], "PD":[0,1,0], "D":[0,0,1]}[phrase.function]
    # Melodic contour summary
    pitches = [n[0] for n in phrase.notes]
    contour = [pitches[-1] - pitches[0],
               max(pitches) - min(pitches),
               sum(abs(pitches[i+1]-pitches[i]) for i in range(len(pitches)-1))
               / max(1, len(pitches)-1)]
    dens = [phrase.density]
    vec = pcp + role_onehot + func_onehot + contour + dens
    # l2 normalise
    n = math.sqrt(sum(x*x for x in vec)) or 1.0
    return [x / n for x in vec]


for p in BANK:
    p.embedding = embed(p)


def cosine(u: list[float], v: list[float]) -> float:
    return sum(a*b for a, b in zip(u, v))


# ---------------------------------------------------------------------------
# 5.  Selector:  constraint-gated neural-style retrieval
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    """Hard filter expressed as a predicate over (previous, candidate)."""
    max_voice_leading: int = 9           # semitones between phrase endpoints
    allow_function: tuple[str, ...] = ("T","PD","D")
    allow_role: tuple[str, ...] = ("open","mid","cad","final")


def grammar_ok(prev: Phrase | None, cand: Phrase) -> bool:
    if prev is None:
        return cand.role == "open"
    return cand.function in FUNCTION_TRANSITIONS[prev.function]


def voice_leading_ok(prev: Phrase | None, cand: Phrase, k: int) -> bool:
    if prev is None:
        return True
    return abs(prev.last_pitch - cand.first_pitch) <= k


def feasible(prev: Phrase | None, cand: Phrase, cons: Constraints) -> bool:
    return (cand.function in cons.allow_function
            and cand.role in cons.allow_role
            and grammar_ok(prev, cand)
            and voice_leading_ok(prev, cand, cons.max_voice_leading))


def score(cand: Phrase, context_embed: list[float], prev: Phrase | None,
          temperature: float = 0.7) -> float:
    """Soft score combining semantic match, voice-leading smoothness,
    and an anti-repetition penalty."""
    sem = cosine(cand.embedding, context_embed)           # in [-1, 1]
    vl = 0.0 if prev is None else -abs(prev.last_pitch - cand.first_pitch) / 12.0
    rep = -1.0 if (prev is not None and cand.pid == prev.pid) else 0.0
    return (1.2 * sem + 0.6 * vl + 0.8 * rep) / max(temperature, 1e-3)


def softmax(xs: list[float]) -> list[float]:
    m = max(xs)
    exps = [math.exp(x - m) for x in xs]
    s = sum(exps)
    return [e / s for e in exps]


def select_phrase(context_embed: list[float], prev: Phrase | None,
                  cons: Constraints, rng: random.Random,
                  temperature: float = 0.7) -> tuple[Phrase, list[tuple[str,float]]]:
    cands = [p for p in BANK if feasible(prev, p, cons)]
    if not cands:
        # Relax grammar first, then voice-leading, rather than silently failing.
        cands = [p for p in BANK
                 if p.function in cons.allow_function
                 and p.role in cons.allow_role]
    raw = [score(p, context_embed, prev, temperature) for p in cands]
    probs = softmax(raw)
    r = rng.random()
    acc = 0.0
    trace = list(zip([p.pid for p in cands], probs))
    for p, pr in zip(cands, probs):
        acc += pr
        if r <= acc:
            return p, trace
    return cands[-1], trace


# ---------------------------------------------------------------------------
# 6.  Form templates (the composer's high-level blueprint)
# ---------------------------------------------------------------------------

FORMS = {
    # 16-bar period with A-A'-B-A' small ternary feel
    "AABA_16":  ["open", "mid", "mid", "cad", "open", "mid", "mid", "final"],
    # 8-bar antecedent + consequent (Schoenberg's sentence)
    "sentence_8": ["open", "mid", "cad", "final"],
    # 14-bar free combinatorial form (A-B-B'-cad-A'-B-final)
    "free_14":  ["open", "mid", "mid", "cad", "open", "mid", "final"],
}

FORM_BARS = {"AABA_16": 16, "sentence_8": 8, "free_14": 14}  # each slot = 2 bars

# Target embeddings per formal role (centroid of suitable phrases)
def role_centroid(role: str) -> list[float]:
    subs = [p for p in BANK if p.role == role]
    if not subs:
        subs = BANK
    dim = len(subs[0].embedding)
    cen = [0.0] * dim
    for p in subs:
        for i, x in enumerate(p.embedding):
            cen[i] += x
    cen = [x / len(subs) for x in cen]
    n = math.sqrt(sum(x*x for x in cen)) or 1.0
    return [x / n for x in cen]


# ---------------------------------------------------------------------------
# 7.  Composition routine
# ---------------------------------------------------------------------------

@dataclass
class Composition:
    phrases: list[Phrase]
    selection_trace: list[list[tuple[str, float]]]
    form: str

    @property
    def total_beats(self) -> float:
        return sum(p.total_beats for p in self.phrases)


def compose(form: str, cons: Constraints, rng: random.Random,
            temperature: float = 0.7) -> Composition:
    plan = FORMS[form]
    phrases: list[Phrase] = []
    traces: list[list[tuple[str,float]]] = []
    for i, role in enumerate(plan):
        # Role-specific hard constraints override
        local_cons = Constraints(
            max_voice_leading=cons.max_voice_leading,
            allow_function=("T","PD","D") if role != "cad" else ("D",),
            allow_role=(role,),
        )
        ctx = role_centroid(role)
        prev = phrases[-1] if phrases else None
        p, trace = select_phrase(ctx, prev, local_cons, rng, temperature)
        phrases.append(p)
        traces.append(trace)
    return Composition(phrases=phrases, selection_trace=traces, form=form)


# ---------------------------------------------------------------------------
# 8.  MIDI export (Standard MIDI File, format 0, single track)
# ---------------------------------------------------------------------------

def _vlq(value: int) -> bytes:
    """Variable-length quantity encoding used throughout the MIDI spec."""
    if value == 0:
        return b"\x00"
    buf = []
    buf.append(value & 0x7F)
    value >>= 7
    while value:
        buf.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(buf))


def composition_to_midi(comp: Composition, path: str,
                        tempo_bpm: int = 92, ppq: int = 480) -> None:
    """Serialise a Composition to a Format-0 Standard MIDI File."""
    events: list[tuple[int, bytes]] = []   # (absolute_ticks, event)

    # Tempo meta event at tick 0
    us_per_qn = int(60_000_000 / tempo_bpm)
    tempo_evt = b"\xFF\x51\x03" + us_per_qn.to_bytes(3, "big")
    events.append((0, tempo_evt))
    # Time signature 4/4
    events.append((0, b"\xFF\x58\x04\x04\x02\x18\x08"))

    offset_beats = 0.0
    for ph in comp.phrases:
        for (pitch, onset, dur, vel) in ph.notes:
            abs_on  = int(round((offset_beats + onset)           * ppq))
            abs_off = int(round((offset_beats + onset + dur)     * ppq))
            events.append((abs_on,  bytes([0x90, pitch, vel])))
            events.append((abs_off, bytes([0x80, pitch, 0])))
        offset_beats += ph.total_beats

    # Build delta-time track
    events.sort(key=lambda e: e[0])
    track_bytes = bytearray()
    last_tick = 0
    for tick, data in events:
        track_bytes += _vlq(tick - last_tick)
        track_bytes += data
        last_tick = tick
    # End of track
    track_bytes += _vlq(0) + b"\xFF\x2F\x00"

    # Header chunk (format 0, 1 track, ppq division)
    header = b"MThd" + (6).to_bytes(4, "big") + \
             (0).to_bytes(2, "big") + (1).to_bytes(2, "big") + \
             ppq.to_bytes(2, "big")
    track  = b"MTrk" + len(track_bytes).to_bytes(4, "big") + bytes(track_bytes)

    with open(path, "wb") as fh:
        fh.write(header + track)


# ---------------------------------------------------------------------------
# 9.  Driver
# ---------------------------------------------------------------------------

def write_log(comp: Composition, fh) -> None:
    fh.write(f"FORM: {comp.form}\n")
    fh.write(f"BARS: {int(comp.total_beats // 4)}\n")
    fh.write(f"PHRASES ({len(comp.phrases)}):\n")
    for i, (p, trace) in enumerate(zip(comp.phrases, comp.selection_trace)):
        fh.write(f"  [{i:02d}] pid={p.pid:<14s} "
                 f"chord={p.chord:<4s} fn={p.function:<3s} "
                 f"role={p.role:<5s} dens={p.density:.2f} "
                 f"first={p.first_pitch:>3d} last={p.last_pitch:>3d}\n")
        top = sorted(trace, key=lambda x: -x[1])[:3]
        fh.write(f"        top-3 candidates: "
                 f"{[(pid, round(pr,3)) for pid,pr in top]}\n")
    fh.write("\n")


def stats(comps: list[Composition]) -> dict:
    all_pids = [p.pid for c in comps for p in c.phrases]
    unique = set(all_pids)
    reuse = 1.0 - (len(unique) / max(1, len(all_pids)))
    # Constraint-hit rates (fraction of candidate pools that obeyed grammar)
    grammar_hits = 0
    total = 0
    for c in comps:
        prev = None
        for p in c.phrases:
            total += 1
            if prev is None or p.function in FUNCTION_TRANSITIONS[prev.function]:
                grammar_hits += 1
            prev = p
    # Average voice-leading jump
    jumps = []
    for c in comps:
        for i in range(1, len(c.phrases)):
            jumps.append(abs(c.phrases[i-1].last_pitch
                             - c.phrases[i].first_pitch))
    avg_jump = sum(jumps) / max(1, len(jumps))
    return {
        "compositions": len(comps),
        "phrases_emitted": len(all_pids),
        "distinct_phrases_used": len(unique),
        "phrase_reuse_rate": round(reuse, 3),
        "grammar_satisfaction_rate": round(grammar_hits / max(1,total), 3),
        "avg_voice_leading_semitones": round(avg_jump, 2),
        "phrase_bank_size": len(BANK),
    }


def main() -> None:
    rng = random.Random(2026)
    cons = Constraints(max_voice_leading=9)
    here = os.path.dirname(os.path.abspath(__file__))
    out  = os.path.join(here, "output")
    os.makedirs(out, exist_ok=True)

    samples = [
        ("AABA_16",  "arcanet_sample_01.mid"),
        ("sentence_8","arcanet_sample_02.mid"),
        ("free_14",  "arcanet_sample_03.mid"),
    ]
    comps = []
    with open(os.path.join(out, "arcanet_log.txt"), "w") as log:
        log.write("ArcaNet selection log\n" + "="*40 + "\n\n")
        for form, fname in samples:
            c = compose(form, cons, rng, temperature=0.6)
            comps.append(c)
            composition_to_midi(c, os.path.join(out, fname), tempo_bpm=96)
            write_log(c, log)

    s = stats(comps)
    with open(os.path.join(out, "arcanet_stats.json"), "w") as fh:
        json.dump(s, fh, indent=2)

    print("Generated MIDI:")
    for _, fname in samples:
        print(f"  {os.path.join(out, fname)}")
    print("\nStatistics:")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
