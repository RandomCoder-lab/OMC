"""
Chained generation: 6 windows × 89 tokens = ~534 chars.
Each window's tail feeds the next as context.
Compare to actual seed passage.
"""
import sys
import re
import torch
from pathlib import Path
from address_stage3 import load_navigator

SEED_PASSAGE = (
    "SLY:\nAy, it stands so that I may hardly\ntarry so long. But I would be loath to fall into\n"
    "my dreams again: I will therefore tarry in\ndespite of the flesh and the blood.\n\n"
    "Messenger:\nYour honour's players, heating your amendment,\n"
    "Are come to play a pleasant comedy;\nFor so your doctors hold it very meet,\n"
    "Seeing too much sadness hath congeal'd your blood,\n"
    "And melancholy is the nurse of frenzy:\n"
    "Therefore they thought it good you hear a play\n"
    "And frame your mind to mirth and merriment,\nWhich bars a thousand harms a"
)

N_WINDOWS   = 6
WINDOW_SIZE = 89
OVERLAP     = 20       # chars of tail fed as next-window prompt
TEMPERATURE = 0.75


def real_word_fraction(text, registry):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    long_words = [w for w in words if len(w) >= 3]
    if not long_words:
        return 0.0
    return sum(1 for w in long_words if w in registry) / len(long_words)


def trigram_overlap(generated, reference):
    def tgs(s):
        return {s[i:i+3] for i in range(len(s)-2)}
    gen_tg  = tgs(generated)
    ref_tg  = tgs(reference)
    if not gen_tg:
        return 0.0
    return len(gen_tg & ref_tg) / len(gen_tg)


def run_chain(nav, start_prompt="SLY:\nAy, it stands so"):
    print(f"\n{'='*65}")
    print("CHAINED GENERATION  —  {N_WINDOWS}× {WINDOW_SIZE}-token windows")
    print(f"{'='*65}")
    print(f"Seed prompt: {repr(start_prompt)}\n")

    full_text = start_prompt
    prompt    = start_prompt

    for i in range(N_WINDOWS):
        gen, _ = nav.navigate_generate(
            n_tokens=WINDOW_SIZE,
            prompt_text=prompt,
            temperature=TEMPERATURE,
        )
        window_text = gen[len(prompt):]   # only the new characters
        full_text  += window_text
        print(f"── Window {i+1}/{N_WINDOWS} ──────────────────────────────────")
        print(repr(window_text))
        prompt = full_text[-OVERLAP:]     # tail as next context

    return full_text[len(start_prompt):]  # generated portion only


def score_and_compare(generated, registry):
    print(f"\n{'='*65}")
    print("SCORING")
    print(f"{'='*65}")

    rwf = real_word_fraction(generated, registry)
    tgo = trigram_overlap(generated, SEED_PASSAGE)

    print(f"Generated ({len(generated)} chars):")
    print(repr(generated[:300]))
    print()
    print(f"Reference ({len(SEED_PASSAGE)} chars):")
    print(repr(SEED_PASSAGE[:300]))
    print()
    print(f"Real-word fraction (≥3-char words in registry) : {rwf:.3f}")
    print(f"3-gram overlap with seed passage               : {tgo:.3f}")

    # Check for verbatim seed fragments
    print("\nVerbatim seed fragments in generated output (≥8 chars):")
    hits = []
    for start in range(0, len(SEED_PASSAGE) - 8):
        fragment = SEED_PASSAGE[start:start+8]
        if fragment in generated:
            hits.append(fragment)
    if hits:
        # Deduplicate by longest non-overlapping
        seen = set()
        for h in sorted(set(hits), key=len, reverse=True):
            if not any(h in s for s in seen):
                seen.add(h)
        for h in sorted(seen):
            print(f"  {repr(h)}")
    else:
        print("  (none found)")

    return rwf, tgo


if __name__ == "__main__":
    cp = Path(__file__).parent / "bloom_checkpoint.pt"
    mp = Path(__file__).parent / "bloom_checkpoint_model.pt"

    print(f"Loading navigator from {cp} ...")
    nav = load_navigator(str(cp))

    registry = nav.word_registry if hasattr(nav, 'word_registry') else {}

    generated = run_chain(nav, start_prompt="SLY:\nAy, it stands so")
    rwf, tgo  = score_and_compare(generated, registry)

    print(f"\n{'='*65}")
    print(f"SUMMARY")
    print(f"{'='*65}")
    print(f"Total generated: {len(generated)} chars across {N_WINDOWS} windows")
    print(f"Real-word fraction : {rwf:.3f}  {'★ GOOD' if rwf > 0.15 else '△ LOW'}")
    print(f"Seed 3-gram overlap: {tgo:.3f}  {'★ STRONG' if tgo > 0.20 else '△ WEAK' if tgo < 0.05 else '~ MODERATE'}")
