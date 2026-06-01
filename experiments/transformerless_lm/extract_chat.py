#!/usr/bin/env python3
"""extract_chat.py — extract the SUBSTANTIVE dialogue from the Claude Code session transcripts into
library/ so it can be folded: the web learns its own GENESIS (the decisions/reasoning that built it),
the meta-layer above the folded OMC source.

The raw .jsonl is mostly tool machinery (huge); we keep only the conversation: user's typed messages +
the assistant's prose (text blocks). DROPPED: tool_use / tool_result blocks, internal `thinking`, and the
command/system wrappers (<local-command-*>, <command-*>, <system-reminder>, interrupt notices). Writes one
library/chathistory__<sid>.txt per session. READ-only on transcripts; writes only to library/ (safe to run
while a fold writes knowledge.db). The actual FOLD of these files is a separate step (after the science fold).
"""
import json, re, sys
from pathlib import Path

PROJ = Path("/home/thearchitect/.claude/projects/-home-thearchitect")
LIB = Path("/home/thearchitect/OMC/experiments/transformerless_lm/library")
WRAP = re.compile(r"<(local-command-caveat|command-name|command-message|command-args|local-command-stdout"
                  r"|system-reminder|user-prompt-submit-hook|command-stdout)[\s\S]*?>", re.I)
NOISE_PREFIX = ("<local-command", "<command-", "[Request interrupted", "Caveat:", "<system-reminder")


def clean_user(text):
    """Strip command/reminder wrappers; keep real typed prose."""
    t = WRAP.sub(" ", text)
    t = re.sub(r"</?[a-z-]+>", " ", t)            # stray tags
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) < 8:
        return ""
    # drop messages that were purely wrapper/noise
    low = t.lower()
    if low.startswith(("caveat", "the messages below", "your questions have been answered")):
        return ""
    return t


def extract(path):
    out = []
    for line in open(path):
        try:
            e = json.loads(line)
        except Exception:
            continue
        m = e.get("message", {})
        role = m.get("role")
        if role not in ("user", "assistant"):
            continue
        c = m.get("content")
        if isinstance(c, str):
            c = [{"type": "text", "text": c}]
        if not isinstance(c, list):
            continue
        for b in c:
            if not isinstance(b, dict):
                continue
            typ = b.get("type")
            if typ == "text":
                txt = b.get("text", "")
                if role == "user":
                    t = clean_user(txt)
                    if t:
                        out.append(("USER", t))
                else:
                    t = re.sub(r"\s+", " ", txt).strip()
                    if len(t) >= 8:
                        out.append(("LM", t))
            elif typ == "tool_use":   # the ACTION — compact summary of what was done (user: "even the tool calls may be useful")
                name = b.get("name", "?"); inp = b.get("input", {}) or {}
                if name == "Bash":
                    arg = inp.get("command", "")
                elif name in ("Write", "Read"):
                    arg = inp.get("file_path", "")
                elif name == "Edit":
                    arg = f"{inp.get('file_path','')}: {inp.get('old_string','')[:60]} -> {inp.get('new_string','')[:60]}"
                else:
                    arg = json.dumps(inp)[:200]
                arg = re.sub(r"\s+", " ", str(arg)).strip()[:400]
                if arg:
                    out.append(("ACTION", f"{name}: {arg}"))
            elif typ == "tool_result":   # the OUTCOME — TRUNCATED (raw results are 360MB of noise; keep the gist)
                rc = b.get("content", "")
                if isinstance(rc, list):
                    rc = " ".join(x.get("text", "") for x in rc if isinstance(x, dict))
                rc = re.sub(r"\s+", " ", str(rc)).strip()
                if len(rc) >= 12:
                    out.append(("RESULT", rc[:300]))
    return out


def main():
    LIB.mkdir(exist_ok=True)
    total_turns = total_files = 0
    for jf in sorted(PROJ.glob("*.jsonl")):
        turns = extract(jf)
        if len(turns) < 4:
            continue
        sid = jf.stem.split("-")[0]
        text = "\n\n".join(f"{who}: {t}" for who, t in turns)
        outp = LIB / f"chathistory__{sid}.txt"
        outp.write_text(text, encoding="utf-8")
        total_turns += len(turns); total_files += 1
        print(f"  {jf.name[:18]}… → {len(turns):>4} turns, {len(text)//1024}KB")
    print(f"\n[extract_chat] {total_files} session files, {total_turns:,} dialogue turns → library/chathistory__*.txt")
    print("[extract_chat] NEXT: fold these (after the science fold) so the web knows its own genesis.")


if __name__ == "__main__":
    main()
