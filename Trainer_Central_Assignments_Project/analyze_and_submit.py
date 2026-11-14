#!/usr/bin/env python3
"""
Scan .txt files in the current folder, use Claude (Sonnet) to find the file(s)
mentioning a target learner email, and post the extracted assignment info to
the Text Dashboard API (POST /api/entries).

Setup:
    pip install anthropic requests python-dotenv
    Fill in .env (next to this script) with:
        ANTHROPIC_API_KEY=sk-ant-...   (required)
        TRAINER_API_KEY=...            (optional, sent as x-api-key header)

Usage:
    python analyze_and_submit.py
    python analyze_and_submit.py --email someone@example.com
    python analyze_and_submit.py --dry-run
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import requests
from dotenv import load_dotenv

MODEL = "claude-sonnet-5-5"
API_BASE = "https://trainer-central-assigments-api.onrender.com"
ENTRIES_ENDPOINT = f"{API_BASE}/api/entries"

ANALYSIS_SYSTEM_PROMPT = """You analyze email notification text files from TrainerCentral.
Each file is an "assignment submitted" notification containing a learner's email,
an assignment/topic name, and a submission link.

You will be given the target learner email to look for, plus a list of files
(filename + full text content).

Identify every file whose body mentions the target email address. For each
matching file, extract:
  - "sender": the learner's email address (or username portion before '@' if no
    full email is present) — this becomes the EntryIn "sender" field.
  - "text": a concise one or two sentence summary suitable for a dashboard
    entry, e.g. "Learner <email> submitted their assignment for topic '<topic>'
    (cohort: <cohort>, message id: <id>)." Include the assignment topic name
    and submission date if present.

Respond with ONLY a JSON object, no other text, in this exact shape:

{
  "matches": [
    {"filename": "...", "sender": "...", "text": "..."}
  ]
}

If no file matches the target email, respond with {"matches": []}.
"""


def find_txt_files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.glob("*.txt") if p.is_file())


def analyze_files(client: anthropic.Anthropic, target_email: str, files: list[Path]) -> list[dict]:
    files_payload = []
    for f in files:
        try:
            content = f.read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  ! skipping {f.name}: {e}", file=sys.stderr)
            continue
        files_payload.append(f"### FILE: {f.name}\n{content}")

    user_message = (
        f"Target learner email: {target_email}\n\n"
        + "\n\n".join(files_payload)
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=2048,
        system=ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw_text = "".join(block.text for block in response.content if block.type == "text").strip()

    if raw_text.startswith("```"):
        raw_text = raw_text.strip("`")
        if raw_text.startswith("json"):
            raw_text = raw_text[len("json"):].strip()

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError as e:
        print("Claude did not return valid JSON:\n", raw_text, file=sys.stderr)
        raise e

    return parsed.get("matches", [])


def submit_entry(sender: str, text: str, api_key: str | None, dry_run: bool) -> None:
    payload = {"text": text, "sender": sender}
    if dry_run:
        print(f"  [dry-run] would POST {ENTRIES_ENDPOINT} -> {payload}")
        return

    headers = {}
    if api_key:
        headers["x-api-key"] = api_key

    resp = requests.post(ENTRIES_ENDPOINT, json=payload, headers=headers, timeout=30)
    print(f"  POST {ENTRIES_ENDPOINT} -> {resp.status_code}")
    try:
        print(f"  response: {resp.json()}")
    except ValueError:
        print(f"  response: {resp.text}")
    resp.raise_for_status()


def main_lovely() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--email", default="shaliniscr@gmail.com", help="Target learner email to search for")
    parser.add_argument("--folder", default=".", help="Folder containing .txt files (default: current folder)")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be submitted without calling the API")
    args = parser.parse_args()

    load_dotenv(Path(__file__).resolve().parent / ".env")

    anthropic_api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_api_key:
        sys.exit("ANTHROPIC_API_KEY is not set. Add it to the .env file next to this script.")

    trainer_api_key = os.environ.get("TRAINER_API_KEY")

    folder = Path(args.folder).resolve()
    files = find_txt_files(folder)
    if not files:
        sys.exit(f"No .txt files found in {folder}")

    print(f"Scanning {len(files)} .txt file(s) in {folder} for '{args.email}'...")

    client = anthropic.Anthropic(api_key=anthropic_api_key)
    matches = analyze_files(client, args.email, files)

    if not matches:
        print(f"No file found mentioning {args.email}.")
        return

    for match in matches:
        filename = match.get("filename", "<unknown>")
        sender = match.get("sender", "").strip()
        text = match.get("text", "").strip()

        if not sender or not text:
            print(f"  ! incomplete match for {filename}, skipping: {match}", file=sys.stderr)
            continue

        print(f"Match: {filename}")
        print(f"  sender: {sender}")
        print(f"  text:   {text}")
        submit_entry(sender, text, trainer_api_key, args.dry_run)

if __name__ == "__main__":
    main_lovely()