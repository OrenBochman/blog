#!/usr/bin/env python3
import argparse
import datetime
import os
import re
import sys

DRAFT_RE = re.compile(r"^draft\s*:\s*(true|True)\s*$")
DATE_RE = re.compile(r"^date\s*:\s*.*$")
YAML_START_RE = re.compile(r"^---\s*$")
YAML_END_RE = re.compile(r"^---\s*$")

DEFAULT_EXCLUDES = {"docs", "_freeze", ".git", "node_modules"}


def find_qmd_files(root, exclude_dirs=None):
    exclude_dirs = exclude_dirs or DEFAULT_EXCLUDES
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        for filename in filenames:
            if filename.endswith(".qmd"):
                yield os.path.join(dirpath, filename)


def read_front_matter(path):
    with open(path, "r", encoding="utf-8") as fh:
        lines = []
        in_yaml = False
        for line in fh:
            if not in_yaml:
                if YAML_START_RE.match(line):
                    in_yaml = True
                    lines.append(line)
                else:
                    break
            else:
                lines.append(line)
                if YAML_END_RE.match(line):
                    break
    return lines


def file_missing_date(path):
    front_matter = read_front_matter(path)
    if not front_matter:
        return False
    has_draft = any(DRAFT_RE.match(line.strip()) for line in front_matter)
    has_date = any(DATE_RE.match(line.strip()) for line in front_matter)
    return has_draft and not has_date


def insert_date(path, date_value):
    with open(path, "r", encoding="utf-8") as fh:
        lines = fh.readlines()

    yaml_start_idx = None
    yaml_end_idx = None
    for idx, line in enumerate(lines):
        if YAML_START_RE.match(line):
            yaml_start_idx = idx
            break
    if yaml_start_idx is None:
        raise ValueError(f"No YAML front matter found in {path}")

    for idx in range(yaml_start_idx + 1, len(lines)):
        if YAML_END_RE.match(lines[idx]):
            yaml_end_idx = idx
            break
    if yaml_end_idx is None:
        raise ValueError(f"Unterminated YAML front matter in {path}")

    insert_idx = None
    for idx in range(yaml_start_idx + 1, yaml_end_idx):
        if DRAFT_RE.match(lines[idx].strip()):
            insert_idx = idx + 1
            break
    if insert_idx is None:
        insert_idx = yaml_start_idx + 1

    date_line = f"date: {date_value}\n"
    if lines[insert_idx - 1].endswith("\n"):
        lines.insert(insert_idx, date_line)
    else:
        lines.insert(insert_idx, date_line)

    with open(path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find Quarto draft .qmd files missing date metadata and optionally insert a date."
    )
    parser.add_argument(
        "root",
        nargs="?",
        default=".",
        help="Root directory to scan (default: current directory)",
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Insert a date field for missing draft files.",
    )
    parser.add_argument(
        "--date",
        help="Date to insert for missing draft files (default: file modified date in YYYY-MM-DD).",
    )
    parser.add_argument(
        "--exclude",
        nargs="*",
        default=None,
        help="Directories to exclude from scanning.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show files that would be fixed without modifying them.",
    )
    return parser.parse_args()


def format_mtime(path):
    mtime = os.path.getmtime(path)
    return datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    missing = []

    for path in find_qmd_files(root, exclude_dirs=set(args.exclude or DEFAULT_EXCLUDES)):
        try:
            if file_missing_date(path):
                missing.append(path)
        except Exception:
            continue

    if not missing:
        print("No draft .qmd files are missing date metadata.")
        return 0

    print(f"Found {len(missing)} draft file(s) missing date metadata:")
    for path in missing:
        print(f"  {path}")

    if args.fix:
        for path in missing:
            date_value = args.date or format_mtime(path)
            if args.dry_run:
                print(f"Would insert date: {date_value} into {path}")
            else:
                insert_date(path, date_value)
                print(f"Inserted date: {date_value} into {path}")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main())
