#!/usr/bin/env python3
# collect_exposure_times.py
# MIRFLICKR EXIF exposure-time collector with debug logging.
# - Handles two-line key/value (e.g., "-Exposure" then "0.008 sec (1/125)")
# - Handles ShutterSpeedValue APEX conversion (t = 2 ** (-APEX))
# - Fraction-first parsing to avoid bogus giant numbers
# - Optional --debug prints exactly what was matched/skipped
# - Filters with --max-sec and --min-sec

import argparse, csv, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# ---- Key/tag patterns ----
KEY_EXPOSURE = re.compile(r'\bexposure(?:\s*time)?\b|exposuretime\b', re.I)
KEY_SHUTTER  = re.compile(r'\bshutter\s*speed(?:\s*value)?\b|shutterspeed(?:value)?\b', re.I)
BAD_EXPOSURE = re.compile(r'\bbias\b|\bcompensation\b|\bprogram\b|\bmode\b', re.I)

# ---- Value patterns ----
RE_FRAC_SECS = re.compile(r'(\d+)\s*/\s*(\d+)\s*(?:s|sec|secs|second|seconds)\b', re.I)
RE_FRAC      = re.compile(r'(\d+)\s*/\s*(\d+)')
RE_SECS_LIT  = re.compile(r'([0-9]*\.?[0-9]+)\s*(?:s|sec|secs|second|seconds)\b', re.I)
RE_NUMBER    = re.compile(r'([0-9]*\.?[0-9]+)')

def _frac_to_float(match_or_text):
    if hasattr(match_or_text, "group"):
        a = float(match_or_text.group(1)); b = float(match_or_text.group(2))
    else:
        m = RE_FRAC.search(match_or_text)
        if not m: return None
        a = float(m.group(1)); b = float(m.group(2))
    if b == 0: return None
    return a / b

def _secs_literal(text):
    m = RE_SECS_LIT.search(text)
    return float(m.group(1)) if m else None

def _number(text):
    m = RE_NUMBER.search(text)
    return float(m.group(1)) if m else None

def _parse_exposure_from_value_line(val_line: str):
    mfs = RE_FRAC_SECS.search(val_line)
    if mfs:
        s = _frac_to_float(mfs)
        if s and s > 0: return s, "exposure_val_fracsecs"
    s = _secs_literal(val_line)
    if s and s > 0: return s, "exposure_val_secs"
    f = _frac_to_float(val_line)
    if f and f > 0: return f, "exposure_val_frac"
    n = _number(val_line)
    if n and n > 0: return n, "exposure_val_num"
    return None, None

def _parse_shutter_from_value_line(val_line: str):
    f = _frac_to_float(val_line)
    if f is not None:
        if f >= 1.5:  # APEX-like
            s = 2.0 ** (-f)
            return s, "shutter_apex_frac"
        else:
            s = f
            return s, "shutter_frac"
    s = _secs_literal(val_line)
    if s and s > 0: return s, "shutter_secs"
    n = _number(val_line)
    if n is not None:
        if n >= 1.5:
            s = 2.0 ** (-n)
            return s, "shutter_apex_num"
        elif n > 0:
            return n, "shutter_num"
    return None, None

def parse_exposure_seconds(lines, debug=False, fname=None):
    n = len(lines)

    # Pass A: two-line key/value
    for i in range(n):
        key = lines[i].strip()
        low = key.lower()
        if BAD_EXPOSURE.search(low):
            if debug:
                print(f"[skip] {fname}: Ignoring key '{key}' (bias/compensation/program)")
            continue

        j = i + 1
        while j < n and lines[j].strip() == "":
            j += 1
        val = lines[j].strip() if j < n else ""

        if KEY_EXPOSURE.search(low):
            s, method = _parse_exposure_from_value_line(val)
            if s:
                if debug:
                    print(f"[ok] {fname}: EXP '{key}' -> '{val}' => {s}s ({method})")
                return s, method, f"{key}\n{val}"
            else:
                if debug:
                    print(f"[miss] {fname}: EXP '{key}' -> '{val}' (no match)")
        if KEY_SHUTTER.search(low):
            s, method = _parse_shutter_from_value_line(val)
            if s:
                if debug:
                    print(f"[ok] {fname}: SHUT '{key}' -> '{val}' => {s}s ({method})")
                return s, method, f"{key}\n{val}"
            else:
                if debug:
                    print(f"[miss] {fname}: SHUT '{key}' -> '{val}' (no match)")

    # Pass B: same-line or XML-ish (rare)
    if debug:
        print(f"[miss] {fname}: No two-line match, checking XML/same-line...")
    whole = "\n".join(lines)
    m_tag = re.search(r'<\s*exposuretime\s*>([^<]+)</\s*exposuretime\s*>', whole, re.I)
    if m_tag:
        raw = m_tag.group(1)
        s = _secs_literal(raw) or _frac_to_float(raw) or _number(raw)
        if s:
            if debug:
                print(f"[ok] {fname}: XML tag '{raw}' => {s}s")
            return s, "xml_tag", raw
    m_attr = re.search(r'\bexposuretime\s*=\s*"([^"]+)"', whole, re.I)
    if m_attr:
        raw = m_attr.group(1)
        s = _secs_literal(raw) or _frac_to_float(raw) or _number(raw)
        if s:
            if debug:
                print(f"[ok] {fname}: XML attr '{raw}' => {s}s")
            return s, "xml_attr", raw

    if debug:
        print(f"[fail] {fname}: No exposure found.")
    return None, None, None

def list_meta_files(root: Path, recursive: bool):
    if recursive:
        yield from (p for p in root.rglob("*") if p.is_file())
    else:
        yield from (p for p in root.iterdir() if p.is_file())

def main():
    ap = argparse.ArgumentParser(description="Collect EXIF exposure times from MIRFLICKR meta/exif directory (debug friendly).")
    ap.add_argument("meta_dir", type=str)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--num", type=int, default=20, help="Top-N longest exposures to print")
    ap.add_argument("--save-csv", type=str, default="exposures.csv")
    ap.add_argument("--save-plot", type=str, default="exposures_hist.png")
    ap.add_argument("--bins", type=int, default=60, help="Number of bins (log-spaced if --log-bins)")
    ap.add_argument("--log-bins", action="store_true", help="Use log-spaced bins for histogram")
    ap.add_argument("--max-sec", type=float, default=60.0, help="Drop exposures > this (0 disables)")
    ap.add_argument("--min-sec", type=float, default=1e-6, help="Drop exposures < this")
    ap.add_argument("--filter", nargs=2, type=float, metavar=("LOW", "HIGH"),
                    help="Filter interval [LOW, HIGH] in seconds; write matching paths to --out-list")
    ap.add_argument("--out-list", type=str, default="filtered_paths.txt",
                    help="Output file for filtered image paths (used with --filter)")
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    root = Path(args.meta_dir)
    if not root.exists() or not root.is_dir():
        raise SystemExit(f"Directory not found: {root}")

    rows = []
    files = list(list_meta_files(root, args.recursive))
    for p in tqdm(files, desc="Scanning"):
        try:
            try:
                text = p.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                text = p.read_text(encoding="latin-1", errors="ignore")
            lines = text.splitlines()
        except Exception:
            continue

        sec, method, src = parse_exposure_seconds(lines, debug=args.debug, fname=p.name)
        if sec is None:
            continue
        if (args.max_sec and args.max_sec > 0 and sec > args.max_sec) or sec < args.min_sec:
            continue
        rows.append({"path": str(p), "exposure_seconds": sec, "method": method, "source": src})

    if not rows:
        print("No exposure times found.")
        return

    # Save CSV
    out_csv = Path(args.save_csv)
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)
    print(f"[+] Wrote {out_csv.resolve()}  ({len(rows)} rows)")

    # Plot distribution
    secs = np.array([r["exposure_seconds"] for r in rows], dtype=np.float64)
    secs = secs[secs > 0]
    if secs.size:
        plt.figure(figsize=(7.5, 4.8))
        if args.log_bins:
            bins = np.logspace(np.log10(secs.min()), np.log10(secs.max()), args.bins)
            plt.hist(secs, bins=bins)
            plt.xscale("log")
        else:
            plt.hist(secs, bins=args.bins)
            plt.xscale("log")
        plt.xlabel("Exposure time (seconds) [log scale]")
        plt.ylabel("Count")
        plt.title(f"Exposure time distribution — {root.name} (N={secs.size})")
        plt.tight_layout()
        out_plot = Path(args.save_plot)
        plt.savefig(out_plot, dpi=150)
        print(f"[+] Saved histogram to {out_plot.resolve()}")
        if args.show: plt.show()

    # Top-N exposures
    rows_sorted = sorted(rows, key=lambda r: r["exposure_seconds"], reverse=True)
    topN = rows_sorted[:args.num]
    print(f"\nTop {len(topN)} exposures (longest first):")
    for r in topN:
        s = r["exposure_seconds"]
        frac = f" (~1/{int(round(1.0/s))})" if 0 < s < 0.5 else ""
        print(f"{s:.6f} s{frac}\t{r['path']}   [{r['method']}]")

    # Filter interval
    if args.filter:
        low, high = args.filter
        matched = [r for r in rows if low <= r["exposure_seconds"] <= high]
        if matched:
            out_list = Path(args.out_list)
            with out_list.open("w") as f:
                for r in matched:
                    f.write(r["path"] + "\n")
            print(f"[+] Wrote {len(matched)} paths in [{low}, {high}]s to {out_list.resolve()}")
        else:
            print(f"[info] No exposures in range [{low}, {high}] seconds.")

if __name__ == "__main__":
    main()
