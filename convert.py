#!/usr/bin/env python3
"""
WAV → VAG and VAG → MP3 converter for PS1 audio.

WAV → VAG  (default):
    python3 convert.py                          # all WAVs from input/ → output/
    python3 convert.py kick.wav                 # specific file
    python3 convert.py -r 44100 -l             # 44 kHz, loop entire sample
    python3 convert.py -l --ls 5600 --le 11200 # loop from sample 5600 to 11200
                                                # (both auto-aligned to multiples of 28)

VAG → MP3  (--decode):
    python3 convert.py --decode         # all VAGs from output/ → output_mp3/
    python3 convert.py --decode output/kick.vag
"""

import argparse
import shutil
import struct
import subprocess
import sys
from pathlib import Path

try:
    import numpy as np
except ImportError:
    sys.exit("numpy is required: pip install numpy")

SCRIPT_DIR  = Path(__file__).parent
INPUT_DIR   = SCRIPT_DIR / "input"
OUTPUT_DIR  = SCRIPT_DIR / "output"
MP3_DIR     = SCRIPT_DIR / "output_mp3"

SAMPLES_PER_BLOCK = 28

# Prediction filter coefficients (f0, f1)
_COEFF = [
    ( 0.0,       0.0      ),
    ( 0.9375,    0.0      ),
    ( 1.796875, -0.8125   ),
    ( 1.53125,  -0.859375 ),
    ( 1.90625,  -0.9375   ),
]

# Precomputed arrays for all 65 (filter × shift) combos — used during encoding
_FI    = np.repeat(np.arange(5), 13)
_SHIFT = np.tile(np.arange(13), 5)
_F0    = np.array([_COEFF[f][0] for f in _FI], dtype=np.float64)
_F1    = np.array([_COEFF[f][1] for f in _FI], dtype=np.float64)
_SCALE = (1 << (12 - _SHIFT)).astype(np.float64)

# ─── ENCODING (WAV → VAG) ────────────────────────────────────────────────────

def _encode_block(samples: np.ndarray, h1: float, h2: float, flags: int) -> tuple[bytes, float, float]:
    p1 = np.full(65, h1)
    p2 = np.full(65, h2)
    nibbles = np.empty((65, SAMPLES_PER_BLOCK), dtype=np.int32)
    err     = np.zeros(65)

    for j in range(SAMPLES_PER_BLOCK):
        predicted = _F0 * p1 + _F1 * p2
        residual  = samples[j] - predicted
        nib       = np.clip(np.round(residual / _SCALE), -8, 7).astype(np.int32)
        decoded   = np.clip(predicted + nib * _SCALE, -32768.0, 32767.0)
        err      += (samples[j] - decoded) ** 2
        nibbles[:, j] = nib
        p2 = p1.copy()
        p1 = decoded

    best  = int(np.argmin(err))
    fi    = int(_FI[best])
    shift = int(_SHIFT[best])
    nibs  = nibbles[best]

    buf = bytearray(16)
    buf[0] = (fi << 4) | shift
    buf[1] = flags
    for i in range(14):
        lo = int(nibs[i * 2])      & 0xF
        hi = int(nibs[i * 2 + 1]) & 0xF
        buf[2 + i] = (hi << 4) | lo

    return bytes(buf), float(p1[best]), float(p2[best])


def _align28(sample: int) -> int:
    """Floor a sample position to the nearest multiple of 28 (SPU requirement)."""
    return (sample // SAMPLES_PER_BLOCK) * SAMPLES_PER_BLOCK


def _pcm_to_vag(pcm: np.ndarray, sample_rate: int, name: str,
                loop: bool, loop_start: int = 0, loop_end: int = -1) -> bytes:
    # Pad to a multiple of 28
    rem = len(pcm) % SAMPLES_PER_BLOCK
    if rem:
        pcm = np.concatenate([pcm, np.zeros(SAMPLES_PER_BLOCK - rem, dtype=np.int16)])

    n_blocks = len(pcm) // SAMPLES_PER_BLOCK

    # Align loop points to multiples of 28 as the SPU requires.
    # transcript: floor(value / 28) * 28 for both start and end.
    if loop:
        ls_block = _align28(max(0, loop_start)) // SAMPLES_PER_BLOCK
        le_sample = loop_end if loop_end >= 0 else len(pcm) - SAMPLES_PER_BLOCK
        le_block  = _align28(min(le_sample, len(pcm) - SAMPLES_PER_BLOCK)) // SAMPLES_PER_BLOCK
        le_block  = max(ls_block, min(le_block, n_blocks - 1))
        ls_block  = min(ls_block, le_block)

        aligned_ls = ls_block * SAMPLES_PER_BLOCK
        aligned_le = le_block * SAMPLES_PER_BLOCK
        loop_len   = aligned_le - aligned_ls
        print(f"    loop start : sample {aligned_ls}  (block {ls_block})")
        print(f"    loop end   : sample {aligned_le}  (block {le_block})")
        print(f"    loop length: {loop_len} samples  ({loop_len // SAMPLES_PER_BLOCK} blocks, "
              f"÷28 remainder = {loop_len % SAMPLES_PER_BLOCK})")

    blocks = bytearray()
    h1 = h2 = 0.0

    for i in range(n_blocks):
        chunk = pcm[i * SAMPLES_PER_BLOCK:(i + 1) * SAMPLES_PER_BLOCK].astype(np.float64)

        if loop:
            if i == ls_block:
                flags = 0x06        # loop start
            elif i == le_block:
                flags = 0x03        # loop end → jump back to loop start
            else:
                flags = 0x00
        else:
            flags = 0x01 if i == n_blocks - 1 else 0x00   # one-shot end

        block, h1, h2 = _encode_block(chunk, h1, h2, flags)
        blocks.extend(block)

    blocks.extend(b'\x00\x07' + b'\x00' * 14)   # terminating silence block

    name_b = name[:15].encode("ascii", errors="replace").ljust(16, b"\x00")
    header = struct.pack(
        ">4sIIII12x16s",
        b"VAGp", 0x00000020, 0, len(blocks), sample_rate, name_b,
    )
    return header + bytes(blocks)


def _load_pcm(path: Path, sample_rate: int) -> np.ndarray:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found — install with: brew install ffmpeg")
    cmd = [
        ffmpeg, "-y", "-i", str(path),
        "-ac", "1", "-ar", str(sample_rate),
        "-f", "s16le", "-loglevel", "error", "pipe:1",
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())
    return np.frombuffer(result.stdout, dtype="<i2").copy()


def wav_to_vag(src: Path, dst: Path, sample_rate: int,
               loop: bool, loop_start: int = 0, loop_end: int = -1) -> None:
    pcm  = _load_pcm(src, sample_rate)
    data = _pcm_to_vag(pcm, sample_rate, src.stem, loop, loop_start, loop_end)
    dst.write_bytes(data)

# ─── DECODING (VAG → MP3) ────────────────────────────────────────────────────

def _vag_to_pcm(data: bytes) -> tuple[np.ndarray, int]:
    if len(data) < 48 or data[:4] != b"VAGp":
        raise ValueError("not a valid VAG file")

    sample_rate = struct.unpack_from(">I", data, 16)[0]
    body        = data[48:]
    samples     = []
    h1 = h2     = 0.0

    for i in range(0, len(body) - 15, 16):
        block      = body[i:i + 16]
        header_b   = block[0]
        flags      = block[1]
        shift      = min(header_b & 0x0F, 12)
        fi         = min((header_b >> 4) & 0x0F, 4)
        f0, f1     = _COEFF[fi]
        scale      = 1 << (12 - shift)

        for j in range(14):
            byte = block[2 + j]
            for nibble_raw in (byte & 0x0F, (byte >> 4) & 0x0F):
                nibble = nibble_raw if nibble_raw < 8 else nibble_raw - 16
                decoded = max(-32768.0, min(32767.0, nibble * scale + f0 * h1 + f1 * h2))
                samples.append(int(decoded))
                h2, h1 = h1, decoded

        if flags & 0x01:  # end-of-data flag
            break

    return np.array(samples, dtype=np.int16), sample_rate


def vag_to_mp3(src: Path, dst: Path) -> None:
    pcm, sample_rate = _vag_to_pcm(src.read_bytes())

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        sys.exit("ffmpeg not found — install with: brew install ffmpeg")

    cmd = [
        ffmpeg, "-y",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
        "-i", "pipe:0",
        "-loglevel", "error",
        str(dst),
    ]
    result = subprocess.run(cmd, input=pcm.tobytes(), capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.decode(errors="replace").strip())

# ─── CLI ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="PS1 VAG audio converter")
    parser.add_argument("files", nargs="*",
                        help="Files to convert (default: auto-detect from folders)")
    parser.add_argument("-d", "--decode", action="store_true",
                        help="Decode VAG → MP3 (default: encode WAV → VAG)")
    parser.add_argument("-r", "--rate", type=int, default=22050,
                        help="Sample rate for encoding (default: 22050)")
    parser.add_argument("-l", "--loop", action="store_true",
                        help="Mark sample as looping (encoding only)")
    parser.add_argument("--ls", type=int, default=0, metavar="SAMPLES",
                        help="Loop start in samples (auto-aligned down to multiple of 28, default: 0)")
    parser.add_argument("--le", type=int, default=-1, metavar="SAMPLES",
                        help="Loop end in samples (auto-aligned down to multiple of 28, default: end of file)")
    args = parser.parse_args()

    if args.decode:
        MP3_DIR.mkdir(exist_ok=True)
        srcs = [Path(f) for f in args.files] if args.files else sorted(OUTPUT_DIR.glob("*.vag"))
        if not srcs:
            sys.exit(f"No VAG files found in {OUTPUT_DIR}")
        ok = fail = 0
        for vag in srcs:
            mp3 = MP3_DIR / (vag.stem + ".mp3")
            try:
                print(f"  {vag.name}", end=" ... ", flush=True)
                vag_to_mp3(vag, mp3)
                print(f"→ {mp3.name}")
                ok += 1
            except Exception as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                fail += 1
    else:
        OUTPUT_DIR.mkdir(exist_ok=True)
        srcs = [Path(f) for f in args.files] if args.files else sorted(INPUT_DIR.glob("*.[wW][aA][vV]"))
        if not srcs:
            sys.exit(f"No WAV files found in {INPUT_DIR}")
        ok = fail = 0
        for wav in srcs:
            vag = OUTPUT_DIR / (wav.stem + ".vag")
            try:
                print(f"  {wav.name}", end=" ... ", flush=True)
                wav_to_vag(wav, vag, args.rate, args.loop, args.ls, args.le)
                print(f"→ {vag.name}")
                ok += 1
            except Exception as exc:
                print(f"FAILED: {exc}", file=sys.stderr)
                fail += 1

    print(f"\n{ok} converted, {fail} failed.")
    if fail:
        sys.exit(1)


if __name__ == "__main__":
    main()
