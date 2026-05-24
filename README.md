# psss — PS1 VAG audio converter

Pure Python WAV → VAG encoder and VAG → MP3 decoder for PlayStation 1 audio.
Implements the full SPU-ADPCM algorithm entirely in Python (no external VAG tools needed).

---

## Features

- **WAV → VAG** — encodes any audio file to PS1 SPU-ADPCM format via ffmpeg + numpy
- **VAG → MP3** — decodes VAG back to MP3 for verification
- Loop point support with automatic alignment to 28-sample block boundaries (PS1 hardware requirement)
- Tries all 65 filter × shift combinations per block and picks lowest MSE — same quality as psxavenc
- Batch mode (entire `input/` folder) or single-file mode

---

## Requirements

```bash
pip install numpy
brew install ffmpeg   # macOS
# apt install ffmpeg  # Linux
```

---

## Usage

```bash
# WAV → VAG (all files in input/)
python3 convert.py

# Single file
python3 convert.py input/kick.wav

# Custom sample rate (default: 22050 Hz)
python3 convert.py -r 44100

# Loop entire sample
python3 convert.py -l

# Loop a specific region (auto-aligned to multiples of 28)
python3 convert.py -l --ls 5600 --le 11200

# VAG → MP3 (all files in output/)
python3 convert.py --decode

# VAG → MP3 single file
python3 convert.py --decode output/kick.vag
```

---

## Folder structure

```
psss/
├── convert.py       # main script
├── input/           # place WAV files here
├── output/          # encoded .vag files appear here
├── output_mp3/      # decoded .mp3 files appear here
└── tools/
    └── CONTEXT.txt  # PS1 VAG format technical reference
```

---

## Sample rates

| Rate | Character |
|------|-----------|
| 22050 Hz | Classic PS1 "underground" aliasing sound (default) |
| 44100 Hz | Full fidelity, no aliasing |

The PS1 SPU natively runs at 44100 Hz. 22050 Hz gives the characteristic lo-fi texture used in many PS1 games.

---

## Technical reference

See `tools/CONTEXT.txt` for a full breakdown of the VAG format, ADPCM encoding algorithm, loop flag values, and hardware constraints.
