# BookForge Studio: Grok Bot runbook

This is the exact procedure a Grok Bot follows to turn a YouTube **video**, **playlist** or **channel** into a branded LaTeX/PDF book with diagrams, charts, illustrations and screenshots. Follow the steps in order. Every command is safe to re-run, because the pipeline resumes from its checkpoints.

## 0. Ground rules

- **The EURI key.** It is `EURI_API_KEY`. Never print or echo it, and never write it into a script, log or commit. Check that it's present with `[ -n "$EURI_API_KEY" ] && echo key-present`.
- **Models.** Use only `euri-chat-fast`, `euri-chat-smart` and `euri-image-fast` (the defaults). EURI allows **2 concurrent requests**, so run one book at a time per key. The pipeline already caps itself at 2.
- **Concurrent jobs.** Give each one its own `--workspace` directory (and its own checkout or git worktree if you are changing code).
- **Long jobs.** Run them in the background, with a log file, and check on them; don't block on them.

## 1. One-time setup (box)

```bash
# System tools (Debian/Ubuntu). TinyTeX/TeX Live provides pdflatex.
sudo apt-get update && sudo apt-get install -y ffmpeg graphviz librsvg2-bin
export PATH=/home/box/bin:$PATH         # where pdflatex lives on the Grok Bot box
which pdflatex ffmpeg yt-dlp dot rsvg-convert

# LaTeX packages used by the book (TinyTeX: tlmgr install <pkg>):
kpsewhich fancyhdr.sty tikz.sty standalone.cls eso-pic.sty || \
  tlmgr install fancyhdr pgf standalone eso-pic

# Python env
git clone https://github.com/Yash-Kavaiya/BookForge-Euron.git && cd BookForge-Euron
python -m venv .venv && . .venv/bin/activate
pip install -e ".[pipeline]"
```

On the Grok Bot box an env already exists: `/workspace/bfe-venv/bin/python`, with the code at `/workspace/bookforge-euron` (main) or `/workspace/bookforge-studio` (the Studio branch worktree). Set `PYTHONPATH` to the checkout you want.

Run the tests offline (EURI is mocked and the key is removed for the test session):

```bash
python -m pytest -q -p no:warnings
```

## 2. Identify the source

Check that the URL resolves before spending any model calls:

```bash
yt-dlp --flat-playlist -J "<URL>" | python -c "import json,sys;d=json.load(sys.stdin);print(d.get('title'), len(d.get('entries') or [1]))"
```

| The user gave | URL form | Notes |
| --- | --- | --- |
| a video | `youtu.be/ID`, `watch?v=ID`, `/shorts/ID`, `/live/ID` | 1 chapter. Duration filters don't apply |
| a playlist | `playlist?list=ID` or `watch?v=..&list=ID` | Chapters follow playlist order. Use `--max-videos N` to cap |
| a channel | `@handle`, `/channel/UC..`, `/c/..`, `/user/..` | Use `--max-videos N` and `--channel-sort newest\|popular`. Chapters are grouped into parts by the channel's playlists unless you pass `--no-group-by-playlists` |

If `yt-dlp` says the playlist doesn't exist (for example an ID truncated in chat), ask the user for the full link. Don't guess.

## 3. Prepare branding (optional)

- **Logo.** Any PNG, JPG, PDF or SVG works. A transparent PNG about 1000×250 px looks best in headers.
- **Cover.** Pass `--cover-image path.png` to use the user's own art. Otherwise a cover is generated with `euri-image-fast` from the title (override the prompt with `--cover-prompt`), or pass `--no-cover`.
- **Header and footer.** `--header-left`, `--header-right` and `--footer`. The page number is always centred in the footer, and the right footer links to each chapter's video.

## 4. Run

```bash
export PATH=/home/box/bin:$PATH
export PYTHONPATH=/workspace/bookforge-studio        # or your checkout
export BOOKFORGE_DISABLE_MLFLOW_TRACING=1            # no MLflow server on the box
OUT=/workspace/books                                  # one dir per concurrent job
nohup python -m bookforge.main "<URL>" \
  --workspace "$OUT" --max-videos 3 --visuals rich \
  --title "Book title" --subtitle "Subtitle" --author "Author line" \
  --header-left "Left header" --header-right "Right header" --footer "Footer text" \
  --logo /path/logo.png --logo-positions title,header,footer \
  > "$OUT/run.log" 2>&1 &
```

All flags have `BOOKFORGE_*` env equivalents (see the README's *BookForge Studio* table). The ones that matter most:

| Need | Flag / env |
| --- | --- |
| How many visuals | `--visuals light\|standard\|rich` / `BOOKFORGE_VISUALS` |
| Diagram renderer | `--diagram-engine tikz\|graphviz\|auto` |
| No AI images | `--no-illustrations --no-cover` |
| Fewer or more screenshots | `BOOKFORGE_MAX_FRAMES_PER_CHAPTER=N` |
| Cheaper writer | `BOOKFORGE_WRITER_MODEL=euri-chat-fast` |
| Skip the LLM critic | `BOOKFORGE_QA_LLM_CRITIC_MODE=never` |

## 5. Monitor

```bash
tr '\r' '\n' < "$OUT/run.log" | grep -v '^\[download\]' | tail -20
python - <<'PY'
import json,glob
for m in glob.glob("/workspace/books/*/manifest.json"):
    d=json.load(open(m)); print(m, [(v["chapter_number"], v["status"]) for v in d["videos"]])
PY
```

Each chapter moves through `pending → media → analyzed → assets → written → verified` (or `failed`), and it typically takes 5-15 minutes. The log shows `Assets for chapter N: X figures (frames, charts, diagrams, illustrations)` and finishes with `BOOK READY: <path>`.

If the process dies or the box restarts, run **the same command again**. The manifest, downloaded media, analyses (cached by transcript hash), rendered visuals and written chapters are all reused.

## 6. Outputs

Everything lands in `<workspace>/<slug>/`, where the slug comes from the playlist, channel or video title:

| File | What |
| --- | --- |
| `book/book.pdf` | The finished book: cover, title page, TOC, list of figures, parts and chapters |
| `book/cover.png` | The cover image |
| `book/latex_source.zip` | `main.tex`, `preamble.tex`, chapters, figures, tables and branding. Rebuild it with `pdflatex main.tex` (twice) |
| `book/book_manifest.json` | Title, chapters included or failed, logos, parts |
| `chapters/NN_slug/` | `chapter.tex`, `figures/` (diagram_*.pdf, chart_*.pdf, illustration_*.png, frame_*.jpg), `assets.json`, `analysis.json`, QA reports |
| `run_metrics.json` | Timings, LLM calls and tokens per chapter |

Page previews to share or check:

```bash
pdfinfo book/book.pdf | grep Pages
pdftoppm -png -r 70 -f 1 -l 3 book/book.pdf /tmp/preview   # cover, title page, TOC
```

Figure counts by type: `cat chapters/*/assets.json | grep -o '"kind": "[a-z]*"' | sort | uniq -c`.

## 7. Troubleshooting

| Symptom | Fix |
| --- | --- |
| `EURI_API_KEY is not set` | Export the key in the environment, never inline on the command line |
| `pdflatex not found` | `export PATH=/home/box/bin:$PATH`, or install TeX Live/TinyTeX |
| `chart '...' dropped: values not stated in transcript` | This is expected: the grounding rule is working, not an error |
| `diagram '...' dropped` / `image ... dropped` | That visual failed to render, so it was skipped. The book still builds. Check `pdflatex` and `dot` for diagrams, and the EURI status or image quota for illustrations |
| `Sign in to confirm you're not a bot` / HTTP 429 from YouTube | Wait and re-run (it resumes). Lower `BOOKFORGE_MEDIA_PREFETCH_CONCURRENCY=1` |
| Captions only in another language (e.g. `hi-orig`) | Normal. The analyst translates into English. The chapter notes it's based on auto-captions |
| `EURI ... HTTP 429` | The rate limit (30/min, 500/hour) was hit. The pipeline retries with backoff, so just leave it running |
| A chapter `failed` | See `error` in `manifest.json`. Re-run to retry only the failed or pending chapters. Set `"status": "pending"` to force a redo |
| `pdflatex failed on the book` | Check `book/main.log`. A branded compile that fails is retried once with plain front matter. If it still fails, the culprit is a chapter: set that chapter to `failed` in the manifest and re-run to compile without it |
| Wrong chapter order (newest-first playlist) | Channels: use `--channel-sort`. Playlists follow YouTube's order |
| Fonts or packages missing in TinyTeX | `tlmgr install <package>` (e.g. `standalone`, `eso-pic`, `pgf`, `fancyhdr`) |

## 8. Report back to the user

Give them:

- the PDF path, or a Drive link if you uploaded it;
- the page count;
- the chapters included and any that failed;
- figure counts by type;
- 2-3 page previews (`pdftoppm`);
- any honest caveats, such as auto-translated captions or charts that were dropped for lacking stated numbers.
