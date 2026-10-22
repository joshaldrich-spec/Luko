# Luko studio

A small pipeline for generating Luko art (and music) with the Gemini API, driven by Claude.

1. A job (JSON) describes what to make: model, a shared style line, reference images, and the items to generate.
2. `gh workflow run studio.yml --json` starts the **Studio (Gemini)** workflow with that job.
3. `gen.py` calls the Gemini API with the `GEMINI_API_KEY` repository secret and saves every output.
4. The results are committed to the `studio` branch under `runs/<run id>/` (with `job.json` and `log.json`).

Reference images live in `refs/`. Each run is capped at 24 generations. API use is billed to the Google account that owns the key.
