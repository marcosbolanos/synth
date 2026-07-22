# Main instructions
- Use `uv` as a package manager with `uv add` and use `uv run` to run commands
- Instead of writing defensive code, ensure having very explicit type contracts so that the inputs of a function behave as expected
- Don't make any dependency fallbacks. Just make sure necessary packages are
  installed. Also don't write any code that checks if packages are installed,
  just install them
- When generating output files, put them in data/outputs/<generator_name_vX_timestap>/files, where the part in <> is some unique name for a subfolder that differentiates the code that generated the outputs from others, and a timestamp. All files are in a subfolder to differentiate. vX is the version of the generating script, like v1, v2 etc so we keep track of how code evolves.
- Web UI: add routes in `webui/app.py`, pages extending `webui/templates/base.html`, and shared assets under `webui/static/`.
