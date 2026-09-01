"""Console-script entry point, with a readable failure when deps are missing.

Mirrors `apps/dreamverse/dreamverse/server_entry.py`: the heavy imports live
behind `main`, so a missing runtime dependency surfaces as one sentence
telling you what to install rather than a traceback out of a transitive
import.
"""

from __future__ import annotations


def cli() -> None:
    try:
        from livestream.main import cli as main_cli
    except ModuleNotFoundError as exc:
        if exc.name in {"fastvideo", "torch", "torchaudio", "transformers"}:
            raise SystemExit("livestream-server requires FastVideo runtime deps. Install "
                             "`fastvideo[livestream]` or run `uv sync --extra livestream` "
                             "from the FastVideo checkout.") from exc
        if exc.name in {"fastapi", "uvicorn", "openai", "dotenv"}:
            raise SystemExit(f"livestream-server requires the `{exc.name}` package; install "
                             "the app's dependencies (`uv pip install -e apps/livestream`).") from exc
        raise

    main_cli()


if __name__ == "__main__":
    cli()
