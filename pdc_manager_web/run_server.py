from __future__ import annotations

import logging
import sys
from pathlib import Path
from time import perf_counter


BASE_DIR = Path(__file__).resolve().parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "pdc_web.log"


def configure_logging() -> None:
    log_format = (
        "%(asctime)s | %(levelname)-8s | "
        "%(name)s | %(message)s"
    )

    formatter = logging.Formatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        LOG_FILE,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)


configure_logging()
logger = logging.getLogger("pdc.server")

logger.info("Starting PDC web server")
logger.info("Log file: %s", LOG_FILE)
logger.info("Loading Waitress")

try:
    from waitress import serve
except Exception:
    logger.exception("Could not import Waitress")
    raise

logger.info("Importing the PDC application")
logger.info("YOLO and classifier models will be loaded now")

import_started = perf_counter()

try:
    from app import app
except Exception:
    logger.exception("Could not import the PDC application")
    raise

import_seconds = perf_counter() - import_started

logger.info(
    "PDC application and models loaded in %.2f seconds",
    import_seconds,
)


if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8000

    logger.info("Starting HTTP server")
    logger.info("Open the interface at http://%s:%d", host, port)
    logger.info("Press Ctrl+C to stop the server")

    try:
        serve(
            app,
            host=host,
            port=port,
            threads=4,
        )
    except KeyboardInterrupt:
        logger.info("Server stopped by the user")
    except Exception:
        logger.exception("The HTTP server stopped because of an error")
        raise