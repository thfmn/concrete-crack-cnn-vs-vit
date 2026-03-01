#  Copyright (C) 2026 by Tobias Hoffmann
#  thoffmann-ml@proton.me
#  https://github.com/thfmn/concrete-crack-cnn-vs-vit
#
#  This work is licensed under the MIT License.
#
#  Author:    Tobias Hoffmann
#  Email:     thoffmann-ml@proton.me
#  License:   MIT
#  Date:      2026
#  Package:   concrete-crack-cnn-vs-vit — CNN vs ViT Benchmark

"""Shared pytest fixtures and marker registration."""

from __future__ import annotations

from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Paths to real-image fixture directories
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SDNET2018_FIXTURE_DIR = FIXTURES_DIR / "sdnet2018"
CRACKSEG9K_FIXTURE_DIR = FIXTURES_DIR / "crackseg9k"


# ---------------------------------------------------------------------------
# Shared fixtures for integration tests
# ---------------------------------------------------------------------------


@pytest.fixture()
def real_sdnet2018_dir() -> Path:
    """Path to the fixed SDNET2018 fixture (12 real images, 2 per class)."""
    assert SDNET2018_FIXTURE_DIR.exists(), (
        f"SDNET2018 fixtures not found at {SDNET2018_FIXTURE_DIR}"
    )
    return SDNET2018_FIXTURE_DIR


@pytest.fixture()
def real_crackseg9k_dir() -> Path:
    """Path to the fixed CrackSeg9k fixture (3 real image+mask pairs)."""
    assert CRACKSEG9K_FIXTURE_DIR.exists(), (
        f"CrackSeg9k fixtures not found at {CRACKSEG9K_FIXTURE_DIR}"
    )
    return CRACKSEG9K_FIXTURE_DIR
