#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Shared fixtures for annotation-pipeline tests.

The on-disk dataset builder lives with the other dataset factories in
``tests/fixtures/dataset_factories.py`` (:func:`build_annotation_dataset`);
these fixtures only wire it into pytest.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest


# HuggingFace cache locations that the installed libraries freeze into module
# constants at import time. ``datasets`` is imported during collection (the
# test modules ``pytest.importorskip`` it), which is before any session
# fixture runs, so setting the environment variables alone is not enough --
# the already-computed constants have to be rebound as well. Each entry maps
# an attribute to its path under the throwaway cache root.
_HF_CACHE_CONSTANTS: dict[str, dict[str, tuple[str, ...]]] = {
    "huggingface_hub.constants": {
        "HF_HOME": (),
        "HF_HUB_CACHE": ("hub",),
        "HUGGINGFACE_HUB_CACHE": ("hub",),
        "HF_ASSETS_CACHE": ("assets",),
        "HUGGINGFACE_ASSETS_CACHE": ("assets",),
        "HF_XET_CACHE": ("xet",),
    },
    "datasets.config": {
        "HF_CACHE_HOME": (),
        "HF_DATASETS_CACHE": ("datasets",),
        "HF_MODULES_CACHE": ("modules",),
        "DOWNLOADED_DATASETS_PATH": ("datasets", "downloads"),
        "EXTRACTED_DATASETS_PATH": ("datasets", "downloads", "extracted"),
    },
}

# Environment variables the same libraries read on a later, fresh import.
# The specific variables have to be overridden individually: they win over
# ``HF_HOME`` when they are already set, so pointing ``HF_HOME`` at a
# scratch directory would leave the rest aimed at the developer's cache.
_HF_CACHE_ENV: dict[str, tuple[str, ...]] = {
    "HF_HOME": (),
    "HF_DATASETS_CACHE": ("datasets",),
    "HF_HUB_CACHE": ("hub",),
    "HF_ASSETS_CACHE": ("assets",),
    "HF_XET_CACHE": ("xet",),
}


@pytest.fixture(scope="session", autouse=True)
def _hermetic_hf_cache(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Path]:
    """Point every HuggingFace cache at a throwaway directory.

    Without this the suite reads and writes the developer's global cache, so
    what passes depends on ambient ``HF_*`` variables rather than on the code
    under test. On a machine whose ``HF_DATASETS_CACHE`` points somewhere
    unwritable, four otherwise-green tests fail with ``PermissionError``
    raised from inside ``datasets``.
    """
    import sys

    root = tmp_path_factory.mktemp("hf-cache")

    saved_env = {name: os.environ.get(name) for name in _HF_CACHE_ENV}
    for name, parts in _HF_CACHE_ENV.items():
        os.environ[name] = str(root.joinpath(*parts))

    saved_constants: list[tuple[Any, str, Any]] = []
    for module_name, attributes in _HF_CACHE_CONSTANTS.items():
        module = sys.modules.get(module_name)
        if module is None:
            # Not imported yet; the environment variables above will steer it.
            continue
        for attribute, parts in attributes.items():
            previous = getattr(module, attribute, None)
            if previous is None:
                continue
            saved_constants.append((module, attribute, previous))
            # Preserve the attribute's type -- these are ``str`` in
            # ``huggingface_hub`` and ``Path`` in ``datasets``.
            setattr(module, attribute, type(previous)(str(root.joinpath(*parts))))

    try:
        yield root
    finally:
        for module, attribute, previous in saved_constants:
            setattr(module, attribute, previous)
        for name, previous in saved_env.items():
            if previous is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = previous


@pytest.fixture
def fixture_dataset_root(tmp_path: Path) -> Path:
    """A tiny dataset with two episodes, 12 frames each at 10 fps."""
    from tests.fixtures import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds",
        episode_specs=[
            (0, 12, "Could you tidy the kitchen please?"),
            (1, 12, "Please clean up the kitchen"),
        ],
        fps=10,
    )


@pytest.fixture
def single_episode_root(tmp_path: Path) -> Path:
    from tests.fixtures import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds_one",
        episode_specs=[(0, 30, "Pour water from the bottle into the cup.")],
        fps=10,
    )


@pytest.fixture
def align_dataset_root(tmp_path: Path) -> Path:
    """One 21-frame episode at 10 fps, i.e. exactly 0.0s -> 2.0s.

    Sampling it at ``frames_per_second=2.0`` lands on the round tile times
    [0.0, 0.5, 1.0, 1.5, 2.0], which keeps the given-subtask alignment tests
    readable — every tile is also an exact source frame, so snapping is a
    no-op and the asserted boundaries are the ones the stub voted for.
    """
    from tests.fixtures import build_annotation_dataset

    return build_annotation_dataset(
        tmp_path / "ds_align",
        episode_specs=[(0, 21, "Pour water from the bottle into the cup.")],
        fps=10,
    )
