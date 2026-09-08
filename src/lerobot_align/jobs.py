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
"""Submit :command:`lerobot-align` runs to Hugging Face Jobs.

The remote runner belongs to this package rather than LeRobot.  A job installs
both the selected LeRobot revision and the explicitly pinned remote
``lerobot-align`` wheel or source commit, then invokes this package's unambiguous console
script.  Submissions must provide an immutable wheel URL or PEP 508 VCS requirement
through ``--job.package_spec``.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
import re
import shlex
import sys
from dataclasses import is_dataclass
from typing import TYPE_CHECKING

from huggingface_hub import HfApi, get_token, run_job
from lerobot.jobs.hf import _pod_forwarded_args, follow_job, resolve_job_tags

if TYPE_CHECKING:
    from .config import AnnotationPipelineConfig

logger = logging.getLogger(__name__)

LEROBOT_GIT_URL = "https://github.com/huggingface/lerobot.git"

# The vLLM image supplies the CUDA-tested torch/vLLM stack. LeRobot and
# lerobot-align are installed without dependencies so pip cannot replace it;
# install and constrain the compatible user-space runtime explicitly.
_RUNTIME_REQUIREMENTS = (
    # The official image includes PyGObject but omits its pycairo dependency.
    # Complete the image's declared requirements so pip check remains strict.
    "pycairo>=1.20.0,<2.0.0",
    # vLLM 0.19.1 ships PyTorch 2.10; torchvision 0.25 is its matching
    # release. Its Transformers 5.5.3 upgrade is also compatible with the Hub
    # 1.x floor required by LeRobot 0.6.1.
    "torchvision>=0.25.0,<0.26.0",
    "transformers>=5.5.3,<5.6.0",
    "numpy>=2.0.0,<2.3.0",
    "opencv-python-headless>=4.9.0,<4.14.0",
    "Pillow>=10.0.0,<13.0.0",
    "einops>=0.8.0,<0.9.0",
    "requests>=2.32.0,<3.0.0",
    "gymnasium>=1.1.1,<2.0.0",
    "safetensors>=0.4.3,<1.0.0",
    "packaging>=24.2,<26.0",
    "termcolor>=2.4.0,<4.0.0",
    "tqdm>=4.66.0,<5.0.0",
    "cmake>=3.29.0.1,<4.2.0",
    "setuptools>=77.0.3,<81.0.0",
    "datasets>=4.8.0,<5.0.0",
    "pyarrow>=21.0.0,<30.0.0",
    "av>=15.0.0,<16.0.0",
    "draccus>=0.11.6,<0.12.0",
    "huggingface-hub>=1.6.0,<2.0.0",
    "pandas>=2.0.0,<3.0.0",
    "jsonlines>=4.0.0,<5.0.0",
    "mergedeep",
    "pyyaml-include",
    "toml",
    "typing-inspect",
    "openai>=2.0,<3.0",
)

# Client-local paths and orchestration settings cannot be replayed in the pod.
# Nested ``--job.*`` flags are removed separately by prefix.
_SUBMITTER_OWNED_ARGS = ("--root", "--repo_id", "--config_path", "--job", "--vlm.api_key")


def resolve_package_spec(package_spec: str | None) -> str:
    """Return the package requirement that the remote pod should install.

    A source version number does not prove that a package is published. Require
    an explicit immutable artifact before any paid resources are provisioned.
    """
    if not package_spec:
        raise ValueError(
            "Remote Jobs require --job.package_spec: this checkout is not a published package. "
            "Provide an HTTPS wheel URL with #sha256=<digest>, or a git+https requirement "
            "pinned to a full commit SHA. No job has been submitted."
        )
    if not (re.search(r"https://[^\s]+\.whl#sha256=[0-9a-f]{64}$", package_spec)
            or re.search(r"git\+https://[^\s]+@[0-9a-f]{40}$", package_spec)):
        raise ValueError("--job.package_spec must name an immutable HTTPS wheel (with SHA256) "
                         "or VCS requirement pinned to a full commit SHA")
    return package_spec


def _local_config_file_args(cfg: AnnotationPipelineConfig) -> list[str]:
    """Return CLI arguments that refer to configuration files on this host."""
    return [
        "--config_path",
        *(f"--{name}" for name in vars(cfg) if is_dataclass(getattr(cfg, name))),
    ]


def build_pod_setup(lerobot_ref: str, package_spec: str) -> str:
    """Build the shell prelude that installs a reproducible remote runtime."""
    lerobot_spec = f"lerobot @ git+{LEROBOT_GIT_URL}@{lerobot_ref}"
    commands = [
        "apt-get update -qq",
        "apt-get install -y -qq git ffmpeg libcairo2-dev pkg-config",
        shlex.join(["python3", "-m", "pip", "install", "--no-deps", lerobot_spec]),
        shlex.join(["python3", "-m", "pip", "install", "--no-deps", package_spec]),
        shlex.join([
            "python3", "-c",
            "from importlib.metadata import distribution; from pathlib import Path; "
            "Path('/tmp/lerobot-align-job-constraints.txt').write_text("
            "distribution('lerobot-align').locate_file('lerobot_align/jobs-constraints.txt').read_text())",
        ]),
        shlex.join(
            [
                "python3",
                "-m",
                "pip",
                "install",
                "--constraint", "/tmp/lerobot-align-job-constraints.txt",
                "--upgrade-strategy",
                "only-if-needed",
                *_RUNTIME_REQUIREMENTS,
            ]
        ),
        "python3 -m pip check",
        shlex.join(
            [
                "python3",
                "-c",
                "import av, cv2, datasets, huggingface_hub, lerobot, lerobot_align, torch, "
                "torchvision, transformers, vllm; import lerobot_align.cli",
            ]
        ),
        # vLLM's cudagraph estimate can over-reserve and starve the KV cache.
        "export VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0",
        "export VLLM_VIDEO_BACKEND=pyav",
    ]
    return " && ".join(commands)


def _declares(argv: list[str], name: str) -> bool:
    """Whether ``argv`` sets ``name``, in either ``--n=v`` or ``--n v`` form."""
    return any(arg == name or arg.startswith(f"{name}=") for arg in argv)


def build_pod_command(
    repo_id: str,
    lerobot_ref: str,
    package_spec: str,
    argv: list[str],
) -> list[str]:
    """Build the pod command while dropping host-only and recursive-job flags."""
    forwarded = _pod_forwarded_args(
        argv,
        drop_names=_SUBMITTER_OWNED_ARGS,
        drop_prefixes=("--job.",),
    )
    # The pod is a freshly provisioned GPU machine with nothing listening on
    # ``VlmConfig.api_base``, so it has to start its own server. ``auto_serve``
    # defaults to False for local runs -- where assuming a 27B server should be
    # spawned is hostile -- and that default is wrong here in a way that fails
    # late, after the image has been paid for and the dataset downloaded. The
    # remote default is therefore inverted, and only an explicit
    # ``--vlm.auto_serve`` from the submitter overrides it: a job pointed at an
    # already-running remote endpoint passes ``--vlm.auto_serve=false``.
    if not _declares(forwarded, "--vlm.auto_serve"):
        forwarded = [*forwarded, "--vlm.auto_serve=true"]
    align = shlex.join(["lerobot-align", f"--repo_id={repo_id}", *forwarded, "--job.target=local"])
    return ["bash", "-c", f"{build_pod_setup(lerobot_ref, package_spec)} && {align}"]


def submit_align_to_hf(cfg: AnnotationPipelineConfig) -> None:
    """Submit an alignment run, then optionally follow it to completion."""
    token = get_token()
    if not token:
        raise RuntimeError("Not logged in to Hugging Face. Run `hf auth login` first.")

    if cfg.repo_id is None:
        raise ValueError(
            "Remote alignment requires --repo_id: the pod cannot access a --root path "
            "from this machine."
        )

    argv = sys.argv[1:]
    passed = {tok.split("=", 1)[0] for tok in argv}
    used_config_files = sorted(passed.intersection(_local_config_file_args(cfg)))
    if used_config_files:
        raise ValueError(
            f"{', '.join(used_config_files)} cannot be used with a remote --job.target: the pod "
            "cannot read config files from this machine. Pass the settings as CLI flags instead."
        )

    for name, value in (("plan.subtasks_path", cfg.plan.subtasks_path),
                        ("plan.subtask_align_calibration_path", cfg.plan.subtask_align_calibration_path),
                        ("staging_dir", cfg.staging_dir)):
        if value is not None:
            raise ValueError(
                f"--{name}={Path(value).name} refers to a submitter-local path and cannot be "
                "read by the remote job. Run this configuration locally, or prepare the "
                "required annotations in the source Hub dataset and use an explicit "
                "--plan.subtask_import source. No job has been submitted."
            )

    if not cfg.push_to_hub:
        logger.warning(
            "WARNING: --push_to_hub is off. The aligned dataset lives only on the pod and is "
            "discarded when the job ends. Pass --push_to_hub=true to keep the result."
        )

    package_spec = resolve_package_spec(cfg.job.package_spec)
    api = HfApi(token=token)
    tags = resolve_job_tags(["lerobot-align", *cfg.job.tags])
    if not api.repo_exists(cfg.repo_id, repo_type="dataset"):
        raise RuntimeError(
            "Remote Jobs require an existing accessible Hub dataset. Local cached files are "
            "never uploaded implicitly. Check --repo_id and token access, or publish the "
            "source separately through a reviewed explicit dataset manifest. No job has been submitted."
        )
    command = build_pod_command(cfg.repo_id, cfg.job.lerobot_ref, package_spec, argv)

    logger.info(
        "Submitting job to HF Jobs (flavor=%s, image=%s) ...", cfg.job.target, cfg.job.image
    )
    job_info = run_job(
        image=cfg.job.image,
        command=command,
        flavor=cfg.job.target,
        env={
            "LEROBOT_VLM_VIDEO_METADATA_SOURCE": cfg.vlm.video_metadata_source,
            "LEROBOT_OPENAI_SEND_MM_KWARGS": "1" if os.environ.get(
                "LEROBOT_OPENAI_SEND_MM_KWARGS", ""
            ).lower() in {"1", "true", "yes"} else "0",
        },
        secrets={"HF_TOKEN": token, **({"LEROBOT_VLM_API_KEY": cfg.vlm.api_key} if cfg.vlm.api_key != "EMPTY" else {})},
        timeout=cfg.job.timeout,
        labels=dict.fromkeys(tags, "true"),
    )
    job_id = job_info.id
    job_url = getattr(job_info, "url", None)
    logger.info("Job submitted: %s", job_id)
    if job_url:
        logger.info("  Job page:     %s", job_url)
    target_repo_id = cfg.new_repo_id or cfg.repo_id
    if cfg.push_to_hub:
        logger.info("  Dataset repo: https://huggingface.co/datasets/%s", target_repo_id)
    logger.info("  Monitor:      hf jobs logs %s", job_id)
    logger.info("  Cancel:       hf jobs cancel %s", job_id)

    if not follow_job(job_id, detach=cfg.job.detach):
        return

    if cfg.push_to_hub:
        logger.info(
            "\nAlignment complete — dataset pushed to https://huggingface.co/datasets/%s",
            target_repo_id,
        )
    else:
        logger.info(
            "\nAlignment complete. Note: --push_to_hub was off, so the result stayed on the pod."
        )
