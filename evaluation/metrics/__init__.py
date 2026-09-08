"""Metric suite for comparing subtask annotation tools against human labels.

Layered so that every headline number can be traced to an assumption:

``segmentation``  timing only, no labels read -- the assumption-free layer
``semantic``      label equivalence, with pluggable matcher backends
``joint``         standard TAS / dense-captioning metrics needing both
``consistency``   cross-episode stability and the script-prior confound
``stats``         clustered bootstrap CIs and paired tests
``score``         per-episode driver that produces one flat record
"""

from .score import score_episode, EpisodeScore  # noqa: F401
