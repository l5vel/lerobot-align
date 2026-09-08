"""Test-only tripwires for the operations covered by dependency advisories.

These do not patch the shipped application or suppress audit findings. They
make the scoped non-applicability assessment fail when a tested workflow starts
using an affected operation, even if application error handling catches it.
"""

from contextlib import ExitStack, contextmanager
from unittest.mock import patch


def watched_operations():
    import accelerate
    from accelerate import big_modeling, utils as accelerate_utils
    from accelerate.utils import modeling as accelerate_modeling
    import torch
    from torch.jit import _script
    from datasets.packaged_modules.folder_based_builder.folder_based_builder import FolderBasedBuilder
    from transformers import PreTrainedTokenizerBase, ProcessorMixin

    return (
        (accelerate, "load_checkpoint_in_model", "accelerate.load_checkpoint_in_model"),
        (accelerate, "load_checkpoint_and_dispatch", "accelerate.load_checkpoint_and_dispatch"),
        (big_modeling, "load_checkpoint_in_model", "accelerate.big_modeling.load_checkpoint_in_model"),
        (big_modeling, "load_checkpoint_and_dispatch", "accelerate.big_modeling.load_checkpoint_and_dispatch"),
        (accelerate_utils, "load_checkpoint_in_model", "accelerate.utils.load_checkpoint_in_model"),
        (accelerate_modeling, "load_checkpoint_in_model", "accelerate.modeling.load_checkpoint_in_model"),
        (FolderBasedBuilder, "_split_generators", "datasets.folder.split"),
        (FolderBasedBuilder, "_generate_examples", "datasets.folder.examples"),
        (FolderBasedBuilder, "_generate_shards", "datasets.folder.shards"),
        (PreTrainedTokenizerBase, "save_pretrained", "transformers.tokenizer.save"),
        (PreTrainedTokenizerBase, "save_chat_templates", "transformers.tokenizer.templates"),
        (ProcessorMixin, "save_pretrained", "transformers.processor.save"),
        (torch.jit, "script", "torch.jit.script"),
        (_script, "_script_impl", "torch.jit.script_impl"),
    )


@contextmanager
def rejected_advisory_operations():
    attempts = []

    def reject(name):
        def call(*args, **kwargs):
            attempts.append(name)
            raise AssertionError(f"Dependency advisory operation reached: {name}")
        return call

    with ExitStack() as stack:
        for owner, attribute, name in watched_operations():
            stack.enter_context(patch.object(owner, attribute, new=reject(name)))
        yield attempts
