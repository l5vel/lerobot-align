# Qwen3.8-27B frame-format visual-token audit

This audit compares the two representations used by the fixed-label alignment
evaluation. Every pair contains the same wrist-camera timestamps sampled at
2 fps, capped at 300 frames, and resized to 224 px wide before it is encoded as
either 5-by-4 contact sheets or one native video.

The primary quantity is the official processor's expanded visual-placeholder
count:

```text
visual_tokens = sum(product(grid_thw) / merge_size^2)
```

For Qwen3.8-27B revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, the spatial patch size is 16,
the merge size is 2, and the video temporal patch size is 2.

| Population | Held-out trajectories | Contact-sheet visual tokens | Native-video visual tokens | Reduction |
|---|---:|---:|---:|---:|
| Corpus A | 757 | 3,568,425 | 1,584,275 | 55.60% |
| Corpus B | 1,305 | 3,869,040 | 1,771,784 | 54.21% |
| Pooled | 2,062 | 7,437,465 | 3,356,059 | 54.88% |

Native video had fewer visual tokens in all 2,062 matched pairs. The census is
metadata-driven because Qwen's resize and grid shapes depend on frame count and
geometry. To check that calculation against pixels, `validation.json` decodes
short, median-length, and long examples from each corpus. All six official
processor grids and input placeholder counts exactly matched the census. The
same 12 payloads also completed on a local Qwen3.8-27B server. The validation
receipt records the requested GPU index and observed NVIDIA H100 80GB device,
but it does not preserve enough launch provenance to independently prove the
server's dtype or device assignment.

This supports a claim about processor-expanded **visual tokens** for this
sampling and layout. It does not by itself measure generated tokens, billing,
latency, or annotation accuracy. Native video also inserts timestamp text, so
whole-prompt token counts are not identical to the plotted visual counts.
Server-reported token telemetry is retained in `validation.json` as a diagnostic
and is not used in the comparison.

Artifacts:

- `rows.jsonl`: all paired trajectory measurements.
- `summary.json`: population audits, aggregates, input hashes, processor config,
  model revision, and decoded-pair validation checksum.
- `validation.json`: six decoded-media checks and twelve GPU3 inference checks.
- `figure3.svg` and `figure3.png`: paired per-trajectory scatter plus normalized
  aggregate contact-sheet/video bars, labeled with absolute totals and
  ratio-of-sums reductions; derived entirely from `rows.jsonl`.

The scripts refuse invalid or unmatched rows. The plotter also refuses to emit
Figure 3 unless both corpora have positive ratio-of-sums, mean-pair, and
median-pair reductions.

## Data provenance and terms

The per-trajectory rows are derived metadata: episode identifiers, durations,
frame geometry, processor grids, and token counts. They contain no annotation
text, source pixels, media, or model predictions. Corpus A is derived from the
exact `AmmarWaheed` repositories and revisions listed in
`evaluation/configs/corpus_a_sources.json`; their Hub cards declare Apache-2.0.
Corpus B is derived from
[`InternRobotics/RoboInter-Data`](https://huggingface.co/datasets/InternRobotics/RoboInter-Data)
at revision `0208b79c34eca4d214cdb5d07f5ae0f7cc634340`; its card declares
CC BY-NC-SA 4.0 and refers users to the original RoboInter, DROID, and RH20T
terms. The repository's software license does not relicense these derived data
records. Use them subject to the upstream terms and attribution requirements;
see the root `NOTICE` and `evaluation/RELEASE_DATA.md` for the full qualification.
