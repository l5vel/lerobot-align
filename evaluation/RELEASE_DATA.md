# Evaluation data and release boundaries

The two historical input archives have been removed. The exact member names,
sizes, SHA-256 hashes, classifications and source interval counts are recorded
in [release_artifact_inventory.json](release_artifact_inventory.json). No source
annotation text or timestamps are stored in that inventory. This is a release
packaging correction, not a rerun or a change to measured numerical results.

The inventory also records release redactions outside the archives: source and
baseline label sequences, calibration label lists, and authoritative text on
fixed-label predicted spans. Numeric measurements and predicted timings are
retained. Original and release hashes are both recorded; old artifact hash files
remain historical provenance. Those redacted files are reports, not runnable
calibration files. Reanalysis requiring labels must reconstruct private inputs.
Open-ended model-generated predictions remain generated research outputs.

`corpus_b_paired_ablations/inputs.tar.gz` contained seed annotations and 39
ground-truth files, together with fits, predictions, splits, scores and software.
`corpus_calibration_diagnosis/analysis_inputs.tar.gz` contained 16 Corpus A
ground-truth files, Corpus B seed annotations and 39 Corpus B ground-truth files,
together with mixed analysis inputs. The whole archives were removed because
separating a generated file from its copied authoritative labels is insufficient
to establish redistribution rights. The analysis script no longer exports them.

Original hashes referenced in frozen reports describe historical private inputs;
they are not a promise that those inputs ship with the public package. Historical
Git commits still contain removed data. **Do not publish this branch's history.**
Publish a reviewed clean source export/new repository without inherited commits.
Do not rewrite an existing shared history merely to make a release check pass.

## Reconstructing inputs with authorized data access

1. Obtain access directly from the dataset authors. Corpus B uses
   [InternRobotics/RoboInter-Data](https://huggingface.co/datasets/InternRobotics/RoboInter-Data),
   whose card records CC BY-NC-SA 4.0 and underlying dataset terms. This project
   makes no determination that a particular redistribution is permitted.
2. After reviewing those terms, download the pinned RH20T source. If access
   requires authentication, use `uv run --no-sync hf auth login` first. Choose
   a new directory outside the source checkout and public artifacts; this
   download can be very large:

   ```bash
   export ROBOINTER_DOWNLOAD="$HOME/lerobot-align-data/robointer-original"
   uv run --no-sync hf download InternRobotics/RoboInter-Data \
     --repo-type dataset --revision 0208b79c34eca4d214cdb5d07f5ae0f7cc634340 \
     --include README.md \
     --include 'Annotation_with_action_lerobotv21/lerobot_rh20t_anno/**' \
     --local-dir "$ROBOINTER_DOWNLOAD"
   ```

   Retain the original card and download records with the data. Review the
   pinned card's archive extraction instructions before preparing the subset.
3. Follow the preprocessing steps in [REPRODUCING.md](REPRODUCING.md):
   census, selection, subset, LeRobot v2.1 migration, ground-truth extraction,
   component construction, split translation and task sharding. Each script's
   `--help` documents its input/output arguments. Preserve index maps and dataset
   revisions alongside the resulting private working directories.
4. Run the shared protocol described in [calibration_protocol.md](calibration_protocol.md)
   with manifests pointing to those private prepared datasets. Do not bypass
   `prepare_robointer.py`'s provenance verification.
5. Reproducing the historical offline analyses additionally requires their saved
   stochastic predictions and fits. The inventory provides integrity references,
   not a substitute for those inputs. Re-running models produces a new experiment.

Both [Corpus A](configs/corpus_a_sources.json) and
[Corpus B](configs/corpus_b_sources.json) now record recovered immutable source
identities and local historical matching evidence. Anonymous metadata checks on
2026-09-08 resolved all 17 exact revisions successfully and reported no Hub gate.
The manifests contain identifiers and hashes, not source annotations. Full file
downloads and independent preparation were not rerun; saved historical
predictions/fits are also still required. Do not describe these source-identity
checks as independent reproduction of the historical studies.
