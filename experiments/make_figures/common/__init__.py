"""Dataset-agnostic helpers shared by the per-dataset DelRec analyses.

Import from the sub-modules:
  * ``common.utils``, device / figure saving, checkpoint & CSV readers,
                          run discovery, generic model rebuilding.
  * ``common.recdel``, multi-seed delay/weight statistics + plots.
  * ``common.gradmaps``, input-gradient temporal maps + energy profiles.

The per-dataset packages (``experiments/make_figures/SSC`` / ``experiments/make_figures/HAR``) supply only the
dataset-specific bits. The hardcoded run set, the ``Config`` to rebuild models
with, and the test-set evaluation, and wire them into these functions.
"""
