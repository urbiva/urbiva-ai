"""urbiva_ai: research code for UVA-Net S10, a lightweight urban accessibility vision model.

Status: implemented as code, not trained. This package contains the model
definition, data pipeline, training loop, evaluation metrics, export and a
local serving endpoint. It ships with no dataset, no weights and no results.

Copyright (c) 2026 the members of the team "Maximilian". All rights reserved.
Licensed under the Urbiva Proprietary License 1.0. See LICENSE.
"""

from urbiva_ai.labels import CATEGORIES, NONE_CATEGORY, BARRIER_CATEGORIES

__all__ = ["CATEGORIES", "NONE_CATEGORY", "BARRIER_CATEGORIES", "__version__"]
__version__ = "0.1.0"
