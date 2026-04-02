import numpy as np
import pytest

from vinD.pipeline import run_analysis


def test_run_analysis_rejects_small_vertex_arrays(tmp_path):
    preds = np.zeros((5, 32), dtype=float)

    with pytest.raises(ValueError, match="20,484"):
        run_analysis(preds, output_dir=tmp_path)
