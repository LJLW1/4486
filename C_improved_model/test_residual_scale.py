"""Fast no-dataset checks for V18 residual scaling.

Run: python -m unittest discover -s C_improved_model -p test_residual_scale.py -v
"""
import unittest

import numpy as np
import torch
from torch import nn

from architectures import ResidualNetwork
from teammate_aligned_model import NN_INDICES
from validate_residual_scale import evaluate_scales


class _Predictor:
    def __init__(self, network):
        self.networks = [network]

    def predict_features(self, features, target_indices, heads, protein_values):
        del target_indices, protein_values
        with torch.no_grad():
            values = self.networks[0](torch.tensor(features, dtype=torch.float32))
        return values[torch.arange(len(features)), heads].numpy()


class ResidualScaleTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4486)
        self.network = ResidualNetwork().eval()
        # Force a known correction to make the scale effect measurable.
        with torch.no_grad():
            nn.init.zeros_(self.network.correction[-1].weight)
            nn.init.constant_(self.network.correction[-1].bias, 0.5)
        self.x = np.zeros((3, len(NN_INDICES) + 4), dtype=np.float32)
        self.x[0, -2] = 1.0
        self.x[2, -1] = 1.0
        self.proteins = np.array([[1.0, np.nan], [np.nan, np.nan], [np.nan, 2.0]])
        self.targets = np.zeros(3, dtype=int)
        self.heads = np.zeros(3, dtype=int)

    def test_default_lambda_is_original_v18(self):
        self.assertEqual(self.network.residual_scale, 1.0)
        with torch.no_grad():
            x = torch.tensor(self.x)
            old_formula = (
                self.network.base(x[:, :len(NN_INDICES)]) +
                self.network.correction(x) * x[:, -2:].amax(dim=1, keepdim=True)
            ).clamp_min(0)
            self.assertTrue(torch.allclose(self.network(x), old_formula, atol=0, rtol=0))

    def test_lambda_zero_suppresses_correction_and_missing_is_invariant(self):
        with torch.no_grad():
            reference = self.network(torch.tensor(self.x)).numpy()
            self.network.residual_scale = 0.0
            scaled = self.network(torch.tensor(self.x)).numpy()
        self.assertTrue(np.allclose(scaled[1], reference[1], atol=0, rtol=0))
        self.assertTrue(np.allclose(reference[[0, 2], 0] - scaled[[0, 2], 0], 0.5, atol=1e-6))

    def test_sweep_restores_network_state(self):
        model = _Predictor(self.network)
        predictions = evaluate_scales(
            model, self.x, self.targets, self.heads, self.proteins, (0.0, 0.25, 0.5, 0.75, 1.0)
        )
        self.assertEqual(set(predictions), {0.0, 0.25, 0.5, 0.75, 1.0})
        self.assertEqual(self.network.residual_scale, 1.0)
        self.assertTrue(np.allclose(
            [values[1] for values in predictions.values()], predictions[1.0][1], atol=1e-6
        ))


if __name__ == "__main__":
    unittest.main()
