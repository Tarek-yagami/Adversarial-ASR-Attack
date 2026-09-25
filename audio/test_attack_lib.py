"""Regression guard for the exact bug this project already shipped once: psychoacoustic masking and
energy filtering pushing the perturbation past the stated epsilon budget. Run with: pytest audio/
"""

import torch

from attack_lib import project_to_epsilon_ball


def test_stays_within_epsilon_even_if_input_drifted_further():
    audio = torch.zeros(1000)
    # Simulate masking/filtering having pushed adv_audio far outside the epsilon ball.
    drifted = audio + torch.linspace(-0.5, 0.5, 1000)
    epsilon = 0.03

    result = project_to_epsilon_ball(drifted, audio, epsilon)

    max_perturbation = (result - audio).abs().max().item()
    assert max_perturbation <= epsilon + 1e-6


def test_stays_within_hard_waveform_range():
    audio = torch.full((10,), 0.98)
    drifted = audio + 0.5  # would exceed 1.0 without the final clamp
    epsilon = 0.05

    result = project_to_epsilon_ball(drifted, audio, epsilon)

    assert result.max().item() <= 1.0
    assert result.min().item() >= -1.0


def test_leaves_input_unchanged_when_already_within_epsilon():
    audio = torch.linspace(-0.9, 0.9, 100)
    epsilon = 0.02
    already_close = audio + 0.01  # within epsilon

    result = project_to_epsilon_ball(already_close, audio, epsilon)

    assert torch.allclose(result, already_close, atol=1e-6)
