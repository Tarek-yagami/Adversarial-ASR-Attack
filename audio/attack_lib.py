"""Shared, testable pieces of the Wav2Vec2 PGD attack (see attack_wav2vec2.ipynb for the full pipeline)."""

import torch


def project_to_epsilon_ball(adv_audio, audio_tensor, epsilon):
    """Clamp adv_audio back onto the L-inf epsilon ball around audio_tensor, then to [-1, 1].

    Must run *after* any post-processing (psychoacoustic masking, energy filtering) that isn't
    itself epsilon-aware - otherwise that post-processing can silently push the perturbation past
    the stated epsilon budget (this was a real bug in an earlier version of this notebook: an
    epsilon=0.03 run was measured producing a max perturbation of 0.175, ~6x over budget).
    """
    perturbation = torch.clamp(adv_audio - audio_tensor, min=-epsilon, max=epsilon)
    return torch.clamp(audio_tensor + perturbation, min=-1, max=1)
