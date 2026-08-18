import torch
import torch.nn as nn


class DeepESCIReranker(nn.Module):
    def __init__(self, input_dim):
        super(DeepESCIReranker, self).__init__()
        # LayerNorm, not BatchNorm: training forwards positives and negatives
        # through the model in two separate batches, so BatchNorm would normalize
        # each side against its own batch statistics -- letting the network
        # separate them for free using the normalization itself instead of the
        # learned weights, with none of that "trick" available at eval time.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        # Raw score, not squashed through sigmoid: MarginRankingLoss(margin=1.0)
        # needs an unbounded score space to be satisfiable. A sigmoid-bounded
        # output caps the max achievable pos/neg gap at 1.0, which pushes the
        # network to saturate at the extremes to chase an unreachable margin
        # (severe overfitting: train loss collapses while val loss diverges).
        return self.mlp(x)
