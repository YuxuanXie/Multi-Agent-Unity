# Model taken from https://arxiv.org/pdf/1810.08647.pdf,
# INTRINSIC SOCIAL MOTIVATION VIA CAUSAL
# INFLUENCE IN MULTI-AGENT RL


# model is a single convolutional layer with a kernel of size 3, stride of size 1, and 6 output
# channels. This is connected to two fully connected layers of size 32 each

import imp
import math
from pdb import set_trace
import numpy as np

from ray.rllib.models.modelv2 import ModelV2
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork as TorchRNN
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch

torch, nn = try_import_torch()
import torch.nn.functional as F

class TorchRNNModel(TorchRNN, nn.Module):
    def __init__(self,
                obs_space,
                action_space,
                num_outputs,
                model_config,
                name,
                fc_size=None,
                lstm_state_size=128):
        nn.Module.__init__(self)
        super().__init__(obs_space, action_space, num_outputs, model_config,
                        name)


        # Holds the current "base" output (before logits layer).
        self._features = None

        self.embeding_size = fc_size if fc_size else [256, 128, 32]
        self.rnn_size = lstm_state_size
        # custom_model_config = model_config["custom_model_config"]

        self.conv = torch.nn.Conv2d(6, 1, 3, stride=2)
        self.fc1 = torch.nn.Linear(81, self.embeding_size[0])
        self.fc2 = torch.nn.Linear(self.embeding_size[0], self.embeding_size[1])
        self.lstm = torch.nn.LSTM(input_size=self.embeding_size[1], hidden_size=self.rnn_size, batch_first=True)
        self.fc3 = torch.nn.Linear(self.rnn_size, self.embeding_size[2])

        self.logits = torch.nn.Linear(self.embeding_size[2], num_outputs)
        self.values = torch.nn.Linear(self.embeding_size[2], 1)

    @override(ModelV2)
    def get_initial_state(self):
        # TODO: (sven): Get rid of `get_initial_state` once Trajectory
        #  View API is supported across all of RLlib.
        # Place hidden states on same device as model.
        h = [
            self.conv.weight.new(1, self.rnn_size).zero_().squeeze(0),
            self.conv.weight.new(1, self.rnn_size).zero_().squeeze(0)
        ]
        return h

    @override(ModelV2)
    def value_function(self):
        assert self._features is not None, "must call forward() first"
        return torch.reshape(self.values(self._features), [-1])

    @override(TorchRNN)
    def forward_rnn(self, inputs, state, seq_lens):
        """Feeds `inputs` (B x T x ..) through the Gru Unit.

        Returns the resulting outputs as a sequence (B x T x ...).
        Values are stored in self._cur_value in simple (B) shape (where B
        contains both the B and T dims!).

        Returns:
            NN Outputs (B x T x ...) as sequence.
            The state batches as a List of two items (c- and h-states).
        """
        bs = inputs.shape[0]
        seq = inputs.shape[1]
        inputs = inputs.reshape(-1, self.obs_space.shape[2], *self.obs_space.shape[:2])
        inputs = self.conv(inputs)
        embeddings = F.relu(self.fc1(inputs.reshape(bs*seq, -1)))
        embeddings = F.relu(self.fc2(embeddings)).reshape(bs, seq, -1)
        core, [h,c] = self.lstm(embeddings, [torch.unsqueeze(state[0], 0), torch.unsqueeze(state[1], 0)])
        self._features = self.fc3(core)
        logits = self.logits(self._features)
        return logits, [torch.squeeze(h, 0), torch.squeeze(c, 0)]




