import torch

from rl_games.algos_torch.network_builder import A2CBuilder


class TeacherA2CBuilder(A2CBuilder):
    """A2C builder compatible with the stored teacher checkpoints.

    The teacher checkpoints in this project were trained with
    rnn.before_mlp=True and rnn.concat_input=True, where the actor MLP consumes
    the concatenation of the RNN output and the original policy observation.
    Stock rl_games 1.6.1 only applies concat_input in the !before_mlp path.
    """

    def build(self, name, **kwargs):
        return self.Network(self.params, **kwargs)

    class Network(A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            input_shape = kwargs.get("input_shape")
            super().__init__(params, **kwargs)

            self._teacher_concat_before_mlp = (
                self.has_rnn and self.is_rnn_before_mlp and self.rnn_concat_input
            )
            if not self._teacher_concat_before_mlp:
                return

            mlp_input_shape = self._calc_input_size(input_shape, self.actor_cnn)
            mlp_args = {
                "input_size": self.rnn_units + mlp_input_shape,
                "units": self.units,
                "activation": self.activation,
                "norm_func_name": self.normalization,
                "dense_func": torch.nn.Linear,
                "d2rl": self.is_d2rl,
                "norm_only_first_layer": self.norm_only_first_layer,
            }
            self.actor_mlp = self._build_mlp(**mlp_args)
            if self.separate:
                self.critic_mlp = self._build_mlp(**mlp_args)

        def forward(self, obs_dict):
            if not getattr(self, "_teacher_concat_before_mlp", False):
                return super().forward(obs_dict)

            obs = obs_dict["obs"]
            states = obs_dict.get("rnn_states", None)
            dones = obs_dict.get("dones", None)
            bptt_len = obs_dict.get("bptt_len", 0)

            if self.has_cnn and self.permute_input and len(obs.shape) == 4:
                obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = self.actor_cnn(obs)
                a_out = a_out.contiguous().view(a_out.size(0), -1)
                c_out = self.critic_cnn(obs)
                c_out = c_out.contiguous().view(c_out.size(0), -1)
                a_out_in = a_out
                c_out_in = c_out

                seq_length = obs_dict.get("seq_length", 1)
                batch_size = a_out.size(0)
                num_seqs = batch_size // seq_length
                a_out = a_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                c_out = c_out.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)

                if len(states) == 2:
                    a_states = states[0]
                    c_states = states[1]
                else:
                    a_states = states[:2]
                    c_states = states[2:]
                a_out, a_states = self.a_rnn(a_out, a_states, dones, bptt_len)
                c_out, c_states = self.c_rnn(c_out, c_states, dones, bptt_len)

                a_out = a_out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                c_out = c_out.transpose(0, 1).contiguous().reshape(batch_size, -1)
                if self.rnn_ln:
                    a_out = self.a_layer_norm(a_out)
                    c_out = self.c_layer_norm(c_out)
                if type(a_states) is not tuple:
                    a_states = (a_states,)
                    c_states = (c_states,)
                states = a_states + c_states

                a_out = self.actor_mlp(torch.cat([a_out, a_out_in], dim=1))
                c_out = self.critic_mlp(torch.cat([c_out, c_out_in], dim=1))
            else:
                out = self.actor_cnn(obs)
                out = out.flatten(1)
                out_in = out

                seq_length = obs_dict.get("seq_length", 1)
                batch_size = out.size(0)
                num_seqs = batch_size // seq_length
                out = out.reshape(num_seqs, seq_length, -1)
                if len(states) == 1:
                    states = states[0]

                out = out.transpose(0, 1)
                if dones is not None:
                    dones = dones.reshape(num_seqs, seq_length, -1).transpose(0, 1)
                out, states = self.rnn(out, states, dones, bptt_len)
                out = out.transpose(0, 1).contiguous().reshape(batch_size, -1)

                if self.rnn_ln:
                    out = self.layer_norm(out)
                out = self.actor_mlp(torch.cat([out, out_in], dim=1))
                if type(states) is not tuple:
                    states = (states,)

                value = self.value_act(self.value(out))
                if self.central_value:
                    return value, states
                if self.is_discrete:
                    logits = self.logits(out)
                    return logits, value, states
                if self.is_multi_discrete:
                    logits = [logit(out) for logit in self.logits]
                    return logits, value, states
                if self.is_continuous:
                    mu = self.mu_act(self.mu(out))
                    if self.fixed_sigma:
                        sigma = self.sigma_act(self.sigma)
                    else:
                        sigma = self.sigma_act(self.sigma(out))
                    return mu, mu * 0 + sigma, value, states

            value = self.value_act(self.value(c_out))
            if self.is_discrete:
                logits = self.logits(a_out)
                return logits, value, states
            if self.is_multi_discrete:
                logits = [logit(a_out) for logit in self.logits]
                return logits, value, states
            if self.is_continuous:
                mu = self.mu_act(self.mu(a_out))
                if self.fixed_sigma:
                    sigma = mu * 0.0 + self.sigma_act(self.sigma)
                else:
                    sigma = self.sigma_act(self.sigma(a_out))
                return mu, sigma, value, states
