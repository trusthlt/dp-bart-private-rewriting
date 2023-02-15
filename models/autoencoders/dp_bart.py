from models.autoencoders.general_classes import Autoencoder_Transformer, GeneralModelConfig
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math
import numpy as np
from transformers import BartForConditionalGeneration, BartConfig, BartModel
from transformers.models.bart.modeling_bart import BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput
from utils import load_neurons_for_pruning, determine_neurons_to_prune, add_neurons_to_prune, non_intersection
import pdb


class DPBartModelConfig(GeneralModelConfig):
    def __init__(self, dp_module='laplace', no_clipping=False,
                 discretize=False, **kwargs):
        super().__init__(**kwargs)
        self.dp_module = dp_module
        self.no_clipping = no_clipping
        self.discretize = discretize


class DPBart(Autoencoder_Transformer):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if config.private or not config.no_clipping:
            bart_config = BartConfig.from_pretrained(self.transformer_type)
            self.model = DPBart_Private(
                config.device, config.max_seq_len, config.private,
                config.epsilon, config.delta, config.dp_module,
                config.dp_mechanism, config.no_clipping, config.mode,
                config.clipping_constant, config.transformer_type,
                config.batch_size, config.norm_ord, config.pruning,
                config.discretize, config.pruning_index_path,
                config.experiment_output_dir, bart_config
                )
            temp_model = BartForConditionalGeneration.from_pretrained(
                config.transformer_type)
            main_model_state = self.model.state_dict()
            temp_model_state = temp_model.state_dict()
            for name, param in temp_model_state.items():
                if name not in main_model_state or main_model_state[name].shape != param.shape:
                    print(f'{name} parameter NOT copied from pretrained model')
                    continue
                if isinstance(param, Parameter):
                    param = param.data
                main_model_state[name].copy_(param)
        else:
            if config.pruning:
                raise Exception("Pruning is not implemented yet for the non-private no clipping BART model.")
            self.model = BartForConditionalGeneration.from_pretrained(
                    config.transformer_type)

    def decode(self, beam_size=3, **inputs):
        '''
        Decoding the model without teacher forcing (for use at inference time).
        '''
        if self.config.private:
            encoder_outputs = self.model.get_encoder_outputs(**inputs)

            # Combine all hidden vectors vectors for each element of the sequence into one large vector, then add noise and de-concatenate
            enc_out_last_hidden = encoder_outputs.last_hidden_state
            batch_dim = enc_out_last_hidden.shape[0]
            max_seq_len = enc_out_last_hidden.shape[1]
            hidden_dim = enc_out_last_hidden.shape[2]

            ### Prune neurons
            if self.config.pruning:
                enc_out_last_hidden[:, :, self.model.k_prune_neurons] = 0
                enc_out_last_hidden_pruned, non_pruned_indexes, smaller_hidden_dim = self.model.reduce_pruned_encoder_outputs(enc_out_last_hidden)
                # Unroll, privatize, reroll
                enc_out_last_hidden_pruned = enc_out_last_hidden_pruned.view(batch_dim, -1)
                enc_out_last_hidden_pruned = self.model.privatize(enc_out_last_hidden_pruned)
                enc_out_last_hidden_pruned = enc_out_last_hidden_pruned.view(batch_dim, self.max_seq_len, smaller_hidden_dim)
                privatized_vecs = self.model.rebuild_pruned_encoder_outputs(
                    enc_out_last_hidden_pruned, non_pruned_indexes,
                    batch_dim=batch_dim)
            ### End prune neurons
            else:
                enc_out_last_hidden = enc_out_last_hidden.view(batch_dim, -1)

                privatized_vecs = self.model.privatize(enc_out_last_hidden)
                privatized_vecs = privatized_vecs.view(batch_dim, max_seq_len,
                                                       hidden_dim)
            encoder_outputs.last_hidden_state = privatized_vecs
            # Only the `last_hidden_state` attribute of `encoder_outputs` is
            # not None
            mod_src_input = None
        else:
            encoder_outputs = self.model.model.encoder(**inputs)

            ### Prune neurons
            if self.config.pruning:
                enc_out_last_hidden = encoder_outputs.last_hidden_state
                enc_out_last_hidden[:, :, self.model.k_prune_neurons] = 0
                encoder_outputs.last_hidden_state = enc_out_last_hidden
            ### End prune neurons

            mod_src_input = inputs['input_ids']

        # With encoder_outputs, don't need to include the original input tensor
        # Instead, a BOS token is generated and decoding process continues based
        # on the provided encoder_outputs (line 428 in HF's `generation_utils.py` of version 4.16.2)
        # Due to DP's post-processing rule, can reuse the privatized
        # encoder_outputs any number of times
        outputs = self.model.generate(mod_src_input, num_beams=beam_size,
                                      max_length=self.config.max_seq_len,
                                      early_stopping=True,
                                      encoder_outputs=encoder_outputs)
        return outputs

    def forward(self, **inputs):
        '''
        inputs: input_ids and attention_mask
        return: logits (batch_size X max_seq_len-1 X vocab_size)
        '''
        if self.config.mode == 'pretrain':
            model_outputs = self.model(**inputs)  # pass through model with teacher forcing, including in private setting (standard way of training transformer-based models)
            logits = model_outputs.logits
            logits = logits[:, 1:, :].reshape(-1, logits.shape[2])
            return logits
        else:  # rewriting
            decoded_outputs = self.decode(**inputs)  # pass through model without teacher forcing, guarantees privacy when rewriting (used during rewriting and validation phase of pre-training)
            return decoded_outputs


class DPBart_Private(BartPretrainedModel):
    def __init__(self, device, max_seq_len, private, epsilon, delta, dp_module,
                 dp_mechanism, no_clipping, mode, clipping_constant,
                 transformer_type, batch_size, norm_ord, pruning, discretize,
                 pruning_index_path, exp_output_dir, config: BartConfig):
        super().__init__(config)
        self.max_seq_len = max_seq_len
        self.hidden_dim = 768  # standard hidden dim in BERT/BART

        self.private = private
        self.epsilon = epsilon
        self.delta = delta
        self.dp_module = dp_module
        self.dp_mechanism = dp_mechanism
        self.no_clipping = no_clipping
        self.mode = mode
        self.clipping_constant = clipping_constant
        self.batch_size = batch_size
        self.norm_ord = norm_ord
        self.transformer_type = transformer_type
        self.pruning = pruning
        self.pruning_index_path = pruning_index_path
        self.exp_output_dir = exp_output_dir
        self.discretize = discretize
        self.config = config

        self.model = BartModel(config)
        self.register_buffer(
                "final_logits_bias",
                torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model,
                                 self.model.shared.num_embeddings,
                                 bias=False)
        self.post_init()

        self.k_prune_neurons, self.v_prune_neurons = None, None
        if pruning:
            self._prepare_pruning(device)

    def add_neurons_to_prune(self, counter):
        out_path = self.pruning_index_path[:-3] + f'_{counter}.pt'
        self.k_prune_neurons = add_neurons_to_prune(
            self.model, self.k_prune_neurons, device=self.device,
            out_path=out_path)

    def _prepare_pruning(self, device):
        try:
            self.k_prune_neurons = load_neurons_for_pruning(
                in_path=self.pruning_index_path)
            num_pruned_neurons = self.k_prune_neurons.shape[0]
            print(f"Loaded encoder output neurons for pruning ({num_pruned_neurons} in total).")
        except FileNotFoundError:
            if self.mode == 'rewrite':
                raise Exception("If pruning in rewriting mode, please specify a valid path to load neuron indexes for pruning with `--pruning_index_path`.")
            print("Could not load encoder output neurons for pruning, computing...")
            out_path = os.path.join(self.exp_output_dir, 'k_prune_neurons_0.pt')
            print(f"Setting the pruning index path to {out_path}")
            self.pruning_index_path = out_path
            self.k_prune_neurons, _ = determine_neurons_to_prune(
                self.model, device=device,
                out_path=out_path)
            print(f"Saved indexes of pruned neurons to {out_path}")

        self.k_prune_neurons = self.k_prune_neurons.to(device)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros(
                (1, new_num_tokens - old_num_tokens),
                device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def privatize(self, hidden, no_noise=False):
        '''
        hidden: (batch_size, max_seq_len*hidden_size)
        '''
        if self.dp_module == 'clip_norm':
            clipped_tensor = self.clip(hidden)
            if no_noise:
                noised_outputs = clipped_tensor
            else:
                noised_outputs = self.add_noise(clipped_tensor)
        elif self.dp_module == 'clip_value':
            sigma = 0.1
            mean = 0.00
            num_sigmas = 1
            left_clip = mean - (sigma * num_sigmas)
            right_clip = mean + (sigma * num_sigmas)
            clipped_tensor = torch.clamp(hidden, left_clip, right_clip)
            if no_noise:
                noised_outputs = clipped_tensor
            else:
                k = hidden.shape[-1]
                if self.dp_mechanism == 'laplace':
                    sensitivity = 2 * sigma * num_sigmas * k
                    laplace = torch.distributions.laplace.Laplace(0, sensitivity / self.epsilon)
                    noise = laplace.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1])))
                elif self.dp_mechanism == 'gaussian':
                    sensitivity = 2 * sigma * num_sigmas * np.sqrt(k)
                    scale = np.sqrt((sensitivity**2 / self.epsilon**2) * 2 * np.log(1.25 / self.delta))
                    gauss = torch.distributions.normal.Normal(0, scale)
                    noise = gauss.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1])))
                else:
                    raise Exception(f"No DP mechanism available called '{self.dp_mechanism}'.")
                noise = noise.to(self.device)

                noised_outputs = clipped_tensor + noise
        else:
            raise Exception(f"No privacy module available called '{self.dp_module}'.")
        return noised_outputs

    def clip(self, hidden):
        norm = torch.linalg.norm(hidden, axis=1, ord=self.norm_ord)
        ones = torch.ones(norm.shape[0]).to(self.device)
        min_val = torch.minimum(ones, self.clipping_constant / norm)
        clipped_tensor = min_val.unsqueeze(-1) * hidden
        return clipped_tensor

    def get_sensitivity_for_clip_by_norm(self, clipped_tensor):
        if self.norm_ord == 1 and self.dp_mechanism == 'laplace':
            sensitivity = torch.tensor(2 * self.clipping_constant)
        elif self.norm_ord == 2 and self.dp_mechanism == 'laplace':
            sensitivity = 2 * self.clipping_constant * torch.sqrt(
                torch.tensor(clipped_tensor.shape[2]))
        elif self.norm_ord == 2 and self.dp_mechanism == 'gaussian':
            sensitivity = torch.tensor(2 * self.clipping_constant)
        else:
            raise Exception("Sensitivity calculation for clipping by norm only implemented for Laplace mechanism with L1/L2 norm clipping, or Gaussian mechanism with L2 norm clipping.")
        return sensitivity

    def add_noise(self, clipped_tensor):
        sensitivity = self.get_sensitivity_for_clip_by_norm(clipped_tensor)
        if self.dp_mechanism == 'laplace':
            laplace = torch.distributions.laplace.Laplace(0, sensitivity / self.epsilon)
            noise = laplace.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1])))
        elif self.dp_mechanism == 'gaussian':
            scale = torch.sqrt((sensitivity**2 / self.epsilon**2) * 2 * torch.log(torch.tensor(1.25 / self.delta)))
            gauss = torch.distributions.normal.Normal(0, scale)
            noise = gauss.sample(sample_shape=torch.Size((clipped_tensor.shape[0], clipped_tensor.shape[1])))
        noise = noise.to(self.device)

        noised_outputs = clipped_tensor + noise

        return noised_outputs

    @staticmethod
    def _reorder_cache(past, beam_idx):
        '''
        Required for generation in HF
        '''
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def get_encoder_outputs(self, **inputs):
        encoder_outputs = self.model.encoder(**inputs)
        return encoder_outputs

    def reduce_pruned_encoder_outputs(self, enc_out_last_hidden):
        # Get the non-intersection of pruned indexes and all 768 original dimensions
        all_indexes = torch.arange(768).to(self.device)
        non_pruned_indexes = non_intersection(self.k_prune_neurons,
                                              all_indexes)

        # Select the non-pruned encoder outputs to be privatized
        # (note down the hidden dimension of this smaller matrix)
        non_pruned_enc_outs = enc_out_last_hidden[:, :, non_pruned_indexes]
        smaller_hidden_dim = non_pruned_enc_outs.shape[-1]
        return non_pruned_enc_outs, non_pruned_indexes, smaller_hidden_dim


    def rebuild_pruned_encoder_outputs(self, enc_out_last_hidden_pruned,
                                       non_pruned_indexes, batch_dim=32):
        new_enc_outs = torch.zeros((batch_dim, self.max_seq_len, 768))
        new_enc_outs = new_enc_outs.to(self.device)
        new_enc_outs[:, :, non_pruned_indexes] = enc_out_last_hidden_pruned
        return new_enc_outs


    def forward(self, **inputs):
        if inputs['input_ids'] is not None:
            # For the case of generation, since encoder hidden states are
            # obtained separately and below noise in the private setting doesn't
            # need to be added a second time
            encoder_outputs = self.get_encoder_outputs(**inputs)
            inputs['encoder_outputs'] = encoder_outputs

            # Combine all hidden vectors for each element of the sequence into one large vector, then add noise and de-concatenate
            enc_out_last_hidden = inputs['encoder_outputs'].last_hidden_state
            batch_dim = enc_out_last_hidden.shape[0]

            if self.discretize:
                raise NotImplementedError

            if self.pruning:
                enc_out_last_hidden[:, :, self.k_prune_neurons] = 0

            if self.private:
                if self.pruning:
                    enc_out_last_hidden_pruned, non_pruned_indexes, smaller_hidden_dim = self.reduce_pruned_encoder_outputs(enc_out_last_hidden)
                    # Unroll, privatize, reroll
                    enc_out_last_hidden_pruned = enc_out_last_hidden_pruned.view(batch_dim, -1)
                    enc_out_last_hidden_pruned = self.privatize(enc_out_last_hidden_pruned)
                    enc_out_last_hidden_pruned = enc_out_last_hidden_pruned.view(batch_dim, self.max_seq_len, smaller_hidden_dim)
                    enc_out_last_hidden = self.rebuild_pruned_encoder_outputs(
                        enc_out_last_hidden_pruned, non_pruned_indexes,
                        batch_dim=batch_dim)
                else:
                    enc_out_last_hidden = enc_out_last_hidden.view(batch_dim, -1)
                    enc_out_last_hidden = self.privatize(enc_out_last_hidden)
                    enc_out_last_hidden = enc_out_last_hidden.view(
                        batch_dim, self.max_seq_len, self.hidden_dim)
            else:
                enc_out_last_hidden = enc_out_last_hidden.view(batch_dim, -1)
                if not self.no_clipping:
                    enc_out_last_hidden = self.privatize(enc_out_last_hidden,
                                                         no_noise=True)
                enc_out_last_hidden = enc_out_last_hidden.view(
                    batch_dim, self.max_seq_len, self.hidden_dim)

            inputs['encoder_outputs'].last_hidden_state =\
                enc_out_last_hidden

        outputs = self.model(**inputs)
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions
            )
