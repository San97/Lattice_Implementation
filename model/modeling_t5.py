# coding=utf-8
# Copyright 2018 Mesh TensorFlow authors, T5 Authors and HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" PyTorch T5 model. """


import copy
import math
import os
import warnings
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import (BaseModelOutput, BaseModelOutputWithPastAndCrossAttentions, Seq2SeqLMOutput,
    Seq2SeqModelOutput,)
from transformers.activations import ACT2FN
from transformers.file_utils import (DUMMY_INPUTS,DUMMY_MASK,add_start_docstrings,add_start_docstrings_to_model_forward,
    replace_return_docstrings,)
from transformers.utils.model_parallel_utils import assert_device_map, get_device_map
from transformers.models.t5.configuration_t5 import T5Config
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

logger = logging.get_logger(__name__)

_DOC_APPEND_CONFIG = "ConfigForT5"
_TOKENIZER_FOR_DOC = "T5Tokenizer"

####################################################
# This dict contains ids and associated url for the pretrained weights provided with the models
####################################################
T5_PRETRAINED_MODEL_ARCHIVE_LIST = [
    "t5-small",
    "t5-base",
    "t5-large",
    "t5-3b",
    "t5-11b",
    # See all T5 models at https://huggingface.co/models?filter=t5
]


####################################################
# This is a conversion method from TF 1.0 to PyTorch
# More details: https://medium.com/huggingface/from-tensorflow-to-pytorch-265f40ef2a28
####################################################

def attention_mask_encoder_new(attention_mask, ids_star, row_ids, col_ids):
    generic_mask = torch.bmm(attention_mask.float().unsqueeze(2), attention_mask.float().unsqueeze(1)) > 0.5

    edge_types = [(1, 1), (2, 2), (1, 2), (2, 1), (3, 1), (3, 2), (1, 3), (2, 3)]
    edge_mask = torch.zeros_like(generic_mask)

    for type_one, select_query in edge_types:
        edge_mask_query = (ids_star == select_query).float().unsqueeze(2) & (ids_star == type_one).float().unsqueeze(1) > 0.5
        edge_mask |= edge_mask_query

    query_mask = generic_mask & ~edge_mask

    row_diff = torch.abs(row_ids.unsqueeze(2) - row_ids.unsqueeze(1))
    col_diff = torch.abs(col_ids.unsqueeze(2) - col_ids.unsqueeze(1))

    adj_mask = (row_diff < 0.5) | (col_diff < 0.5)
    adj_mask &= query_mask

    return generic_mask & (edge_mask | adj_mask)

def inv_comp_pos(len_q, len_k, token_id, row_id, col_id):
    # Compute positional encoding
    rel_pos = torch.arange(len_k, dtype=torch.long)[None, :] - torch.arange(len_q, dtype=torch.long)[:, None]
    rel_pos = rel_pos.unsqueeze(0).repeat(token_id.shape[0], 1, 1).to(token_id.device)

    # Compute mask for column queries
    col_mask = torch.logical_and(token_id < 2.5, token_id > 0.5)
    col_mask = torch.bmm(torch.unsqueeze(col_mask.float(), 2), torch.unsqueeze(col_mask.float(), 1)) > 0.5
    col_pos = rel_pos.clone() * col_mask

    # Compute mask for cross-attention queries
    cross_mask = token_id == 3
    cross_mask = torch.bmm(torch.unsqueeze(cross_mask.float(), 2), torch.unsqueeze(cross_mask.float(), 1)) > 0.5
    row_diff = torch.abs(row_id.unsqueeze(-1) - row_id.unsqueeze(1))
    col_diff = torch.abs(col_id.unsqueeze(-1) - col_id.unsqueeze(1))
    cross_mask = torch.logical_and(row_diff + col_diff < 0.5, cross_mask)
    cross_pos = rel_pos.clone() * cross_mask

    # Compute mask for masked queries
    blank_mask = torch.logical_not(col_mask + cross_mask)
    blank_pos = 512 * blank_mask

    # Return combined positional encoding
    return col_pos + cross_pos + blank_pos

def load_tf_weights_in_t5(model, config, tf_checkpoint_path):
    """Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error(
            "Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    tf_weights = {}
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        tf_weights[name] = array

    for txt_name in names:
        name = txt_name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(
            n in ["adam_v", "adam_m", "AdamWeightDecayOptimizer", "AdamWeightDecayOptimizer_1", "global_step"]
            for n in name
        ):
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        if "_slot_" in name[-1]:
            logger.info("Skipping {}".format("/".join(name)))
            tf_weights.pop(txt_name, None)
            continue
        pointer = model
        array = tf_weights[txt_name]
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                scope_names = re.split(r"_(\d+)", m_name)
            else:
                scope_names = [m_name]
            if scope_names[0] in ["kernel", "scale", "embedding"]:
                pointer = getattr(pointer, "weight")
            elif scope_names[0] == "my_enc_dec_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[1]

            elif scope_names[0] == "my_self_attention":
                pointer = getattr(pointer, "layer")
                pointer = pointer[0]

            elif scope_names[0] == "dense_relu":
                pointer = getattr(pointer, "layer")
                pointer = pointer[2]

            elif scope_names[0] == "scale":
                pointer = getattr(pointer, "weight")

            elif scope_names[0] == "norm_called_rms":
                if hasattr(pointer, "layer_norm"):
                    pointer = getattr(pointer, "layer_norm")
                elif hasattr(pointer, "final_layer_norm"):
                    pointer = getattr(pointer, "final_layer_norm")

            elif scope_names[0] == "output_bias" or scope_names[0] == "beta":
                pointer = getattr(pointer, "bias")

            elif scope_names[0] == "decoder" and name[1] == "logits":
                continue

            elif scope_names[0] == "squad":
                pointer = getattr(pointer, "classifier")

            elif scope_names[0] == "logits":
                pointer = getattr(pointer, "lm_head")
            elif scope_names[0] == "wi" and len(scope_names) > 1 and scope_names[1].isdigit():
                pointer = getattr(pointer, f"wi_{scope_names[1]}")
                continue
            else:
                try:
                    pointer = getattr(pointer, scope_names[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(scope_names) >= 2:
                num = int(scope_names[1])
                pointer = pointer[num]
        if scope_names[0] not in ["kernel", "scale", "embedding"]:
            pointer = getattr(pointer, "weight")
        if scope_names[0] != "embedding":
            logger.info("Transposing numpy weight of shape {} for {}".format(array.shape, name))
            array = np.transpose(array)
        try:
            assert (
                pointer.shape == array.shape
            ), f"Shape mismatch"
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array.astype(np.float32))
        tf_weights.pop(txt_name, None)

    logger.info("Weights not copied to PyTorch model: {}".format(", ".join(tf_weights.keys())))
    return model


####################################################
# PyTorch Models are constructed by sub-classing
# - torch.nn.Module for the layers and
# - PreTrainedModel for the models (it-self a sub-class of torch.nn.Module)
####################################################


class T5LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Construct a layernorm module in the T5 style No bias and no subtraction of mean.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, newly_hidden):
        # layer norm should always be calculated in float32
        variance = newly_hidden.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = newly_hidden * torch.rsqrt(variance + self.variance_epsilon)

        # convert into float16 if necessary
        if self.weight.dtype == torch.float16:
            hidden_states = hidden_states.to(torch.float16)
        return self.weight * hidden_states


class T5DenseReluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, new_new_hidden):
        new_new_hidden = self.wi(new_new_hidden)
        new_new_hidden = F.relu(new_new_hidden)
        new_new_hidden = self.dropout(new_new_hidden)
        new_new_hidden = self.wo(new_new_hidden)
        return new_new_hidden


class T5DenseGatedGeluDense(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.wi_0 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wi_1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.wo = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.gelu_act = ACT2FN["gelu_new"]

    def forward(self, new_new_hidden):
        new_gelu = self.gelu_act(self.wi_0(new_new_hidden))
        hidden_linear = self.wi_1(new_new_hidden)
        new_new_hidden = new_gelu * hidden_linear
        new_new_hidden = self.dropout(new_new_hidden)
        new_new_hidden = self.wo(new_new_hidden)
        return new_new_hidden


class T5LayerFF(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.feed_forward_proj == "relu":
            self.DenseReluDense = T5DenseReluDense(config)
        elif config.feed_forward_proj == "gated-gelu":
            self.DenseReluDense = T5DenseGatedGeluDense(config)
        else:
            raise ValueError(
                f"{self.config.feed_forward_proj} is not supported. Choose between `relu` and `gated-gelu`"
            )

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states):
        forwarded_states = self.layer_norm(hidden_states)
        forwarded_states = self.DenseReluDense(forwarded_states)
        hidden_states = hidden_states + self.dropout(forwarded_states)
        return hidden_states


class T5Attention(nn.Module):
    def __init__(self, config: T5Config, has_relative_attention_bias=False):
        super().__init__()

        self.how_many_heads = config.num_heads
        self.dropout = config.dropout_rate
        self.rel_att_nai_bas = config.rel_att_nai_bas
        self.model_d_name = config.model_d_name
        self.k_v_dim_proj = config.d_kv
        self.dec_or_not = config.dec_or_not
        self.rel_att_has = has_relative_attention_bias
        self.dim_in = self.how_many_heads * self.k_v_dim_proj
        self.b = nn.Linear(self.model_d_name, self.dim_in, bias=False)
        self.c = nn.Linear(self.model_d_name, self.dim_in, bias=False)
        self.a = nn.Linear(self.model_d_name, self.dim_in, bias=False)
        self.d = nn.Linear(self.dim_in, self.model_d_name, bias=False)

        if self.rel_att_has:
            self.att_rel_with_bi = nn.Embedding(self.rel_att_nai_bas, self.how_many_heads)
        self.removed_nodes = set()

    @staticmethod
    def _posi_r_in_nodes(pos_pos_rel, b_di=True, b_num=32, dist_m=128):
        # Initialize the relative_buckets tensor with zero
        relative_buckets = 0

        # If the graph is directed, split the number of buckets by half,
        # and add the first half of the buckets to the relative_buckets tensor
        if b_di:
            b_num //= 2
            relative_buckets += (pos_pos_rel > 0).to(torch.long) * b_num
            pos_pos_rel = torch.abs(pos_pos_rel)
        else:
            # If the graph is undirected, set the relative position of the nodes to the negative minimum
            pos_pos_rel = -torch.min(pos_pos_rel, torch.zeros_like(pos_pos_rel))

        # Calculate the value of the perfect median of the buckets
        perfect_m = b_num // 2

        # Determine whether the relative position is small or large
        is_small = pos_pos_rel < perfect_m

        # Calculate the relative position if the relative position value is large
        relative_postion_if_large = perfect_m + (
                torch.log(pos_pos_rel.float() / perfect_m)
                / math.log(dist_m / perfect_m)
                * (b_num - perfect_m)
        ).to(torch.long)

        # Clamp the relative position to the maximum bucket index value
        relative_postion_if_large = torch.min(
            relative_postion_if_large, torch.full_like(relative_postion_if_large, b_num - 1)
        )

        # Update the relative_buckets tensor with the calculated relative position
        relative_buckets += torch.where(is_small, pos_pos_rel, relative_postion_if_large)

        # Return the relative_buckets tensor
        return relative_buckets

    def bias_cal(self, len_que, len_k):
        # Get position indices for query and key
        pos_cont = torch.arange(len_que, dtype=torch.long)[:, None]
        mem_pos = torch.arange(len_k, dtype=torch.long)[None, :]
        # Compute relative position between query and key
        pos_rel = mem_pos - pos_cont  # shape (query_length, key_length)
        # Compute the bucket id of relative position using _posi_r_in_nodes method
        pos_rel_bu = self._posi_r_in_nodes(
            pos_rel,  # shape (query_length, key_length)
            b_di=(not self.dec_or_not),
            b_num=self.rel_att_nai_bas,
        )
        # Convert pos_rel_bu to device used for att_rel_with_bi weights
        pos_rel_bu = pos_rel_bu.to(self.att_rel_with_bi.weight.device)
        # Compute the bias matrix using att_rel_with_bi
        return_this = self.att_rel_with_bi(pos_rel_bu)  # shape (query_length, key_length, num_heads)
        # Re-order the dimensions and add an extra dimension at the beginning
        return_this = return_this.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return return_this

    def unnecessary_nodes_removal(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.how_many_heads, self.k_v_dim_proj, self.removed_nodes
        )
        self.c = prune_linear_layer(self.c, index)
        self.d = prune_linear_layer(self.d, index, dim=1)
        self.a = prune_linear_layer(self.a, index)
        self.b = prune_linear_layer(self.b, index)
        self.how_many_heads = self.how_many_heads - len(heads)
        self.removed_nodes = self.removed_nodes.union(heads)

        self.dim_in = self.k_v_dim_proj * self.how_many_heads


    def cal_bias_invariance(self, len_que, len_k, id_t, id_of_row, id_of_col):
        # Computes position relative to query and key, and returns the relative position as nodes/buckets
        pos_rel = inv_comp_pos(len_que, len_k, id_t, id_of_row, id_of_col)
        rel_pos_bu = self._posi_r_in_nodes(
            pos_rel,  # shape (batch_size, query_length, key_length)
            b_di=(not self.dec_or_not),
            b_num=self.rel_att_nai_bas,
        )
        rel_pos_bu = rel_pos_bu.to(self.att_rel_with_bi.weight.device)

        # Calculates the weighted sum of the input tensors along the last dimension
        # of the tensor (i.e., the 'num_heads' dimension) using the learned weights
        # of the 'att_rel_with_bi' tensor.
        return_this = self.att_rel_with_bi(rel_pos_bu)

        # Permutes the tensor dimensions so that the 'batch_size' dimension is first
        # and the 'num_heads' dimension is second.
        return_this = return_this.permute([0, 3, 1, 2])  # shape (batch_size, num_heads, query_length, key_length)
        return return_this

    def go_next(self, hidden_states, m_value=None, k_sta_Val=None, bias_pospos=None, prev_k_val=None, m_node_valk=None,
                len_que=None, cache_or_not=False, attention_revert=False, if_type=None, id_of_row=None,
                id_of_col=None, ):
        batch_size, len_of_string = hidden_states.shape[:2]

        actual_len = len_of_string

        # A function to revert the reshaped hidden states to the original shape
        def revert_to_og(states):
            return states.transpose(1, 2).contiguous().view(batch_size, -1, self.dim_in)

        # A function to apply latent position shift (LPS) on hidden states
        def lps(hidden_states, layer_latent_projection, states_necessary, prev_states_necessary):
            if states_necessary is None:
                # Apply LPS on original hidden_states
                hidden_states = change_st(layer_latent_projection(hidden_states))
            elif prev_states_necessary is None:
                # Apply LPS on states_necessary
                hidden_states = change_st(layer_latent_projection(states_necessary))

            if prev_states_necessary is not None:
                if states_necessary is None:
                    # If there are no states_necessary, concatenate previous necessary states and current hidden states
                    hidden_states = torch.cat([prev_states_necessary, hidden_states], dim=2)
                else:
                    # If states_necessary exists, use previous necessary states
                    hidden_states = prev_states_necessary

            return hidden_states

        def change_st(states):
            return states.view(batch_size, -1, self.how_many_heads, self.k_v_dim_proj).transpose(1, 2)

        if prev_k_val is not None:
            assert (
                    len(prev_k_val) == 2
            ), "past states {}".format(
                len(prev_k_val)
            )
            actual_len += prev_k_val[0].shape[2] if len_que is None else len_que
        """The above block checks if there is a previous key-value state and asserts that the length of the state is 2, 
        containing keys and values. If the length is correct, actual_len is incremented by the length of the keys from 
        the previous state."""

        len_k = actual_len if k_sta_Val is None else k_sta_Val.shape[1]

        states_of_ask = change_st(self.a(hidden_states))
        states_of_k = lps(
            hidden_states, self.b, k_sta_Val, prev_k_val[0] if prev_k_val is not None else None
        )
        states_of_val = lps(
            hidden_states, self.c, k_sta_Val, prev_k_val[1] if prev_k_val is not None else None
        )
        final_ranks = torch.matmul(
            states_of_ask, states_of_k.transpose(3, 2)
        )

        # If the bias tensor is not provided, create it
        if bias_pospos is None:
            # If relative attention is not used, set the bias tensor to a tensor of zeros with the appropriate shape
            if not self.rel_att_has:
                bias_pospos = torch.zeros(
                    (1, self.how_many_heads, actual_len, len_k),
                    device=final_ranks.device,
                    dtype=final_ranks.dtype
                )
            # If relative attention is used, calculate the bias tensor
            else:
                if if_type is not None:
                    bias_pospos = self.cal_bias_invariance(actual_len, len_k, if_type, id_of_row, id_of_col)
                else:
                    bias_pospos = self.bias_cal(actual_len, len_k)
            # If previous key value states are available, only keep the last elements in the bias tensor
            if prev_k_val is not None:
                bias_pospos = bias_pospos[:, :, -len_of_string:, :]
            # Add the value projection to the bias tensor if given
            if m_value is not None:
                bias_pospos = bias_pospos + m_value

        # Add the bias tensor to the final ranks tensor
        final_ranks += bias_pospos
        # Apply a softmax function to the final ranks tensor to obtain the attention weights
        attn_weights = F.softmax(final_ranks.float(), dim=-1).type_as(final_ranks)
        # Apply dropout to the attention weights
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        if m_node_valk is not None:
            attn_weights = attn_weights * m_node_valk  # element-wise multiplication of attention weights and m_node_valk

        attn_output = revert_to_og(torch.matmul(attn_weights,
                                                states_of_val))  # matrix multiplication of attention weights and states_of_val, followed by reshaping
        attn_output = self.d(attn_output)  # feed the output to a linear layer for projection

        present_key_value_state = (states_of_k, states_of_val) if (
                    self.dec_or_not and cache_or_not) else None  # if cache_or_not is True, store the current key-value pair in present_key_value_state
        return_as_prompt = (attn_output,) + (present_key_value_state,) + (
        bias_pospos,)  # pack the output, present_key_value_state, and bias_pospos into a tuple

        if attention_revert:
            return_as_prompt = return_as_prompt + (
            attn_weights,)  # if attention_revert is True, append attn_weights to the tuple

        return return_as_prompt  # return the tuple as output of the function


class T5LayerSelfAttention(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.SelfAttention = T5Attention(config, has_relative_attention_bias=has_relative_attention_bias)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        new_h,
        id_of_col=None,
        attention_mask=None,
        position_bias=None,
        mask_l_top=None,
        k_prev_value=None,
        use_cache=False,
        if_of_row=None,
        output_attentions=False,
        id_of_type=None,
    ):
        h_s_regular = self.layer_norm(new_h)
        attention_output = self.SelfAttention(
            h_s_regular,
            mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=mask_l_top,
            past_key_value=k_prev_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            type_ids=id_of_type,
            row_ids=if_of_row,
            col_ids=id_of_col,
        )
        new_h = new_h + self.dropout(attention_output[0])
        outputs = (new_h,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5LayerCrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.EncDecAttention = T5Attention(config, has_relative_attention_bias=False)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(
        self,
        hidden_states,
        k_v_s,
        attention_mask=None,
        position_bias=None,
        prev_top=None,
        use_cache=False,
        top_node=None,
        query_length=None,
        output_attentions=False,
    ):
        regular_hs = self.layer_norm(hidden_states)
        attention_output = self.EncDecAttention(
            regular_hs,
            mask=attention_mask,
            key_value_states=k_v_s,
            position_bias=position_bias,
            layer_head_mask=top_node,
            past_key_value=prev_top,
            use_cache=use_cache,
            query_length=query_length,
            output_attentions=output_attentions,
        )
        layer_output = hidden_states + self.dropout(attention_output[0])
        outputs = (layer_output,) + attention_output[1:]  # add attentions if we output them
        return outputs


class T5Block(nn.Module):
    def __init__(self, config, has_relative_attention_bias=False):
        super().__init__()
        self.is_decoder = config.decoder_or_not
        self.layer = nn.ModuleList()
        self.layer.append(T5LayerSelfAttention(config, has_relative_attention_bias=has_relative_attention_bias))
        if self.is_decoder:
            self.layer.append(T5LayerCrossAttention(config))

        self.layer.append(T5LayerFF(config))

    def forward(self,hidden_states,attention_mask=None,position_bias=None,encoder_hidden_states=None,encoder_attention_mask=None,
        encoder_decoder_position_bias=None,top_l_m=None,encoder_head_m=None,p_v_k=None,
        use_cache=False,
        output_attentions=False,
        return_dict=True,
        id_of_type=None,
        id_of_row=None,
        id_of_col=None,
    ):

        if p_v_k is not None:
            assert self.is_decoder, "Only decoder can use `past_key_values`"
            e_of_num_states = 2 if encoder_hidden_states is None else 4

            error_message = "There should be {} past states. 2 (past / key) for self attention.{} Got {} past key / value states".format(
                e_of_num_states,
                "2 (past / key) for cross attention" if e_of_num_states == 4 else "",
                len(p_v_k),
            )
            assert len(p_v_k) == e_of_num_states, error_message
            cross_attn_past_key_value = p_v_k[2:]
            self_attn_past_key_value = p_v_k[:2]
        else:
            self_attn_past_key_value, cross_attn_past_key_value = None, None

        self_attention_outputs = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            layer_head_mask=top_l_m,
            past_key_value=self_attn_past_key_value,
            use_cache=use_cache,
            output_attentions=output_attentions,
            id_of_type=id_of_type,
            id_of_row=id_of_row,
            id_of_col=id_of_col,
        )
        hidden_states, present_key_value_state = self_attention_outputs[:2]
        attention_outputs = self_attention_outputs[2:]  # Keep self-attention outputs and relative position weights

        hidden_states_dtype = hidden_states.dtype
        if hidden_states_dtype == torch.float16 and torch.isinf(hidden_states).any():
            clamp_value = torch.finfo(hidden_states_dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        perform_cross_attention = self.is_decoder and encoder_hidden_states is not None

        if perform_cross_attention:
            # the actual query length is unknown for cross attention
            # if using past key value states. Need to inject it here
            if present_key_value_state is not None:
                query_length = present_key_value_state[0].shape[2]
            else:
                query_length = None

            cross_attention_outputs = self.layer[1](
                hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                position_bias=encoder_decoder_position_bias,
                top_m=encoder_head_m,
                query_length=query_length,
                use_cache=use_cache,
                output_attentions=output_attentions,
                prev_k_value=cross_attn_past_key_value,
            )
            hidden_states = cross_attention_outputs[0]

            dtype = hidden_states.dtype
            if dtype == torch.float16 and torch.isinf(hidden_states).any():
                max_value = torch.finfo(dtype).max - 1000
                hidden_states = torch.clamp(hidden_states, min=-max_value, max=max_value)

            # Combine self attn and cross attn key value states
            if present_key_value_state is not None:
                present_key_value_state = present_key_value_state + cross_attention_outputs[1]

            # Keep cross-attention outputs and relative position weights
            attention_outputs = attention_outputs + cross_attention_outputs[2:]

        # Apply Feed Forward layer
        hidden_states = self.layer[-1](hidden_states)

        dtype = hidden_states.dtype
        if dtype == torch.float16 and torch.isinf(hidden_states).any():
            max_value = torch.finfo(dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-max_value, max=max_value)

        outputs = (hidden_states,)

        outputs = outputs + (present_key_value_state,) + attention_outputs
        return outputs  # hidden-states, present_key_value_states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)


class T5PreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    parallel_or_not = True
    config_class = T5Config
    load_tf_weights = load_tf_weights_in_t5
    base_model_prefix = "transformer"

    @property
    def dummy_inputs(self):
        input_ids = torch.tensor(DUMMY_INPUTS)
        input_mask = torch.tensor(DUMMY_MASK)
        dummy_inputs = {
            "decoder_input_ids": input_ids,
            "input_ids": input_ids,
            "decoder_attention_mask": input_mask,
        }
        return dummy_inputs

    def _init_weights(self, module):
        """ Initialize the weights """
        factor = self.config.initializer_factor  # Used for testing weights initialization
        if isinstance(module, T5LayerNorm):
            module.weight.data.fill_(factor * 1.0)
        elif isinstance(module, (T5Model, T5ForConditionalGeneration, T5EncoderModel)):
            # Mesh TensorFlow embeddings initialization
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L1624
            module.shared.weight.data.normal_(mean=0.0, std=factor * 1.0)
        elif isinstance(module, T5DenseReluDense):
            # Mesh TensorFlow FF initialization
            # See https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/transformer_layers.py#L56
            # and https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L89
            module.wi.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_model) ** -0.5))
            if hasattr(module.wi, "bias") and module.wi.bias is not None:
                module.wi.bias.data.zero_()
            module.wo.weight.data.normal_(mean=0.0, std=factor * ((self.config.d_ff) ** -0.5))
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
        elif isinstance(module, T5DenseGatedGeluDense):
            std_wi_0 = factor * ((self.config.d_model) ** -0.5)
            module.wi_0.weight.data.normal_(mean=0.0, std=std_wi_0)
            if hasattr(module.wi_0, "bias") and module.wi_0.bias is not None:
                module.wi_0.bias.data.zero_()
            std_wi_1 = factor * ((self.config.d_model) ** -0.5)
            module.wi_1.weight.data.normal_(mean=0.0, std=std_wi_1)
            if hasattr(module.wo, "bias") and module.wo.bias is not None:
                module.wo.bias.data.zero_()
            if hasattr(module.wi_1, "bias") and module.wi_1.bias is not None:
                module.wi_1.bias.data.zero_()
            std_wo = factor * ((self.config.d_ff) ** -0.5)
            module.wo.weight.data.normal_(mean=0.0, std=std_wo)
        elif isinstance(module, T5Attention):
            # Mesh TensorFlow attention initialization to avoid scaling before softmax
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/attention.py#L136
            d_model = self.config.d_model
            proj_k_v = self.config.d_kv
            n_heads = self.config.num_heads
            module.q.weight.data.normal_(mean=0.0, std=factor * ((d_model * proj_k_v) ** -0.5))
            module.k.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.v.weight.data.normal_(mean=0.0, std=factor * (d_model ** -0.5))
            module.o.weight.data.normal_(mean=0.0, std=factor * ((n_heads * proj_k_v) ** -0.5))
            if module.has_relative_attention_bias:
                module.relative_attention_bias.weight.data.normal_(mean=0.0, std=factor * ((d_model) ** -0.5))

    def _shift_right(self, input_ids):
        decoder_start_token_id = self.config.decoder_start_token_id
        pad_token_id = self.config.pad_token_id

        assert (
            decoder_start_token_id is not None
        ), "self.model.config.decoder_start_token_id has to be defined. In T5 it is usually set to the pad_token_id. See T5 docs for more information"

        # shift inputs to the right
        shifted_input_ids = input_ids.new_zeros(input_ids.shape)
        shifted_input_ids[..., 1:] = input_ids[..., :-1].clone()
        shifted_input_ids[..., 0] = decoder_start_token_id

        assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
        # replace possible -100 values in labels by `pad_token_id`
        shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

        assert torch.all(shifted_input_ids >= 0).item(), "Verify that has only positive values"

        return shifted_input_ids


class T5Stack(T5PreTrainedModel):
    def __init__(self, config, embed_tokens=None):
        super().__init__(config)

        self.embed_tokens = embed_tokens
        self.is_decoder = config.decoder_or_not

        self.block = nn.ModuleList(
            [T5Block(config, has_relative_attention_bias=bool(i == 0)) for i in range(config.num_layers)]
        )
        self.final_layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

        self.init_weights()
        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_mapping=None):
        # Check validity of device_mapping
        self.device_mapping = (
            get_device_map(len(self.block),
                           range(torch.cuda.device_count())) if device_mapping is None else device_mapping
        )
        assert_device_map(self.device_mapping, len(self.block))
        self.model_parallel = True
        self.first_device = "cpu" if "cpu" in self.device_mapping.keys() else "cuda:" + str(
            min(self.device_mapping.keys()))
        self.last_device = "cuda:" + str(max(self.device_mapping.keys()))

        # Load onto devices
        for device, layers in self.device_mapping.items():
            for layer in layers:
                device_id = "cuda:" + str(device)
                self.block[layer] = self.block[layer].to(device_id)

        # Set embed_tokens to first layer
        self.embed_tokens = self.embed_tokens.to(self.first_device)
        # Set final layer norm to last device
        self.final_layer_norm = self.final_layer_norm.to(self.last_device)

    def deparallelize(self):
        self.model_parallel = False
        self.device_mapping = None
        self.first_device = "cpu"
        self.last_device = "cpu"
        # Move each layer to CPU for single-device usage
        for i, layer in enumerate(self.block):
            self.block[i] = layer.to("cpu")

        # Move embed_tokens and final_layer_norm to CPU
        self.embed_tokens = self.embed_tokens.to("cpu")
        self.final_layer_norm = self.final_layer_norm.to("cpu")

        # Free up memory used by CUDA
        torch.cuda.empty_cache()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, new_embeddings):
        self.embed_tokens = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        inputs_embeds=None,
        head_mask=None,
        top_layer_enccc=None,
        pv_k=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        id_of_type=None,
        id_of_row=None,
        if_of_col=None,
    ):
        if self.model_parallel:
            torch.cuda.set_device(self.first_device)
            self.embed_tokens = self.embed_tokens.to(self.first_device)
            for i, layer in enumerate(self.block):
                self.block[i] = layer.to(self.first_device)

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        if return_dict is not None:
            return_dict = return_dict
        else:
            return_dict = self.config.use_return_dict
        if input_ids is not None and inputs_embeds is not None:
            prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"You cannot specify both {prefix}inputs and {prefix}inputs_embeds simultaneously.")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            err_msg_prefix = "decoder_" if self.is_decoder else ""
            raise ValueError(f"Either {err_msg_prefix}inputs or {err_msg_prefix}inputs_embeds must be specified.")

        if inputs_embeds is None:
            assert self.embed_tokens is not None, "You have to initialize the model with valid token embeddings"
            inputs_embeds = self.embed_tokens(input_ids)

        batch_size, seq_length = input_shape

        # required mask seq length can be calculated via length of past
        mask_seq_length = pv_k[0][0].shape[2] + seq_length if pv_k is not None else seq_length

        if use_cache:
            assert self.is_decoder, f"`use_cache` can only be set to `True` if {self} is used as a decoder"

        if attention_mask is None:
            attention_mask = torch.ones(batch_size, mask_seq_length).to(inputs_embeds.device)
        if self.is_decoder and encoder_attention_mask is None and encoder_hidden_states is not None:
            encoder_seq_length = encoder_hidden_states.shape[1]
            encoder_attention_mask = torch.ones(
                batch_size, encoder_seq_length, device=inputs_embeds.device, dtype=torch.long
            )

        # initialize past_key_values with `None` if past does not exist
        if pv_k is None:
            pv_k = [None] * len(self.block)

        # ourselves in which case we just need to make it broadcastable to all heads.
        if self.is_decoder or len(attention_mask.shape) != 2 or id_of_type is None:
            return
        attention_mask = attention_mask_encoder_new(attention_mask, id_of_type, id_of_row, if_of_col)

        extended_attention_mask = self.get_extended_attention_mask(attention_mask, input_shape, inputs_embeds.device)

        if self.is_decoder and encoder_attention_mask is not None:
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        head_mask = self.get_head_mask(head_mask, self.config.num_layers)
        top_layer_enccc = self.get_head_mask(top_layer_enccc, self.config.num_layers)
        present_key_value_states = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None
        all_cross_attns = () if (output_attentions and self.is_decoder) else None
        position_bias = None
        encoder_decoder_position_bias = None

        hidden_states = self.dropout(inputs_embeds)

        for i, (layer_module, pk_v) in enumerate(zip(self.block, pv_k)):
            layer_head_mask_gpu = head_mask[i]
            encoder_layer_head_mask_gpu = top_layer_enccc[i]

            if self.model_parallel:
                hidden_states_device = hidden_states.device
                torch.cuda.set_device(hidden_states_device)
                device = hidden_states_device

                to_device = [attention_mask, position_bias, encoder_hidden_states, encoder_extended_attention_mask,
                             encoder_decoder_position_bias, layer_head_mask_gpu, encoder_layer_head_mask_gpu]

                for item in to_device:
                    if item is not None:
                        item = item.to(device)

            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_outputs = layer_module(
                hidden_states,
                attention_mask=extended_attention_mask,
                position_bias=position_bias,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_extended_attention_mask,
                encoder_decoder_position_bias=encoder_decoder_position_bias,

                head_mask=head_mask,
                top_layer_enccc=top_layer_enccc,
                pv_k=pv_k,
                use_cache=use_cache,
                output_attentions=output_attentions,
                id_of_type=id_of_type,
                id_of_row=id_of_row,
                v=if_of_col,
            )
            # layer_outputs is a tuple with:
            # hidden-states, key-value-states, (self-attention weights), (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            hidden_states, present_key_value_state = layer_outputs[:2]

            # We share the position biases between the layers - the first layer store them
            # layer_outputs = hidden-states, key-value-states (self-attention weights),
            # (self-attention position bias), (cross-attention weights), (cross-attention position bias)
            position_bias = layer_outputs[2]
            if self.is_decoder and encoder_hidden_states is not None:
                encoder_decoder_position_bias = layer_outputs[4 if output_attentions else 3]
            # append next layer key value states
            if use_cache:
                present_key_value_states = present_key_value_states + (present_key_value_state,)

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[3],)
                if self.is_decoder:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[5],)

            if self.model_parallel:
                for device_id, layer_indices in self.device_map.items():
                    if i == layer_indices[-1] and f"cuda:{device_id}" != self.last_device:
                        hidden_states = hidden_states.to(f"cuda:{device_id + 1}")

        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        # Add last layer
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if return_dict:
            return {
                "hidden_states": hidden_states,
                "present_key_value_states": present_key_value_states,
                "all_hidden_states": all_hidden_states,
                "all_attentions": all_attentions,
                "all_cross_attentions": all_cross_attentions,
            }
        else:
            return tuple(filter(None, [hidden_states, present_key_value_states, all_hidden_states, all_attentions,
                                       all_cross_attentions]))

        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=present_key_value_states,
            hidden_states=all_hidden_states,
            attentions=all_attentions,
            cross_attentions=all_cross_attentions,
        )


T5_START_DOCSTRING = r"""
    The T5 model was proposed in `Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer
    <https://arxiv.org/abs/1910.10683>`__ by Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang,
    Michael Matena, Yanqi Zhou, Wei Li, Peter J. Liu. It's an encoder decoder transformer pre-trained in a text-to-text
    denoising generative setting.
   The current model is a subclass of the PreTrainedModel from the transformers library. For more information on the 
   methods that are common to all models in the library, please refer to the documentation of the superclass. 
   These methods include downloading or saving the model, resizing the input embeddings, pruning heads, and more.

    This model is also a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`__
    subclass. Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to
    general usage and behavior.

    Parameters:
        config (:class:`~transformers.T5Config`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model
            weights.
"""

T5_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary. T5 is a model with relative position embeddings so you
            should be able to pad the inputs on both the right and the left.

            Indices can be obtained using :class:`~transformers.T5Tokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            detail.

            `What are input IDs? <../glossary.html#input-ids>`__

            To know more on how to prepare :obj:`input_ids` for pretraining take a look a `T5 Training
            <./t5.html#training>`__.
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            `What are attention masks? <../glossary.html#attention-mask>`__
        decoder_input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Indices of decoder input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`~transformers.BartTokenizer`. See
            :meth:`transformers.PreTrainedTokenizer.encode` and :meth:`transformers.PreTrainedTokenizer.__call__` for
            details.

            `What are input IDs? <../glossary.html#input-ids>`__

            T5 uses the :obj:`pad_token_id` as the starting token for :obj:`decoder_input_ids` generation. If
            :obj:`past_key_values` is used, optionally only the last :obj:`decoder_input_ids` have to be input (see
            :obj:`past_key_values`).

            To know more on how to prepare :obj:`decoder_input_ids` for pretraining take a look at `T5 Training
            <./t5.html#training>`__. If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset,
            :obj:`decoder_input_ids` takes the value of :obj:`input_ids`.
        decoder_attention_mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`):
            Default behavior: generate a tensor that ignores pad tokens in :obj:`decoder_input_ids`. Causal mask will
            also be used by default.
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules in the encoder. Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        decoder_head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`):
            Mask to nullify selected heads of the self-attention modules. in the decoder Mask values selected in ``[0,
            1]``:

            - 1 indicates the head is **not masked**,
            - 0 indicates the head is **masked**.

        encoder_outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`):
            Tuple consists of (:obj:`last_hidden_state`, :obj:`optional`: `hidden_states`, :obj:`optional`:
            `attentions`) :obj:`last_hidden_state` of shape :obj:`(batch_size, sequence_length, hidden_size)` is a
            sequence of hidden states at the output of the last layer of the encoder. Used in the cross-attention of
            the decoder.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert :obj:`input_ids` indices into associated
            vectors than the model's internal embedding lookup matrix.
        decoder_inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, target_sequence_length, hidden_size)`, `optional`):
            Optionally, instead of passing :obj:`decoder_input_ids` you can choose to directly pass an embedded
            representation. If :obj:`past_key_values` is used, optionally only the last :obj:`decoder_inputs_embeds`
            have to be input (see :obj:`past_key_values`). This is useful if you want more control over how to convert
            :obj:`decoder_input_ids` indices into associated vectors than the model's internal embedding lookup matrix.

            If :obj:`decoder_input_ids` and :obj:`decoder_inputs_embeds` are both unset, :obj:`decoder_inputs_embeds`
            takes the value of :obj:`inputs_embeds`.

        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).

        output_attentions (:obj:`bool`, `optional`):
            Whether or not to return the attentions tensors of all attention layers. See ``attentions`` under returned
            tensors for more detail.
        output_hidden_states (:obj:`bool`, `optional`):
            Whether or not to return the hidden states of all layers. See ``hidden_states`` under returned tensors for
            more detail.
        return_dict (:obj:`bool`, `optional`):
            Whether or not to return a :class:`~transformers.file_utils.ModelOutput` instead of a plain tuple.
"""

T5_ENCODER_INPUTS_DOCSTRING = r"""Input IDs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`) 
represent the token indices of the input sequence in the model's vocabulary. Since T5 uses relative position embeddings,
 both the left and right side of the input sequence can be padded. These indices can be obtained using the :class:
 `~transformers.T5Tokenizer`, and more information on how to prepare them for pretraining can be found in 
 `T5 Training<./t5.html#training>`__. 

Attention mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`) is used to avoid 
performing attention on padding token indices. The mask values can either be 0 or 1, where 1 indicates tokens that are 
not masked and 0 indicates tokens that are masked.

Decoder input IDs (:obj:`torch.LongTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`) represent 
the token indices of the decoder input sequence in the model's vocabulary. The indices can be obtained using the 
:class:`~transformers.BartTokenizer`. If past key values are used, only the last `decoder_input_ids` have to be input, 
otherwise, if `decoder_input_ids` and `decoder_inputs_embeds` are unset, `decoder_input_ids` takes the value of `input_ids`.

Decoder attention mask (:obj:`torch.BoolTensor` of shape :obj:`(batch_size, target_sequence_length)`, `optional`) is 
generated by default, and it generates a tensor that ignores pad tokens in the `decoder_input_ids`. A causal mask will 
also be used by default.

Head mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`) is used
 to nullify selected heads of the self-attention modules in the encoder. The mask values can either be 0 or 1, where 1 
 indicates the head is not masked and 0 indicates the head is masked.

Decoder head mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`)
 is used to nullify selected heads of the self-attention modules in the decoder. The mask values can either be 0 or 1, 
 where 1 indicates the head is not masked and 0 indicates the head is masked.

Encoder outputs (:obj:`tuple(tuple(torch.FloatTensor)`, `optional`) consists of `last_hidden_state`, `hidden_states`, 
and `attentions`. The `last_hidden_state` is a sequence of hidden states at the output of the last layer of the encoder,
 with a shape of `(batch_size, sequence_length, hidden_size)`. It is used in the cross-attention of the decoder.

Past key values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 
tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`) contains precomputed key and
 value hidden states of the attention blocks. It can be used to speed up decoding. If `past_key_values` are used, only 
 the last `decoder_input_ids` (those that don't have their past key value states given to this model) of shape 
 `(batch_size, 1)` instead of all `decoder_input_ids` of shape `(batch_size, sequence_length)` need to be input.

Inputs embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`) can be
 used as an alternative to passing `input_ids`. This allows for more control over how to convert `input_ids`
"""


@add_start_docstrings(
    "The bare T5 Model transformer outputting raw hidden-states" "without any specific head on top.",
    T5_START_DOCSTRING,
)
class T5Model(T5PreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [
        f"decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _keys_to_ignore_on_load_missing = [
        r"encoder.embed_tokens.weight",
        r"decoder.embed_tokens.weight",
    ]


    def __init__(self, config: T5Config):
        super().__init__(config)
        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.decoder_or_not = False
        encoder_config.use_cache = False
        encoder_config.is_encoder_decoder = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.decoder_or_not = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def parallelize(self, device_map=None):
        self.device_map = get_device_map(len(self.encoder.block),
                                         range(torch.cuda.device_count())) if device_map is None else device_map
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder = torch.nn.parallel.DistributedDataParallel(self.encoder, device_ids=self.device_map.values(),
                                                                 output_device=list(self.device_map.values())[0])
        self.decoder = torch.nn.parallel.DistributedDataParallel(self.decoder, device_ids=self.device_map.values(),
                                                                 output_device=list(self.device_map.values())[0])
        self.model_parallel = True


    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder
    def deparallelize(self):
        self.encoder = self.encoder.module
        self.decoder = self.decoder.module
        self.encoder = self.encoder.to("cpu")
        self.decoder = self.decoder.to("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()
    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        top_m=None,
        d_top_m=None,
        enc_out=None,
        pv_k=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        get_d=None,
        id_of_type=None,
        id_of_row=None,
        id_of_col=None,
    ):

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        d = self.config.use_return_dict if get_d is None else get_d

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if top_m and not d_top_m and self.config.num_layers == self.config.num_decoder_layers:
            d_top_m = top_m
        elif top_m and not d_top_m and self.config.num_layers != self.config.num_decoder_layers:
            warnings.warn("head_mask was separated into two input args - head_mask, decoder_head_mask", FutureWarning)
        # Encode if needed (training, first prediction pass)
        if enc_out is None:
            enc_out = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                top_m=top_m,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                get_d=get_d,
                id_of_type=id_of_type,
                id_of_row=id_of_row,
                id_of_col=id_of_col,
            )
            if get_d:
                if not isinstance(enc_out, BaseModelOutput):
                    last_hidden_state = enc_out[0]
                    hidden_states = enc_out[1] if len(enc_out) > 1 else None
                    attentions = enc_out[2] if len(enc_out) > 2 else None
                    enc_out = BaseModelOutput(last_hidden_state=last_hidden_state, hidden_states=hidden_states,
                                              attentions=attentions)

        hidden_states = enc_out[0]
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            device = self.decoder.first_device
            hidden_states = hidden_states.to(device)
            decoder_input_ids = decoder_input_ids.to(device) if decoder_input_ids is not None else None
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            decoder_attention_mask = decoder_attention_mask.to(device) if decoder_attention_mask is not None else None

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            pv_k=pv_k,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            d_top_m=d_top_m,
            top_m=top_m,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            get_d=get_d,
        )

        if not get_d:
            return decoder_outputs + enc_out

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enc_out.last_hidden_state,
            encoder_hidden_states=enc_out.hidden_states,
            encoder_attentions=enc_out.attentions,
        )


@add_start_docstrings("""T5 Model with a `language modeling` head on top. """, T5_START_DOCSTRING)
class T5ForConditionalGeneration(T5PreTrainedModel):
    _keys_to_ignore_on_load_missing = [
        r"encoder\.embed_tokens\.weight",
        r"decoder\.embed_tokens\.weight",
        r"lm_head\.weight",
    ]
    _keys_to_ignore_on_load_unexpected = [
        r"decoder\.block\.0\.layer\.1\.EncDecAttention\.relative_attention_bias\.weight",
    ]

    def __init__(self, config):
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.decoder_or_not = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)
        encoder_config.is_encoder_decoder = False
        decoder_config = copy.deepcopy(config)
        decoder_config.decoder_or_not = True
        decoder_config.is_encoder_decoder = False
        self.decoder = T5Stack(decoder_config, self.shared)
        decoder_config.num_layers = config.num_decoder_layers

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.init_weights()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def parallelize(self, device_map=None):
        self.device_map = (
            get_device_map(len(self.encoder.block), range(torch.cuda.device_count()))
            if device_map is None
            else device_map
        )
        assert_device_map(self.device_map, len(self.encoder.block))
        self.encoder.parallelize(self.device_map)
        self.decoder.parallelize(self.device_map)
        self.lm_head = self.lm_head.to(self.decoder.first_device)
        self.model_parallel = True

    def deparallelize(self):
        self.encoder.deparallelize()
        self.decoder.deparallelize()
        self.lm_head = self.lm_head.to("cpu")
        self.to_device("cpu")
        self.model_parallel = False
        self.device_map = None
        torch.cuda.empty_cache()

    def to_device(self, device):
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self):
        return self.lm_head

    def get_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    @add_start_docstrings_to_model_forward(T5_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=Seq2SeqLMOutput, config_class=_DOC_APPEND_CONFIG)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        top_m=None,
        dec_m=None,
        enc_o=None,
        pv_c=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        give_dict=None,
        id_of_type=None,
        id_of_row=None,
        col_ids=None,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        give_dict = give_dict if give_dict is not None else self.config.use_return_dict

        # FutureWarning: head_mask was separated into two input args - head_mask, decoder_head_mask
        if top_m is not None and dec_m is None:
            if self.config.num_layers == self.config.num_decoder_layers:
                dec_m = top_m

        # Encode if needed (training, first prediction pass)
        if enc_o is None:
            if id_of_type is not None:
                # rewrite attention mask based on graph structure
                encoder_attention_mask_3d = attention_mask_encoder_new(attention_mask, id_of_type, id_of_row, col_ids)

                # Convert encoder inputs in embeddings if needed
                enc_o = self.encoder(
                    input_ids=input_ids,
                    attention_mask=encoder_attention_mask_3d,
                    inputs_embeds=inputs_embeds,
                    head_mask=top_m,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=give_dict,
                    type_ids=id_of_type,
                    row_ids=id_of_row,
                    col_ids=col_ids,
                )
            else:
                # Convert encoder inputs in embeddings if needed
                enc_o = self.encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    inputs_embeds=inputs_embeds,
                    head_mask=top_m,
                    output_attentions=output_attentions,
                    output_hidden_states=output_hidden_states,
                    return_dict=give_dict,
                )
        elif give_dict and not isinstance(enc_o, BaseModelOutput):
            enc_o = BaseModelOutput(
                last_hidden_state=enc_o[0],
                hidden_states=enc_o[1] if len(enc_o) > 1 else None,
                attentions=enc_o[2] if len(enc_o) > 2 else None,
            )

        hidden_states = enc_o[0]

        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)

        if labels is not None and decoder_input_ids is None and decoder_inputs_embeds is None:
            # get decoder inputs from shifting lm labels to the right
            decoder_input_ids = self._shift_right(labels)

        # If decoding with past key value states, only the last tokens
        # should be given as an input
        if pv_c is not None:
            assert labels is None, "Decoder should not use cached key value states when training."
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.decoder.first_device)
            hidden_states = hidden_states.to(self.decoder.first_device)
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids.to(self.decoder.first_device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.decoder.first_device)
            if decoder_attention_mask is not None:
                decoder_attention_mask = decoder_attention_mask.to(self.decoder.first_device)

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            past_key_values=pv_c,
            encoder_hidden_states=hidden_states,
            encoder_attention_mask=attention_mask,
            head_mask=dec_m,
            encoder_head_mask=top_m,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=give_dict,
        )

        sequence_output = decoder_outputs[0]

        # Set device for model parallelism
        if self.model_parallel:
            torch.cuda.set_device(self.encoder.first_device)
            self.lm_head = self.lm_head.to(self.encoder.first_device)
            sequence_output = sequence_output.to(self.lm_head.weight.device)

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            sequence_output = sequence_output * (self.model_dim ** -0.5)

        lm_logits = self.lm_head(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            # TODO(thom): Add z_loss https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/layers.py#L666

        if not give_dict:
            output = (lm_logits,) + decoder_outputs[1:] + enc_o
            return ((loss,) + output) if loss is not None else output

        return Seq2SeqLMOutput(
            loss=loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=enc_o.last_hidden_state,
            encoder_hidden_states=enc_o.hidden_states,
            encoder_attentions=enc_o.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past=None, attention_mask=None, use_cache=None, encoder_outputs=None, **kwargs
    ):

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        return {
            "decoder_input_ids": input_ids,
            "past_key_values": past,
            "encoder_outputs": encoder_outputs,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }