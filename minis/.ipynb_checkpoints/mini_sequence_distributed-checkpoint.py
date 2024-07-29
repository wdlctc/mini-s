import math
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
)


import transformers

import torch.distributed as dist

from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Generator,
    Iterator,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

def _all_to_all(
    input_: torch.Tensor,
    world_size: int,
    group: dist.ProcessGroup,
    scatter_dim: int,
    gather_dim: int,
):
    input_list = [t.contiguous() for t in torch.tensor_split(input_, world_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(world_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()
    
class _AllToAll(torch.autograd.Function):
    """All-to-all communication.

    Args:
        input_: input matrix
        process_group: communication group
        scatter_dim: scatter dimension
        gather_dim: gather dimension
    """

    @staticmethod
    def forward(ctx, input_, process_group, scatter_dim, gather_dim):
        ctx.process_group = process_group
        ctx.scatter_dim = scatter_dim
        ctx.gather_dim = gather_dim
        ctx.world_size = dist.get_world_size(process_group)
        output = _all_to_all(input_, ctx.world_size, process_group, scatter_dim, gather_dim)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = _all_to_all(
            grad_output,
            ctx.world_size,
            ctx.process_group,
            ctx.gather_dim,
            ctx.scatter_dim,
        )
        return (
            grad_output,
            None,
            None,
            None,
        )

def all_to_all(
    input_: torch.Tensor,
    process_group: dist.ProcessGroup,
    scatter_dim: int = 2,
    gather_dim: int = 1,
):
    return _AllToAll.apply(input_, process_group, scatter_dim, gather_dim)

class LlamaMLPWarpper(nn.Module):
    def __init__(
        self,
        module,
        mini_s = 8,
        chunk_size = 4096,
        chunk_mode = True
    ):
        super().__init__()
        self.module = module
        self.mini_s = mini_s
        self.chunk_size = chunk_size
        self.chunk_mode = chunk_mode
        
    def forward(self, x):

        bsz, q_len, _ = x.size()

        if self.chunk_mode:
            chunk_size = self.chunk_size
        else:
            chunk_size = math.ceil(q_len / self.mini_s)
        
        x_list = list(x.split(chunk_size, dim=1))
            

        output_list = [None for _ in range(len(x_list))]

        for i in range(len(x_list)):
            output = self.module(x_list[i])
            output_list[i] = output
            
        down_proj = torch.cat(output_list, dim=1)

        return down_proj  

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state dict of the wrapped module
        module_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        
        # Create a new state dict without the 'module.' prefix
        new_state_dict = {k: v for k, v in module_state_dict.items()}
        
        return new_state_dict



import torch.nn.functional as F

class _cross_entropy(torch.autograd.Function):

    @classmethod
    def forward(cls, ctx, hidden_states, indices, weights):
        logits = F.linear(hidden_states, weights).float()
        loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
        loss_i = loss_fct(logits, indices)

        ignore_index = -100
        mask = indices != ignore_index
        reverse_mask = indices == ignore_index
        
        batch_size = torch.sum(indices != ignore_index)

        grad_input = F.softmax(logits, dim=-1)
        grad_input[mask, indices[mask]] -= 1
        grad_input[reverse_mask] = 0
        grad_input = grad_input.to(hidden_states.dtype)
        if hasattr(weights, 'grad') and weights.grad != None:
            torch.addmm(
                    weights.grad,
                    grad_input.T,
                    hidden_states,
                    out=weights.grad,
                )
        else:
            weights.grad = grad_input.T @ hidden_states
            
        grad_input = grad_input @ weights

        weights.grad_mul = False
        
        ctx.save_for_backward(grad_input, weights)
        
        return loss_i

    @classmethod
    def backward(cls, ctx, dneg_logprobs):
        """We know d(-log(p[i])/dlogit[k] = -id_mat[i,k] + p[k]
        so we initialize the gradient as neg_logprobs, so we can just exponentiate
        to get p[k], which is most of what we need...  neg_logprobs will be
        modified in place to become the gradient we want
        """
        # load saved tensors
        grad_input, weights = ctx.saved_tensors
        # dneg_logprobs = dneg_logprobs / weights.mul
        if weights.grad_mul is False:
            weights.grad *= dneg_logprobs
            weights.grad_mul = True
        grad_input *= dneg_logprobs
        
        return grad_input, None, None


class FusedCrossEntropyLMhead(nn.Module):
    def __init__(
        self,
        original_weight = None
    ):
        super().__init__()
        if original_weight is None:
            self.LM_head_weight = nn.Parameter(torch.empty(hidden_size, vocab_size))
        else:
            self.LM_head_weight = original_weight
        self.cross_entropy = _cross_entropy.apply

    def forward(self, hidden_states, labels):
        ignore_index = -100
        loss = self.cross_entropy(hidden_states, labels, self.LM_head_weight)
        return loss
        
class LlamaForCausalLMWarpper(nn.Module):
    def __init__(
        self,
        module,
        mini_s = 16
    ):
        super().__init__()
        self.model = module.model
        self.config = module.config
        self.vocab_size = module.vocab_size
        self.lm_head = module.lm_head
        self.mini_s = mini_s
        
        self.group = torch.distributed.distributed_c10d._get_default_group()
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
    def narrow_processing(self, hidden_states, labels):

        bsz, q_len, hidden_size = hidden_states.size()
        tmp = q_len // self.mini_s

        if labels is None:
            hidden_states = hidden_states[..., -1:, :]
            logits = self.lm_head(hidden_states)
            logits = logits.float()
            return logits, None

        if self.rank == self.world_size - 1:
            hidden_states = hidden_states[..., :-1, :]
        
        labels = labels.to(hidden_states.device)

        Fused = FusedCrossEntropyLMhead(self.lm_head.weight)
        
        loss = None
        for i in range(self.mini_s):


            shift_hidden_states = hidden_states[..., i * tmp : (i+1)*tmp, :].contiguous()
            shift_hidden_states = shift_hidden_states.view(-1, hidden_size)
            shift_labels = labels[..., i * tmp : (i+1)*tmp ].contiguous()
            shift_labels = shift_labels.view(-1)

            loss_i = Fused(shift_hidden_states, shift_labels)

            if not torch.isnan(loss_i):
                if loss is None:
                    loss = loss_i
                else:
                    loss = loss + loss_i

        if torch.is_nonzero(loss):
            loss = loss / torch.sum(torch.ne(labels, -100))


        return None, loss

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from transformers import AutoTokenizer, LlamaForCausalLM

        >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

        >>> prompt = "Hey, are you conscious? Can you talk to me?"
        >>> inputs = tokenizer(prompt, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
        ```"""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        logits, loss = self.narrow_processing(hidden_states, labels)
        dist.all_reduce(loss)
        loss = loss / self.world_size
                
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
    def save_pretrained(self, *args, **kwargs):
        # Check if the module has a save_pretrained method
        self.model.save_pretrained(*args, **kwargs)


class LinearWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
        pre = False,
    ):
        super().__init__()
        self.module = module
        self.group = group
        self.pre = pre
        
    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:

        inputs = args[0]
    
        if self.pre:
            inputs = all_to_all(inputs, self.group, scatter_dim=1, gather_dim=-1)
        outputs = self.module(inputs)
        if not self.pre:
            outputs = all_to_all(outputs, self.group, scatter_dim=-1, gather_dim=1)
        return outputs

class LlamaAttentionWarpper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        group: Optional[Any] = None,
    ):
        super().__init__()
        self.module = module
        if group:
            self.group = group
        else:
            self.group = torch.distributed.distributed_c10d._get_default_group()
        self.world_size = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        
        self.module.num_heads = module.num_heads // self.world_size
        self.module.num_key_value_heads = module.num_key_value_heads // self.world_size
        self.module.hidden_size = module.hidden_size // self.world_size

        self.replace_linear('q_proj', self.module.q_proj)
        self.replace_linear('k_proj', self.module.k_proj)
        self.replace_linear('v_proj', self.module.v_proj)
        self.replace_linear('o_proj', self.module.o_proj, pre=True)

    def replace_linear(self, name, module, pre=False):
        module = LinearWarpper(module, self.group, pre)
        setattr(self.module, name, module)
        
    def forward(self, *args, **kwargs) -> torch.Tensor:
        outputs = self.module(*args, **kwargs)
        return outputs
       
class minisequence(nn.Module):
    def __init__(
        self,
        module):
        super().__init__()
        
        self.module = module
        
        self.RecursiveVisit('module', self.module, self)

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
        self.module.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        
    def RecursiveVisit(self, name, module, upper_module):

        
        has_parameters = any(isinstance(param, nn.Parameter) for param in module.parameters())
        has_child = any(isinstance(child, nn.Module) for child in module.children())
        is_LlamaMLP = isinstance(module, transformers.models.llama.modeling_llama.LlamaMLP)
        is_LlamaForCausalLM = isinstance(module, transformers.models.llama.modeling_llama.LlamaForCausalLM)
        is_LlamaAttention = isinstance(module, transformers.models.llama.modeling_llama.LlamaAttention)


        if has_child and not is_LlamaMLP:
            for n, child in module.named_children():
                self.RecursiveVisit(n, child, module)

        
        if is_LlamaMLP:
            module = LlamaMLPWarpper(module)
            setattr(upper_module, name, module)
            
        if is_LlamaForCausalLM:
            module = LlamaForCausalLMWarpper(module)
            setattr(upper_module, name, module)
        if is_LlamaAttention:
            module = LlamaAttentionWarpper(module)
            setattr(upper_module, name, module)

    def forward(self, *args, **kwargs):
        outputs = self.module(*args, **kwargs)
        return outputs

    def state_dict(self, destination=None, prefix='', keep_vars=False):
        # Get the state dict of the wrapped module
        module_state_dict = self.module.state_dict(destination, prefix, keep_vars)
        
        # Create a new state dict without the 'module.' prefix
        new_state_dict = {k: v for k, v in module_state_dict.items()}
        
        return new_state_dict

    def save_pretrained(self, *args, **kwargs):
        # Check if the module has a save_pretrained method
        self.module.save_pretrained(*args, **kwargs)