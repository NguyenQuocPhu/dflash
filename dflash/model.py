import time
import torch
from types import SimpleNamespace
from typing import Callable, Optional
from typing_extensions import Unpack
from torch import nn
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3RMSNorm,
    Qwen3RotaryEmbedding,
    Qwen3Config,
    Qwen3PreTrainedModel,
    Qwen3MLP,
    GradientCheckpointingLayer,
    FlashAttentionKwargs,
    rotate_half,
    eager_attention_forward,
    ALL_ATTENTION_FUNCTIONS,
)
from transformers import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.cache_utils import Cache

# ---------------------------------------------------------------------------
# Model utilities
# ---------------------------------------------------------------------------

def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    """
    Quyết định các layer của Target Model (Sếp) mà Draft Model (Đệ) sẽ được phép trích xuất thông tin (Hidden states).
    - Nếu Đệ chỉ có 1 layer: Lấy đúng layer ở giữa của Sếp.
    - Nếu Đệ có nhiều layer: Trải đều (nội suy tuyến tính) từ layer số 1 đến áp chót của Sếp.
    Mục đích: Giúp Đệ nhận được đa dạng ngữ nghĩa từ nông đến sâu của Sếp.
    """
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: list[torch.Tensor],
    layer_ids: Optional[list[int]],
) -> torch.Tensor:
    """
    Hàm "Rút phao cứu sinh" từ Sếp.
    Lấy toàn bộ trạng thái ẩn (hidden_states) sinh ra từ Sếp, lọc ra các layer chỉ định (layer_ids),
    và ghép (concatenate) chúng lại dọc theo chiều cuối cùng (dim=-1) thành một Tensor duy nhất.
    Tensor này chính là 'target_hidden' sẽ được truyền cho Đệ.
    """
    offset = 1
    selected_states = [hidden_states[layer_id + offset] for layer_id in layer_ids]
    return torch.cat(selected_states, dim=-1)


def sample(logits: torch.Tensor, temperature: float = 0.0) -> torch.Tensor:
    if temperature < 1e-5:
        return torch.argmax(logits, dim=-1)
    bsz, seq_len, vocab_size = logits.shape
    logits = logits.view(-1, vocab_size) / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).view(bsz, seq_len)


def _cuda_time() -> float:
    torch.cuda.synchronize()
    return time.perf_counter()


@torch.inference_mode()
def dflash_generate(
    model: "DFlashDraftModel",
    target: nn.Module,
    input_ids: torch.LongTensor,
    max_new_tokens: int,
    stop_token_ids: Optional[list[int]],
    temperature: float,
    block_size: Optional[int] = None,
    mask_token_id: Optional[int] = None,
    return_stats: bool = False,
):
    """
    Trái tim của DFlash: Vòng lặp Speculative Decoding (Đoán -> Kiểm duyệt -> Chốt).
    """
    # num_input_tokens: Độ dài câu hỏi ban đầu của user
    num_input_tokens = input_ids.shape[1]
    # max_length: Tổng chiều dài tối đa (câu hỏi + câu trả lời)
    max_length = num_input_tokens + max_new_tokens
    block_size = model.block_size if block_size is None else block_size
    mask_token_id = model.mask_token_id if mask_token_id is None else mask_token_id

    # output_ids: Một mảng trống khổng lồ dùng để điền dần các từ được sinh ra vào.
    output_ids = torch.full(
        (1, max_length + block_size), mask_token_id, dtype=torch.long, device=target.device,
    )
    # position_ids: Mảng đánh số thứ tự (0, 1, 2, 3...) để model nhận biết vị trí của từng từ trong câu.
    position_ids = torch.arange(output_ids.shape[1], device=target.device).unsqueeze(0)
    
    # past_key_values: Chính là KV Cache (Bộ nhớ đệm). Lưu lại lịch sử tính toán của các chữ cũ để khỏi tính lại.
    # Ta cần 2 bộ nhớ riêng biệt cho Sếp (target) và Đệ (draft).
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = _cuda_time() if return_stats else None
    
    # 1. PREFILL PHASE: Cho ông Sếp (target) đọc toàn bộ câu hỏi (input_ids) trước.
    # Mục đích là lấy từ đầu tiên và quan trọng nhất là lấy trạng thái ẩn (target_hidden) làm 'phao' đầu tiên.
    output = target(
        input_ids, # Đầu vào là câu hỏi
        position_ids=position_ids[:, :num_input_tokens], # Vị trí của câu hỏi
        past_key_values=past_key_values_target, # Lưu kết quả tính toán vào bộ nhớ đệm của Sếp
        use_cache=True,
        logits_to_keep=1, # Chỉ giữ lại bảng xác suất (logits) của chữ cuối cùng để tiết kiệm RAM
        output_hidden_states=block_size > 1, # Yêu cầu Sếp phải nôn ra trạng thái ẩn (hidden states) để làm phao
    )

    output_ids[:, :num_input_tokens] = input_ids
    # Lấy chữ đầu tiên do Sếp rặn ra điền vào mảng kết quả
    output_ids[:, num_input_tokens:num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
    time_to_first_token = _cuda_time() - prefill_start if return_stats else None

    decode_start = _cuda_time() if return_stats else None
    acceptance_lengths = []
    start = num_input_tokens
    draft_prefill = True

    # 2. VÒNG LẶP CHÍNH (DECODE PHASE)
    while start < max_length:
        # block_output_ids: Khung chứa tạm thời cho cụm từ sắp được nháp (độ dài = block_size)
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            # 2a. DRAFTING: Thằng Đệ sinh ra một đoạn nháp dài 'block_size' (ví dụ 16 từ).
            # Lưu ý nó truyền 'target_hidden' (phao của Sếp) vào để tính toán.
            
            # noise_embedding: Biến các ID (mã số) của cụm từ nháp thành các vector toán học (embedding)
            noise_embedding = target.model.embed_tokens(block_output_ids)
            
            # draft_logits: Xác suất (tỷ lệ phần trăm) do thằng Đệ dự đoán cho các chữ tiếp theo
            draft_logits = target.lm_head(model(
                target_hidden=target_hidden, # Phao của Sếp
                noise_embedding=noise_embedding, # Vector của từ nháp
                position_ids=position_ids[:, past_key_values_draft.get_seq_length(): start + block_size],
                past_key_values=past_key_values_draft, # Lưu lịch sử vào bộ nhớ đệm của Đệ
                use_cache=True,
                is_causal=False,
            )[:, 1 - block_size :, :])
            past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill and return_stats:
                draft_prefill = False
                decode_start = _cuda_time()

        # 2b. VERIFYING: Ném nguyên cụm 16 chữ nháp đó cho Sếp (target) chạy kiểm tra song song (1 lượt forward).
        output = target(
            block_output_ids, # Input bây giờ là 16 chữ nháp của Đệ
            position_ids=block_position_ids,
            past_key_values=past_key_values_target, # Tiếp tục lưu lịch sử vào bộ nhớ đệm của Sếp
            use_cache=True,
            output_hidden_states=block_size > 1, # Lại yêu cầu Sếp nôn ra hidden states để chốc nữa làm phao mới
        )

        # posterior: Đáp án chuẩn (xác suất các chữ tiếp theo) do Sếp tự tính toán ra. Dùng để đối chiếu.
        posterior = sample(output.logits, temperature)
        
        # 2c. ACCEPTANCE: Trọng tài đối chiếu từng chữ. 
        # Khớp đến đoạn nào thì chốt chiều dài đến đoạn đó (acceptance_length).
        acceptance_length = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)[0].item()
        
        # Cập nhật kết quả cuối cùng: Lấy mảng nháp đã duyệt thành công + thêm 1 chữ do Sếp sửa/chốt.
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[:, : acceptance_length + 1]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]
        start += acceptance_length + 1
        past_key_values_target.crop(start)
        acceptance_lengths.append(acceptance_length + 1)

        if block_size > 1:
            # 2d. CẬP NHẬT PHAO: Rút trích target_hidden mới từ những chữ vừa duyệt thành công để dùng cho vòng lặp sau.
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[:, :acceptance_length + 1, :]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:] for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :min(start + 1, max_length)]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(output_ids[0][num_input_tokens:], stop_token_ids).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    if not return_stats:
        return output_ids

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = _cuda_time() - decode_start
    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=total_decode_time / num_output_tokens,
        acceptance_lengths=acceptance_lengths,
    )


# ---------------------------------------------------------------------------
# DFlash model
# ---------------------------------------------------------------------------

def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_len = q.size(-2)
    q_embed = (q * cos[..., -q_len:, :]) + (rotate_half(q) * sin[..., -q_len:, :])
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = False
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len = hidden_states.shape[:-1]
        ctx_len = target_hidden.shape[1]
        
        # Truy vấn (Query) của thằng Đệ: Câu hỏi hiện tại nó đang muốn tìm ngữ cảnh.
        q = self.q_proj(hidden_states)
        q = q.view(bsz, q_len, -1, self.head_dim)
        q = self.q_norm(q).transpose(1, 2)
        
        # BÍ THUẬT CỦA DFLASH Ở ĐÂY:
        # Tính K (Key - Từ khóa), V (Value - Nội dung) từ 'target_hidden' (phao của Sếp)
        k_ctx = self.k_proj(target_hidden)
        v_ctx = self.v_proj(target_hidden)
        
        # Tính K, V từ 'hidden_states' (dữ liệu hiện tại của chính Đệ)
        k_noise = self.k_proj(hidden_states)
        v_noise = self.v_proj(hidden_states)
        
        # Ghép (Cat) K và V của cả Sếp và Đệ lại với nhau.
        # Từ nay thằng Đệ sẽ tra cứu thông tin (Attention) trên cả 2 nguồn này cùng lúc (Cross-Attention).
        k = torch.cat([k_ctx, k_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        v = torch.cat([v_ctx, v_noise], dim=1).view(bsz, ctx_len + q_len, -1, self.head_dim)
        
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)
        cos, sin = position_embeddings
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            k, v = past_key_values.update(k, v, self.layer_idx, cache_kwargs)
        attn_fn: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attn_fn = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        attn_output, attn_weights = attn_fn(
            self,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        attn_output = attn_output.reshape(bsz, q_len, -1)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class Qwen3DFlashDecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Qwen3DFlashAttention(config=config, layer_idx=layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        target_hidden: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.FloatTensor, Optional[tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )[0]
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(Qwen3PreTrainedModel):
    config_class = Qwen3Config
    _no_split_modules = ["Qwen3DFlashDecoderLayer"]

    def __init__(self, config) -> None:
        super().__init__(config)
        self.config = config
        self.layers = nn.ModuleList(
            [Qwen3DFlashDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.target_layer_ids = self.config.dflash_config.get(
            "target_layer_ids", build_target_layer_ids(config.num_target_layers, config.num_hidden_layers)
        )
        self.norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)
        
        # Lớp FC (Fully Connected) này dùng để ép (nén) cái 'target_hidden' khổng lồ 
        # (do bị ghép từ nhiều layer của Sếp) về lại đúng kích thước não (hidden_size) của Đệ.
        self.fc = nn.Linear(len(self.target_layer_ids) * config.hidden_size, config.hidden_size, bias=False)
        self.hidden_norm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = self.config.dflash_config.get("mask_token_id", None)
        self.post_init()

    def forward(
        self,
        position_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        noise_embedding: Optional[torch.Tensor] = None,
        target_hidden: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: bool = False,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                use_cache=use_cache,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        return self.norm(hidden_states)

    @torch.inference_mode()
    def spec_generate(
        self,
        target: nn.Module,
        input_ids: torch.LongTensor,
        max_new_tokens: int,
        stop_token_ids: list[int],
        temperature: float,
    ):
        self.eval()
        return dflash_generate(
            self,
            target=target,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
