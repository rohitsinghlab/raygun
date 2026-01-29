# e1_adapter.py
import torch
import torch.nn as nn
from types import SimpleNamespace

class E1Adapter(nn.Module):
    """
    Adapter that exposes a small, ESM-like interface around Profluent E1 models.
    - e1_model : E1
    - batch_preparer : E1BatchPreparer instance (provides tokenizer + pad/mask ids)
    """

    def __init__(self, e1_model, batch_preparer, device=None, use_autocast=False):
        super().__init__()
        self.e1 = e1_model
        self.batch_preparer = batch_preparer
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.use_autocast = use_autocast

        # expose attributes similar to the old ESM object
        self.embed_dim = getattr(self.e1.config, "hidden_size", None) or getattr(self.e1, "hidden_size", None)
        # tokenizer vocab helpers
        try:
            vocab = self.batch_preparer.tokenizer.get_vocab()
        except Exception:
            vocab = {}
        self.token_to_id = vocab
        self.id_to_token = {v: k for k, v in vocab.items()}
        # derive pad/mask ids if set on batch_preparer; fallback to vocab lookups
        self.padding_idx = getattr(self.batch_preparer, "pad_token_id", None) or vocab.get("<pad>", None)
        self.mask_idx = getattr(self.batch_preparer, "mask_token_id", None) or vocab.get("<mask>", None)

        for p in self.e1.parameters():
            p.requires_grad = False

        # move model to device
        self.to(self.device)
        # put into eval by default (caller controls train/eval)
        self.e1.eval()

    def get_tok(self, idx):
        return self.id_to_token.get(int(idx), "<unk>")

    def _build_position_and_sequence_ids(self, input_ids, attention_mask=None):
        """
        Build the three arrays E1 expects:
        - within_seq_position_ids : positions inside each sequence (0..L-1). padded positions -> 0
        - global_position_ids     : fallback to same as within_seq_position_ids
        - sequence_ids            : integer id of sequence for each token (0..B-1), padded -> 0
        Inputs:
            input_ids: LongTensor (B, L)
            attention_mask: LongTensor or ByteTensor (B, L), 1 for real tokens, 0 for pad; if None infer from input_ids != pad
        Returns:
            within_seq_position_ids, global_position_ids, sequence_ids  (all LongTensors, shape B x L)
        """
        B, L = input_ids.shape
        device = input_ids.device

        if attention_mask is None:
            if self.padding_idx is not None:
                attention_mask = (input_ids != self.padding_idx).long()
            else:
                # If no pad id known, treat all tokens as real
                attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

        # within_seq_position_ids: for each row, positions 0..(len-1), padded positions get 0.
        # We'll place positions starting from 0 for the first real token.
        within = torch.zeros((B, L), dtype=torch.long, device=device)
        for i in range(B):
            # length of real tokens
            real_len = int(attention_mask[i].sum().item())
            if real_len > 0:
                within[i, :real_len] = torch.arange(real_len, dtype=torch.long, device=device)

        # global_position_ids: fallback to same as within (safe default)
        global_pos = within.clone()

        # sequence_ids: each token labeled with the sequence index (0..B-1), pad tokens get 0
        seq_ids = torch.zeros((B, L), dtype=torch.long, device=device)
        for i in range(B):
            if attention_mask[i].sum().item() > 0:
                seq_ids[i, : int(attention_mask[i].sum().item())] = i

        return within, global_pos, seq_ids

    def forward(self, input_ids, attention_mask=None, return_hidden=True, **kwargs):
        """
        Forward adaptor. Returns last hidden state tensor (B, L, C) by default.
        Accepts:
            - input_ids : LongTensor (B, L)
            - attention_mask : LongTensor (B, L) or None
        Additional kwargs are ignored but accepted for compatibility.
        """
        # Ensure tensors are on adapter device
        device = self.device
        input_ids = input_ids.to(device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)

        within_seq_position_ids, global_position_ids, sequence_ids = self._build_position_and_sequence_ids(input_ids, attention_mask)

        # prepare call args
        call_kwargs = dict(
            input_ids=input_ids,
            within_seq_position_ids=within_seq_position_ids,
            global_position_ids=global_position_ids,
            sequence_ids=sequence_ids,
            past_key_values=None,
            use_cache=False,
            output_attentions=False,
            # we ask for hidden states to be returned, enabling extraction
            output_hidden_states=True,
        )

        # use autocast if requested and running on CUDA (user can toggle)
        if self.use_autocast and device.type == "cuda":
            torch_autocast = torch.amp.autocast
            ctx = torch_autocast("cuda", dtype=torch.bfloat16)  # follow your earlier usage
        else:
            # no-op context
            from contextlib import nullcontext
            ctx = nullcontext()

        # with torch.no_grad():
        with ctx:
            model_out = self.e1.model(**call_kwargs)

        # Try to extract hidden states in a few possible ways:
        # Prefer last_hidden_state if present; else check hidden_states tuple; else assume model_out[0]
        hidden = getattr(model_out, "last_hidden_state", None)
        if hidden is None:
            # sometimes `hidden_states` is present as a tuple of layer outputs
            hidden_states = getattr(model_out, "hidden_states", None)
            if hidden_states is not None and len(hidden_states) > 0:
                hidden = hidden_states[-1]
            else:
                # fallback: first element of returned tuple
                try:
                    hidden = model_out[0]
                except Exception:
                    raise RuntimeError("Unable to extract hidden states from e1.model() output. Inspect model_out structure.")

        # Ensure hidden is a torch.Tensor
        if not isinstance(hidden, torch.Tensor):
            hidden = torch.tensor(hidden, device=device)
        
        if not hidden.requires_grad:
            hidden.requires_grad_(True)

        return hidden

    # convenience helpers to mimic previous interface
    def to(self, device):
        # override to move both wrapper and inner modules
        dev = torch.device(device) if device is not None else None
        if dev is None:
            return super().to()
        self.device = dev
        # move adapter wrapper
        super().to(dev)
        # move inner model
        try:
            self.e1.to(dev)
        except Exception:
            pass
        return self
