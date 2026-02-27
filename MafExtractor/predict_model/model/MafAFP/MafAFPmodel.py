from copy import copy
import torch.nn as nn
import torch
from torch import Tensor
from torch.nn import Module, Identity, Linear
from esm.tokenization import EsmSequenceTokenizer
from esm.models.esmc import ESMC
from huggingface_hub import PyTorchModelHubMixin
from networkx import DiGraph
clasfier_params= dict(
    att_mode=True,
    seq_dim=1152, maf_dim=128, aux_dim=None, hidden=512,
    task="binary",
    n_heads=8, ca_dropout=0.1, include_self_in_kv=True,
    film_hidden_mult=2, film_scale_init=0.1,
    gate_temp=2.0, init_gate_bias_maf=-1.0, init_gate_bias_aux=-1.5,
    init_alpha_maf_logit=-0.5, init_alpha_aux_logit=-1.0,
    post_dropout=0.2, head_hidden_mult=0.5, ln_eps=1e-5,
)


class MafAFPClassifier(ESMC, PyTorchModelHubMixin):


    ESM_PRETRAINED_CONFIGS = {

        "esmc_600m": {
            "embedding_dimensions": 1152,
            "num_heads": 18,
            "num_encoder_layers": 36,
        },
    }

    ESM_PRETRAINED_CHECKPOINT_PATHS = {
        "esmc_600m": "/model/weights/esmc_600m_2024_12_v0.pth",
    }

    AVAILABLE_CLASSIFIER_HIDDEN_RATIOS = {1, 2, 4}

    @classmethod
    def from_pretrained(cls, *args, **kwargs) -> "MafAFPClassifier":
        """
        The base model code is not compatible with HuggingFace Hub because the ESMC folks
        store the tokenizer within the model class, which is not a JSON serializable
        configuration. In addition, the base code implements a custom `from_pretrained`
        method but it does not follow the HuggingFace Hub conventions. Therefore, let's
        compensate by redirecting the call to `from_pretrained` to the HuggingFace Hub
        mixin and ensure that we load the tokenizer in the constructor.
        """

        return super(PyTorchModelHubMixin, cls).from_pretrained(*args, **kwargs)

    @classmethod
    def from_esm_pretrained(
        cls,
        model_name: str,
        classifier_hidden_ratio: int,
        id2label: dict[int, str],
        use_flash_attention: bool = True,
    ) -> "MafAFPClassifier":
        """
        Since the base model pretrained weights are stored in a proprietary pickle format,
        let's implement a custom factory method to load those weights.
        """
        import os
        root_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if model_name not in cls.ESM_PRETRAINED_CONFIGS:
            raise ValueError(f"Unknown model name: {model_name}")

        model_args = cls.ESM_PRETRAINED_CONFIGS.get(model_name)

        model = cls(
            **model_args,
            classifier_hidden_ratio=classifier_hidden_ratio,
            id2label=id2label,
            use_flash_attention=use_flash_attention,
        )

        checkpoint_path = cls.ESM_PRETRAINED_CHECKPOINT_PATHS.get(model_name)

        checkpoint_path = root_path + checkpoint_path

        state_dict = torch.load(checkpoint_path)

        model.load_state_dict(state_dict, strict=False)

        return model

    def __init__(
        self,
        embedding_dimensions: int,
        num_heads: int,
        num_encoder_layers: int,
        classifier_hidden_ratio: int,
        id2label: dict[int, str],
        use_flash_attention: bool = True,
    ) -> None:
        if classifier_hidden_ratio not in self.AVAILABLE_CLASSIFIER_HIDDEN_RATIOS:
            raise ValueError(
                f"Invalid classifier_hidden_ratio: {classifier_hidden_ratio}. "
                "Must be one of (1, 2, 4)."
            )

        if len(id2label) < 1:
            raise ValueError("id2label must contain at least one label.")

        # This is required for the base class but is not used otherwise.
        tokenizer = EsmSequenceTokenizer()

        super().__init__(
            d_model=embedding_dimensions,
            n_heads=num_heads,
            n_layers=num_encoder_layers,
            tokenizer=tokenizer,
            use_flash_attn=use_flash_attention,
        )

        # Remove pretrained sequence head from the base model.
        self.sequence_head = Identity()

        self.classifier = GatedFiLMClassifier(
            seq_dim=embedding_dimensions,  # 1152 for esmc_600m
            maf_dim=clasfier_params['maf_dim'],  # 128
            hidden=clasfier_params['hidden'],
            task=clasfier_params['task'],
            film_hidden_mult=2,
            film_scale_init=clasfier_params.get('film_scale', 0.1),
            gate_temp=clasfier_params.get('gate_temp', 2.0),
            init_gate_bias_maf=clasfier_params.get('init_gate_bias_maf', -1.5),
            init_alpha_maf_logit=clasfier_params.get('init_alpha_maf_logit', -1.0),
            post_dropout=clasfier_params.get('dropout', 0.2),
            head_hidden_mult=0.5,
        )

        id2label = {int(index): str(label) for  label,index in id2label.items()}

        self.embedding_dimensions = embedding_dimensions
        self.id2label = id2label
        self.graph: DiGraph | None = None

    @property
    def num_encoder_layers(self) -> int:
        return len(self.transformer.blocks)

    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @property
    def num_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def label2id(self) -> dict[str, int]:
        return {label: index for index, label in self.id2label.items()}

    @property
    def num_classes(self) -> int:
        return len(self.id2label)

    def freeze_base(self) -> None:
        """Prevent the base model parameters from being updated during training."""

        for module in (self.embed, self.transformer):
            for param in module.parameters():
                param.requires_grad = False

    def unfreeze_last_k_encoder_layers(self, k: int) -> None:
        """Allow the last k encoder layers to be trainable."""

        if k <= 0:
            return

        for module in self.transformer.blocks[-k:]:
            for param in module.parameters():
                param.requires_grad = True

    def forward_backbone_only(self, sequence_tokens, sequence_id: Tensor | None = None):
        out = super().forward(sequence_tokens=sequence_tokens,sequence_id=sequence_id)
        return out.embeddings[:, 0, :]  # CLS

    def forward_classifier_only(self, h_seq, h_maf):
        return self.classifier(h_seq.float(), h_maf.float())


    @torch.inference_mode()
    def predict_terms(
        self, sequence_tokens: Tensor, top_p: float = 0.5
    ) -> dict[str, float]:
        """Predicts GO terms based on the input sequence tokens."""

        assert sequence_tokens.ndim == 1, "sequence must be a 1D tensor."
        assert 0 < top_p <= 1, "top_p must be in the range (0, 1]."

        z = self.forward(sequence_tokens.unsqueeze(0)).squeeze(0)

        probabilities = torch.sigmoid(z).tolist()

        probabilities = {
            self.id2label[index]: probability
            for index, probability in enumerate(copy(probabilities))
            if probability > top_p
        }

        return probabilities


class GatedFiLMClassifier(nn.Module):
    def __init__(
        self,
        seq_dim: int = 1152,
        maf_dim: int = 128,
        hidden: int = 512,
        task: str = "binary",
        # FiLM
        film_hidden_mult: int = 2,
        film_scale_init: float = 0.1,
        # Gating
        gate_temp: float = 2.0,
        init_gate_bias_maf: float = -1.5,
        init_alpha_maf_logit: float = -1.0, # sigmoid(-1)≈0.27
        post_dropout: float = 0.2,
        # MLP Head
        head_hidden_mult: float = 0.5,      # head = hidden * 0.5
        ln_eps: float = 1e-5,
    ):
        super().__init__()
        assert task in {"binary", "regression"}
        self.task = task
        self.hidden = hidden
        self.gate_temp = gate_temp
        self.seq_proj = nn.Linear(seq_dim, hidden)
        self.maf_proj = nn.Linear(maf_dim, hidden)
        film_in = maf_dim
        self.film = nn.Sequential(
            nn.Linear(film_in, film_hidden_mult * hidden),
            nn.GELU(),
            nn.Linear(film_hidden_mult * hidden, 2 * hidden),
        )
        self.film_gamma_scale = nn.Parameter(torch.tensor(film_scale_init))
        self.gate_maf = nn.Linear(maf_dim, hidden)
        nn.init.constant_(self.gate_maf.bias, init_gate_bias_maf)
        self.alpha_maf_logit = nn.Parameter(torch.tensor(float(init_alpha_maf_logit)))
        self.post_ln = nn.LayerNorm(hidden, eps=ln_eps)
        self.post_dp = nn.Dropout(post_dropout)
        head_hidden = max(1, int(hidden * head_hidden_mult))
        self.head = nn.Sequential(
            nn.Linear(hidden, head_hidden),
            nn.GELU(),
            nn.Dropout(post_dropout),
            nn.Linear(head_hidden, 1),
        )

        self.reset_parameters()


    def reset_parameters(self):
        for lin in [self.seq_proj, self.maf_proj]:
            nn.init.xavier_uniform_(lin.weight); nn.init.zeros_(lin.bias)
        # FiLM
        for m in self.film:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)
        # Gates
        nn.init.xavier_uniform_(self.gate_maf.weight)
        # Head
        for m in self.head:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def _prep_seq_tokens(self, h_seq: torch.Tensor):
        if h_seq.dim() == 2:        # [B, D]
            z_seq = self.seq_proj(h_seq)            # [B, H]
            q_seq = z_seq.unsqueeze(1)              # [B, 1, H]
            kv_seq = q_seq                           # [B, 1, H]
        elif h_seq.dim() == 3:      # [B, L, D]
            z_seq = self.seq_proj(h_seq)            # [B, L, H]

            q_seq = z_seq[:, :1, :]                 # [B, 1, H]
            kv_seq = z_seq                          # [B, L, H]
        else:
            raise ValueError("h_seq must be [B, D] or [B, L, D].")
        return q_seq, kv_seq

    def _tokenize_global(self, x: torch.Tensor, proj: nn.Linear) -> torch.Tensor:

        return proj(x).unsqueeze(1)


    def forward(
        self,
        h_seq: torch.Tensor,          # [B, D] 或 [B, L, D]
        h_maf: torch.Tensor,          # [B, D_maf]

    ) -> torch.Tensor:

        h_seq = torch.nan_to_num(h_seq, 0.0, 0.0, 0.0)
        h_maf = torch.nan_to_num(h_maf, 0.0, 0.0, 0.0)
        q_seq, kv_seq = self._prep_seq_tokens(h_seq)     # [B,1,H], [B,L,H] or [B,1,H]
        kv_maf = self._tokenize_global(h_maf, self.maf_proj)   # [B,1,H]
        if self.include_self_in_kv and kv_seq.size(1) > 1:
            kv = torch.cat([kv_maf, kv_seq], dim=1)  # [B,1+L,H]
        else:
            kv = kv_maf

        z_maf_ca, attn_weights  = self.ca_maf(q_seq, kv, kv, need_weights=True,average_attn_weights=False)  # [B,1,H]
        self.ca_maf.last_attn = attn_weights.detach()

        z_seq = q_seq  # [B,1,H]
        film_in = [h_maf]
        film_in = torch.cat(film_in, dim=-1)          # [B, D_maf + D_aux]
        gamma_beta = self.film(film_in)               # [B, 2H]
        gamma, beta = gamma_beta.chunk(2, dim=-1)     # [B,H], [B,H]
        gamma = torch.tanh(gamma) * self.film_gamma_scale
        z_seq_film = (1.0 + gamma).unsqueeze(1) * z_seq + beta.unsqueeze(1)  # [B,1,H]

        # Gating + α
        g_maf = torch.sigmoid(self.gate_maf(h_maf) / self.gate_temp).unsqueeze(1)   # [B,1,H]
        alpha_maf = torch.sigmoid(self.alpha_maf_logit)                              # scalar

        z = z_seq_film + alpha_maf * (g_maf * z_maf_ca)
        z = self.post_ln(z)           # [B,1,H]
        z = self.post_dp(z)
        z = z.squeeze(1)              # [B,H]
        out = self.head(z)            # [B,1]
        return out


