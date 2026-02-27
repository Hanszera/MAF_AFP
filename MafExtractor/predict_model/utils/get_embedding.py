import torch

from esm.tokenization import EsmSequenceTokenizer
from esm.tokenization import (
    get_esmc_model_tokenizers,
)
from esm.utils.misc import stack_variable_length_tensors
from esm.utils.sampling import _BatchedESMProteinTensor


def esm_encoder_seq(sequences,max_length) -> torch.Tensor:
    tokenizer = get_esmc_model_tokenizers()
    pad = tokenizer.pad_token_id

    assert pad is not None
    encode_se = stack_variable_length_tensors(
        [
            tokenize_sequence(x, tokenizer, max_length=max_length,add_special_tokens=True)
            for x in sequences.sequence['sequences']
        ],
        constant_value=pad,
    )
    return encode_se
def tokenize_sequence(
    sequence: str,
    sequence_tokenizer: EsmSequenceTokenizer,
    max_length:int,
    add_special_tokens: bool = True

) -> torch.Tensor:
    sequence = sequence.replace('_', sequence_tokenizer.mask_token)
    sequence_tokens = sequence_tokenizer.encode(
        sequence, add_special_tokens=add_special_tokens,padding="longest", truncation=True, max_length=max_length
    )
    sequence_tokens = torch.tensor(sequence_tokens, dtype=torch.int64)
    return sequence_tokens


def get_esm_embedding(seq,max_length) -> _BatchedESMProteinTensor:
    protein_tensor = _BatchedESMProteinTensor(sequence=esm_encoder_seq(seq,max_length).to('cuda'))

    return protein_tensor