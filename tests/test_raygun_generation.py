import pytest
import torch
from raygun.pretrained import raygun_2_2mil_800M
from esm.pretrained import esm2_t33_650M_UR50D


@pytest.mark.parametrize("seq, target_length", [
    (
        "MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK",
        210
    )
])
def test_raygun_generation(seq, target_length):
    device = 0
    noise = 0.1

    # Load models
    raymodel = raygun_2_2mil_800M().to(device)
    esmmodel, alph = esm2_t33_650M_UR50D()
    bc = alph.get_batch_converter()
    esmmodel = esmmodel.to(device)
    esmmodel.eval()

    # Tokenize sequence
    data = [("input", seq)]
    _, _, tok = bc(data)

    # ESM embedding
    with torch.no_grad():
        esmemb = esmmodel(tok.to(device), repr_layers=[33], return_contacts=False)["representations"][33][:, 1:-1]

    # Fixed-length embedding
    results = raymodel(esmemb, return_logits_and_seqs=True)
    emb = results["fixed_length_embedding"]

    # Assertions for fixed-length embedding
    assert emb.shape[0] == 1, "Batch size should be 1"
    assert emb.shape[1] == 50, "Sequence length should be 50"
    assert emb.shape[2] == 1280, "Embedding dimension should be 1280"

    # Sample new sequence
    target_len = torch.tensor([target_length], dtype=int)
    results = raymodel(
        esmemb,
        target_lengths=target_len,
        noise=noise,
        return_logits_and_seqs=True
    )

    generated_seq = results["generated-sequences"][0]

    # Assertions for generation
    assert isinstance(generated_seq, str), "Generated sequence should be a string"
    assert len(generated_seq) == target_length, f"Generated sequence should be {target_length} residues long"

