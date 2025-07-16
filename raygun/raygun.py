#---------------------------------------------------------------------
# Implementation of the basic raygun function.
#---------------------------------------------------------------------

#--- imports ---
import torch
import numbers
import numpy as np


RAYGUN_MODEL_NAMES = {'raygun_100k_750M': None,
                      'raygun_2_2mil_800M': None, 
                      'raygun_4_4mil_800M': None}
MODEL = None
FIRST_CALL = True


def main():
    """
    Main function to execute the raygun functionality.
    This function is a placeholder for the actual implementation.
    """
    seq= ''.join(np.random.choice(['A', 'C', 'G', 'T'], size=100, replace=True).tolist())
    results = run_raygun(['seq1'], [seq], target_lengths=[[91]], noise=0.0, return_logits_and_seqs=True)

    print('-' * 30)
    print(results)
    print('-' * 30)

    print(results['seq1']['logits'].shape)


def run_raygun(seq_ids, seqs, target_lengths, noise, model="raygun_4_4mil_800M", 
               return_logits_and_seqs=True, embedding_model=None):
    """
    Inputs:
        - seq_ids (list[str])
        - seqs (list[str])
        - target_lengths (list[list[int]]) - either a global target length or a 
          list of list of target lengths for each sequence. So for example, if we only have one sequence 
          and want to change it to 100 aas, target_lengths=[[100]] or target_lengths=100
        - noise (float or list[float]) - either a global noise level or one noise level for each 
            sequence. Generally between 0.0 and 0.5.
        - model (str) - name of model
        - return logits_and_seqs (bool) - Flag for where or not to return logits and sequences
        - embedding_model - takes sequence as input and outputs ESM2 embeddings. Only supports ESM2_t33_650M_UR50D. 
    
    Output: dictionary where keys are the sequence ids and values are the:
        - fixed length embedding
        - reconstructed embedding
        - logits (if return_logits_and_seqs is True)
        - generated sequences (if return_logits_and_seqs is True)
    """
    # validate inputs
    assert len(seq_ids) == len(seqs)
    if isinstance(target_lengths, numbers.Number):
        target_lengths = [[target_lengths] for _ in range(len(seqs))]
    assert len(target_lengths) == len(seqs)
    if isinstance(noise, numbers.Number):
        noise = [noise for _ in range(len(seqs))]
    assert len(noise) == len(seqs)


    # load model if it is not already passed in 
    if isinstance(model, str):
        load_raygun_model(model)
        model = RAYGUN_MODEL_NAMES[model]

    if not embedding_model:
        embedding_model = ESMUtility()

    # run the raygun model
    out = {}
    for _id, _seq, _target_lengths, _noise in zip(seq_ids, seqs, target_lengths, noise):
        out[_id] = {}
        out[_id]['original_sequence'] = _seq
        
        if return_logits_and_seqs:
            fle, re, log, gs = run_raygun_single_seq(_seq, _target_lengths, _noise, 
                    model, return_logits_and_seqs, embedding_model)
            out[_id]['fixed_length_embedding'] = fle.squeeze()
            out[_id]['reconstructed_embedding'] = re.squeeze()
            out[_id]['logits'] = log.squeeze()
            out[_id]['generated_sequences'] = gs
        else:
            fle, re, = run_raygun_single_seq(_seq, _target_lengths, _noise, 
                    model, return_logits_and_seqs, embedding_model)
            out[_id]['fixed_length_embedding'] = fle.squeeze()
            out[_id]['reconstructed_embedding'] = re.squeeze()

    return out


def run_raygun_single_seq(seq, target_lengths, noise, model="raygun_2_2mil_800M", 
        return_logits_and_seqs=True, embedding_model=None):
    global FIRST_CALL

    # handle target lengths
    if isinstance(target_lengths, numbers.Number):
        target_lengths = torch.tensor([target_lengths], dtype=int) 
    else:
        target_lengths = torch.tensor(target_lengths, dtype=int)

    # load model if it is not already passed in 
    if isinstance(model, str):
        load_raygun_model(model)
        model = RAYGUN_MODEL_NAMES[model]
        
    # handle embedding model
    if not embedding_model:
        embedding_model = ESMUtility()
    
    embedding = embedding_model(seq, average_pool=False)
    results = model(embedding, target_lengths=target_lengths, noise=noise, return_logits_and_seqs=return_logits_and_seqs)

    if FIRST_CALL:
        pass
        # print('-' * 30)
        # print("shape of esm embedding:", embedding.shape)
        # print("results type:", type(results))
        # print('-' * 30)

    FIRST_CALL = False

    if return_logits_and_seqs:
        fixed_length_embedding = results['fixed_length_embedding']
        reconstructed_embedding = results['reconstructed_embedding']
        logits = results['logits']
        generated_sequences = results['generated-sequences']

        return fixed_length_embedding, reconstructed_embedding, logits, generated_sequences
    else:
        fixed_length_embedding = results['fixed_length_embedding']
        reconstructed_embedding = results['reconstructed_embedding']
        return fixed_length_embedding, reconstructed_embedding


def load_raygun_model(model_name, device=None):
    """
    Load a Raygun model by name.
    """
    global RAYGUN_MODEL_NAMES

    # handle device
    if device:
        device = torch.device(device) if isinstance(device, str) else device
        assert isinstance(device, torch.device), "Device must be a valid string or torch.device"
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # load model and return 
    if model_name not in RAYGUN_MODEL_NAMES:
        raise ValueError(f"Model {model_name} is not supported. Supported models: {RAYGUN_MODEL_NAMES}")
    elif model_name == 'raygun_100k_750M':
        from .pretrained import raygun_100k_750M
        RAYGUN_MODEL_NAMES['raygun_100k_750M'] = raygun_100k_750M().to(device)
    elif model_name == 'raygun_2_2mil_800M':
        from .pretrained import raygun_2_2mil_800M
        RAYGUN_MODEL_NAMES['raygun_2_2mil_800M'] = raygun_2_2mil_800M().to(device)
    elif model_name == 'raygun_4_4mil_800M':
        from .pretrained import raygun_4_4mil_800M
        RAYGUN_MODEL_NAMES['raygun_4_4mil_800M'] = raygun_4_4mil_800M().to(device)


class ESMUtility:
    """
    Utility class for using the ESM protein language model.
    """

    SUPPORTED_MODELS = {'esm2_t33_650M_UR50D': {'nlayers': 33}}

    def __init__(self, esm_model_name='esm2_t33_650M_UR50D', device=None):
        # set device
        if device:
            self.device = torch.device(device) if isinstance(device, str) else device
            assert isinstance(self.device, torch.device), "Device must be a valid string or torch.device"
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # set model
        self.esm_model_name = esm_model_name
        self.esm, self.alphabet = self.load_model(esm_model_name)

        # set model to device and eval mode
        self.esm = self.esm.to(self.device)
        self.esm.eval()

        # for debugging 
        self.first_call = True
    

    def __call__(self, seqs, device=None, average_pool=False, return_contacts=False):
        """
        Get embeddings for a list of sequences.
        """
        return self.get_embedding(seqs, device, average_pool, return_contacts)


    def load_model(self, esm_model_name):
        if esm_model_name == 'esm2_t33_650M_UR50D':
            from esm.pretrained import esm2_t33_650M_UR50D
            return esm2_t33_650M_UR50D()
        raise ValueError(f"Unknown model: {esm_model_name}")

    
    def get_embedding(self, seq, device=None, average_pool=False, return_contacts=False):
        """
        Get embeddings for a single sequence.
        """
        # handle device 
        if device is None:
            device = self.device
        else:
            device = torch.device(device) if isinstance(device, str) else device
            assert isinstance(device, torch.device), "Device must be a valid string or torch.device"

        data = [("seq", seq)]
        bc = self.alphabet.get_batch_converter()
        _, _, tok = bc(data)
        num_layers = self.SUPPORTED_MODELS[self.esm_model_name]['nlayers']
        embeddings = self.esm(tok.to(self.device), repr_layers=[num_layers], return_contacts=return_contacts)['representations'][num_layers]

        # remove start and end tokens
        embeddings = embeddings[:, 1:-1, :]

        # average pool if desired
        if average_pool:
            embeddings = embeddings.mean(dim=1)

        # convert to specified device
        embedding = embeddings.to(device)

        if self.first_call:
            print(f"Using ESM model: {self.esm_model_name} with {num_layers} layers")
            print(f"embeddings shape: {embedding.shape}")

        self.first_call = False
        return embedding


if __name__ == "__main__": 
    main()
