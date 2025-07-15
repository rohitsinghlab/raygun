#---------------------------------------------------------------------
# Implementation of the basic raygun function.
#---------------------------------------------------------------------

#--- imports ---
import torch


RAYGUN_MODEL_NAMES = {'raygun_100k_750M': None,
                      'raygun_2_2mil_800M': None, 
                      'raygun_4_4mil_800M': None}
MODEL = None


def main():
    """
    Main function to execute the raygun functionality.
    This function is a placeholder for the actual implementation.
    """
    print("Raygun functionality is not yet implemented.")

    seq= "MKTAY" * 20
    print(run_raygun(seq, target_length=75, noise=0.0))


def run_raygun(seq, target_length, noise, 
               model_name="raygun_2_2mil_800M", return_logits_and_seqs=True, embedding_model=None):
    # load the model
    load_raygun_model(model_name)
    model = RAYGUN_MODEL_NAMES[model_name]

    # handle embedding model
    if not embedding_model:
        embedding_model = ESMUtility()
    
    embedding = embedding_model(seq, average_pool=False)
    results = model(embedding, target_lengths=target_length, noise=noise, return_logits_and_seqs=return_logits_and_seqs)

    return results


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
        from pretrained import raygun_100k_750M
        RAYGUN_MODEL_NAMES['raygun_100k_750M'] = raygun_100k_750M().to(device)
    elif model_name == 'raygun_2_2mil_800M':
        from pretrained import raygun_2_2mil_800M
        RAYGUN_MODEL_NAMES['raygun_2_2mil_800M'] = raygun_2_2mil_800M().to(device)
    elif model_name == 'raygun_4_4mil_800M':
        from pretrained import raygun_4_4mil_800M
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
        Get embeddings for a list of sequences.
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