from raygun.modelv2.raygun import Raygun

class RayFlow(nn.Embedding):
    def __init__(self, raygun, 
                rfdenoiser):
        self.raygun = raygun
        self.ray