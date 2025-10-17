from raygun.pretrained import raygun_2_2mil_800M, raygun_4_4mil_800M, raygun_100k_750M, raygun_8_8mil_800M

def test_raygun4_4mil():
    model = raygun_4_4mil_800M(return_lightning_module=True)
    assert model.model.esmdecoder.fixed_batching == False
    return

def test_raygun100k_750mil():
    model = raygun_100k_750M()
    return

def test_raygun2_2mil():
    model = raygun_2_2mil_800M(return_lightning_module=True)
    assert model.model.esmdecoder.fixed_batching == False
    return

def test_raygun8_8mil():
    model = raygun_8_8mil_800M(return_lightning_module=True)
    assert model.model.esmdecoder.fixed_batching == True
    return