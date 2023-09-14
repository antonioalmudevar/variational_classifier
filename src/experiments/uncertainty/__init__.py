def train_uncertainty(method, **kwargs):
    
    if method=="vanilla":
        from .train_vanilla import train_vanilla as train
    elif method=="temperature":
        from .train_temperature import train_temperature as train
    elif method=="ensembles":
        from .train_ensembles import train_ensembles as train
    elif method=="mc_dropout":
        from .train_mc_dropout import train_mc_dropout as train
    elif method=="ll_dropout":
        from .train_ll_dropout import train_ll_dropout as train
    elif method=="vc":
        from .train_vc import train_vc as train
    else:
        raise ValueError
    
    train(**kwargs)


def test_uncertainty(method, **kwargs):
    
    if method=="vanilla":
        from .test_vanilla import test_vanilla as test
    elif method=="temperature":
        from .test_temperature import test_temperature as test
    elif method=="ensembles":
        from .test_ensembles import test_ensembles as test
    elif method=="mc_dropout":
        from .test_mc_dropout import test_mc_dropout as test
    elif method=="ll_dropout":
        from .test_ll_dropout import test_ll_dropout as test
    elif method=="vc":
        from .test_vc import test_vc as test
    else:
        raise ValueError
    
    test(**kwargs)
