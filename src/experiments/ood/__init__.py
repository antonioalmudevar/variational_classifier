
def test_ood(method, **kwargs):
    
    if method=="vanilla":
        from .test_vanilla import test_vanilla as test
    elif method=="vc":
        from .test_vc import test_vc as test
    elif method=="temperature":
        from .test_temperature import test_temperature as test
    elif method=="ensembles":
        from .test_ensembles import test_ensembles as test
    elif method=="mc_dropout":
        from .test_mc_dropout import test_mc_dropout as test
    elif method=="ll_dropout":
        from .test_ll_dropout import test_ll_dropout as test
    else:
        raise ValueError
    
    test(**kwargs)
