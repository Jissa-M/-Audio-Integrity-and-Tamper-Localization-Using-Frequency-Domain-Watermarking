import pytest
from adaptive_detector import detect_adaptive

def windows(scores):
    return [dict(start=i*.25,end=i*.25+.5,score=s,valid=s is not None,valid_frames=0 if s is None else 31,total_frames=31) for i,s in enumerate(scores)]

def test_local_dip():
    w=windows([.35,.35,.01,.01,.35,.35,.35]);p,info=detect_adaptive(w,.09,.2)
    assert info['threshold']==.09
    assert [(x['start'],x['end']) for x in p]==[(.5,1.25)]

def test_uniform_degradation_is_not_local_splice():
    w=windows([.02,.01,.02,.01,.02]);p,_=detect_adaptive(w,.09,.2)
    assert not p and not any(x['suspicious'] for x in w)

def test_invalid_frames_excluded_from_reference():
    w=windows([.35,None,.01,.35,.35]);p,info=detect_adaptive(w,.09,.2)
    assert info['median_score']==.35 and not w[1]['suspicious'] and p

def test_no_evidence():
    w=windows([None,None]);p,info=detect_adaptive(w,.09,.2)
    assert not p and info['status']=='undetermined'

@pytest.mark.parametrize('cap,drop',[(float('nan'),.2),(.09,0),(.09,float('nan'))])
def test_bad_parameters(cap,drop):
    with pytest.raises(ValueError):detect_adaptive(windows([.3]),cap,drop)
