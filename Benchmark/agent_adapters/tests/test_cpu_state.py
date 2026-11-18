from copy import deepcopy
import pytest
from agent_adapters.cpu_state import render_cpu_state
from scictrl.synthesize_cpu_diagnostic import PACKS,public_case


@pytest.mark.parametrize('layer',['layer1','layer2'])
def test_cpu_native_envelope(tmp_path,layer):
    case=public_case(PACKS[0],layer,False)
    text=render_cpu_state(case,tmp_path)
    assert case['objective'] in text
    assert (tmp_path/'analysis.py').exists()==(layer=='layer1')
    assert (tmp_path/'public_state.json').exists()


def test_nested_private_boundary(tmp_path):
    case=public_case(PACKS[0],'layer2',False)
    case['state']['data']['oracle']={'expected':2}
    with pytest.raises(ValueError):render_cpu_state(case,tmp_path)


def test_false_checkpoint_rejected(tmp_path):
    case=public_case(PACKS[0],'layer2',False)
    case['state']['observations']=[{'output':99}]
    with pytest.raises(ValueError):render_cpu_state(case,tmp_path)
