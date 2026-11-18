import pytest
from build_experiment_appendices import blind_metrics, esc, listing


def row(key,p,status='completed'):
    return {'key':key,'status':status,'answer':{'issue_probability':p}}


def test_balanced_accuracy_and_tied_auc():
    labels={'a':{'clean':False,'domain':'math'},'b':{'clean':True,'domain':'math'}}
    m=blind_metrics([row('a',.5),row('b',.5)],labels)
    assert m['balanced_accuracy']==.5
    assert m['auc']==.5
    assert m['flawed_correct']==1 and m['clean_correct']==0


def test_unavailable_requests_are_not_scientific_errors():
    m=blind_metrics([row('missing',0,'failed')],{})
    assert m['records']==1 and m['completed']==0
    assert m['auc'] is None and m['balanced_accuracy'] is None


def test_no_tex_execution_through_a_listing():
    with pytest.raises(ValueError):listing('\\end{lstlisting}\\input{secret}')
    assert esc('a_b%')=='a\\_b\\%'
