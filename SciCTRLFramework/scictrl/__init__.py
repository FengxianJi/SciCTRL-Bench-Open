"""SciCTRL APIs, lazily imported to keep private evaluation out of transports."""
from importlib import import_module

_EXPORTS={
    'CSFIEnvironment':('environment','CSFIEnvironment'),
    'evaluate_executable_submission':('evaluator','evaluate_executable_submission'),
    'evaluate_submission':('evaluator','evaluate_submission'),
    'build_case':('injector','build_case'),
    'public_view':('injector','public_view'),
    'validate_case':('injector','validate_case'),
    'Layer':('models','Layer'),'Outcome':('models','Outcome'),
    'normalize_trajectory':('normalizer','normalize_trajectory'),
    'ExecutableCSFIEnvironment':('executable_runtime','ExecutableCSFIEnvironment'),
    'executable_case':('executable_runtime','executable_case'),
    'validate_ml_group_leakage_workspace':('executable_runtime','validate_ml_group_leakage_workspace'),
    'validate_attached_executable_workspace':('executable_runtime','validate_attached_executable_workspace'),
}
__all__=list(_EXPORTS)

def __getattr__(name):
    if name not in _EXPORTS:raise AttributeError(name)
    module,attr=_EXPORTS[name]
    value=getattr(import_module('.'+module,__name__),attr)
    globals()[name]=value
    return value
