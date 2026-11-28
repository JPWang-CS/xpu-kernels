"""CLI adapter for fused_owner_combine."""
from ._moe_workload import inputs as _inputs, run_cli as _run_cli
def inputs(device):
    return _inputs('fused_owner_combine',device)
def run_cli(args):
    return _run_cli('fused_owner_combine',args)
