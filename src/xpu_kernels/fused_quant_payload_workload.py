"""CLI adapter for fused_quant_payload."""
from ._moe_workload import inputs as _inputs, run_cli as _run_cli
def inputs(device):
    return _inputs('fused_quant_payload',device)
def run_cli(args):
    return _run_cli('fused_quant_payload',args)
