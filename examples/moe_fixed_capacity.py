"""Two-rank fixed-capacity MoE composition using the three exported kernels.

Pass an already initialized, dedicated EP group in ascending global-rank order.
All group members call this function together. The caller supplies the local
expert computation callback. Device-validation records are collected in
``docs/operators/fused_owner_combine/optimization.md``.
"""
def fixed_capacity_moe(hidden_states, topk_ids, topk_weights, active, *, ep_group, compute_experts):
    import torch
    import torch.distributed as dist
    import torch_npu
    from xpu_kernels import fused_quant_payload, fused_send_slots, fused_owner_combine
    if not dist.is_initialized() or dist.get_world_size(ep_group) != 2:
        raise ValueError('an initialized two-rank EP group is required')
    ranks=dist.get_process_group_ranks(ep_group)
    if ranks != sorted(ranks):
        raise ValueError('EP group must use ascending global-rank order')
    local_rank=dist.get_rank(ep_group)
    if local_rank not in (0,1):
        raise ValueError('caller must belong to the EP group')
    expert_start=local_rank*128
    expert_end=expert_start+128
    payload,_=fused_quant_payload(hidden_states,topk_ids,active)
    gathered=torch.empty((32,2624),dtype=torch.int8,device=payload.device)
    dist.all_gather_into_tensor(gathered,payload,group=ep_group)
    tokens=gathered[:,:2560].contiguous()
    scale=gathered[:,2560:2564].contiguous().view(torch.float32).reshape(32)
    expert_ids=gathered[:,2564:2596].contiguous().view(torch.int32).reshape(32,8)
    gathered_active=gathered[:,2596].contiguous().bool()
    route_ids=torch.where(gathered_active[:,None],expert_ids,expert_end%256)
    expanded_x,inverse_index,expert_tokens,expanded_scale=torch_npu.npu_moe_init_routing_v2(
        tokens,route_ids,scale=scale,offset=None,active_num=256,
        expert_capacity=-1,expert_num=256,drop_pad_mode=0,
        expert_tokens_num_type=1,expert_tokens_num_flag=True,quant_mode=-1,
        active_expert_range=[expert_start,expert_end],row_idx_type=0)
    # The callback owns local expert weights and retains the original expert math.
    down_proj=compute_experts(expanded_x,expanded_scale,expert_tokens.long(),hidden_states.dtype)
    send_slots=fused_send_slots(down_proj.contiguous(),inverse_index.contiguous(),
        route_ids.contiguous(),gathered_active,expert_start,expert_end)
    received=torch.empty_like(send_slots)
    dist.all_to_all_single(received,send_slots,group=ep_group)
    return fused_owner_combine(received,topk_ids,topk_weights,active)
