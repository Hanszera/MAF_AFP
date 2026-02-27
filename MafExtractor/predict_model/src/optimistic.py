def build_llrd_param_groups(
    model,
    base_lr: float = 5e-5,          # 主干基础学习率（建议比你原先5e-4小一个量级，更稳）
    head_lr_mult: float = 5.0,      # 分类头相对主干的LR倍数（例如5x）
    weight_decay: float = 1e-4,
    lr_decay: float = 0.8,          # 层间学习率衰减系数（越靠前层 LR 越小）
):
    """
    目标：
      • 解冻的 Transformer block 使用 LLRD（越靠近输出的层，LR 越大）
      • LayerNorm 和 bias 参数不做权重衰减（no_decay）
      • 分类头（model.classifier.*）使用更大学习率（head_lr = base_lr * head_lr_mult）

    说明：
      • 仅收集 requires_grad=True 的参数
      • 层索引来自命名：transformer.blocks.{i}.*
      • 最后一层（i = num_layers-1）的 LR = base_lr
        前一层 LR = base_lr * lr_decay，以此类推
    """
    import re
    n_layers = model.num_encoder_layers  # ESMC: 36
    layer_pat = re.compile(r"^transformer\.blocks\.(\d+)\.")

    # no_decay: LayerNorm权重（1维）、bias，一般也包括embedding（若训练时解冻）
    def is_no_decay(n, p):
        if p.ndim == 1:
            return True
        if n.endswith(".bias"):
            return True
        # 部分LayerNorm命名可能是 "ln", "layer_norm", "norm" 等，这里宽松匹配
        ln_keys = ("layernorm", "layer_norm", "ln", "norm")
        return any(k in n.lower() for k in ln_keys)

    # 分类头命名前缀（你当前头是 model.classifier.*）
    def is_head(n):
        return n.startswith("classifier.")

    # 计算某个block的层内学习率（越靠后层幂指数越小）
    def lr_for_layer_idx(i: int):
        # i 越大越靠后：最后一层 i = n_layers-1 -> pow = 0 -> lr = base_lr
        pow_k = (n_layers - 1 - i)
        return base_lr * (lr_decay ** pow_k)

    # 先将参数按组聚合，避免过多的小 param_group 影响性能
    groups = {}  # key: (group_name, lr, wd) -> list(params)
    def add_param(group_name, lr, wd, p):
        key = (group_name, lr, wd)
        if key not in groups:
            groups[key] = []
        groups[key].append(p)

    # 遍历所有可训练参数
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if is_head(n):
            # 分类头单独使用更大学习率，并区分 decay / no_decay
            lr = base_lr * head_lr_mult
            wd = 0.0 if is_no_decay(n, p) else weight_decay
            add_param("head_no_decay" if wd == 0.0 else "head_decay", lr, wd, p)
            continue

        m = layer_pat.match(n)
        if m is not None:
            # Transformer block（只会包含你解冻的最后K层；前面层被冻结则不会进入）
            i = int(m.group(1))
            lr = lr_for_layer_idx(i)
            wd = 0.0 if is_no_decay(n, p) else weight_decay
            add_param(f"layer{i}_no_decay" if wd == 0.0 else f"layer{i}_decay", lr, wd, p)
            continue

        # 其它（例如 embed.* 或顶层未归类模块；一般被冻结，不会进来；若进来按主干最小LR处理）
        lr = base_lr * (lr_decay ** (n_layers - 1))
        wd = 0.0 if is_no_decay(n, p) else weight_decay
        add_param("others_no_decay" if wd == 0.0 else "others_decay", lr, wd, p)

    # 组装为 AdamW param_groups
    param_groups = []
    print("\n[LLRD] Param groups summary:")
    for (name, lr, wd), params in groups.items():
        param_groups.append({"params": params, "lr": lr, "weight_decay": wd})
        n_params = sum(p.numel() for p in params)
        print(f"  - {name:20s} | lr={lr:.2e} | wd={wd:.1e} | #params={n_params:,}")
    print()
    return param_groups