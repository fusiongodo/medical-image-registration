import torch
import torch.nn.functional as F

def _match_keypoints_single(
    logits_b, gt, cell_size, radius, dustbin_idx=64,
    match_mode="conf_distance", epsilon=1.0,
):
    """
    logits_b:   (65, Hc, Wc)
    gt:         (N, 3) — (x, y, conf)
    match_mode: "conf_distance" sorts by conf/(dist+epsilon) desc;
                "distance" sorts by dist asc (original behaviour).
    returns dict with matches [(det_i, gt_j)], det_cells (M,2), gt_px (N,2),
    tp/fp/fn counts, and index sets for repeatability.
    """
    prob = logits_b.softmax(dim=0)
    Hc, Wc = logits_b.shape[1], logits_b.shape[2]

    with torch.no_grad():
        scores = prob[:dustbin_idx].max(dim=0).values
        det_mask = (scores > 0.005)
        det_cells = det_mask.nonzero(as_tuple=False)
        det_px = det_cells.float() * cell_size + cell_size / 2
    N = gt.shape[0]
    M = det_cells.shape[0]
    gt_px = gt[:, :2]
    gt_cell_row = (gt_px[:, 1] / cell_size).long().clamp(0, Hc - 1)
    gt_cell_col = (gt_px[:, 0] / cell_size).long().clamp(0, Wc - 1)
    gt_bin_row = (gt_px[:, 1] % cell_size).long().clamp(0, 7)
    gt_bin_col = (gt_px[:, 0] % cell_size).long().clamp(0, 7)
    gt_bin_idx = gt_bin_row * 8 + gt_bin_col

    matches = []
    if M > 0 and N > 0:
        with torch.no_grad():
            det_xy = det_px[:, [1, 0]]
            dist = torch.cdist(det_xy.float(), gt_px.float())
            valid = dist <= radius
            if not valid.any():
                valid_pairs = dist.new_zeros((0, 2), dtype=torch.long)
            else:
                valid_pairs = valid.nonzero(as_tuple=False)

            if match_mode == "conf_distance":
                conf_gt = gt[:, 2]
                pair_scores = conf_gt[valid_pairs[:, 1]] / (
                    dist[valid_pairs[:, 0], valid_pairs[:, 1]] + epsilon
                )
                order = pair_scores.argsort(descending=True)
            else:
                order = dist[valid_pairs[:, 0], valid_pairs[:, 1]].argsort()

            # Single bulk device->host transfer, then a plain-Python greedy
            # loop over CPU arrays. The previous version indexed valid_pairs
            # per iteration with int(gpu_tensor), which forces a device sync
            # on every candidate pair — the actual cost when M or N is large.
            ordered_pairs = valid_pairs[order].cpu().numpy()

        matched_det = bytearray(M)
        matched_gt  = bytearray(N)
        max_matches = min(M, N)
        for det_i, gt_j in ordered_pairs:
            if len(matches) >= max_matches:
                break
            det_i, gt_j = int(det_i), int(gt_j)
            if matched_det[det_i] or matched_gt[gt_j]:
                continue
            matches.append((det_i, gt_j))
            matched_det[det_i] = 1
            matched_gt[gt_j] = 1

    matched_det_ids = {det_i for det_i, _ in matches}
    matched_gt_ids = {gt_j for _, gt_j in matches}

    return {
        "matches": matches,
        "det_cells": det_cells,
        "gt_px": gt_px,
        "gt_cell_row": gt_cell_row,
        "gt_cell_col": gt_cell_col,
        "gt_bin_idx": gt_bin_idx,
        "matched_det_ids": matched_det_ids,
        "matched_gt_ids": matched_gt_ids,
        "tp": len(matches),
        "fp": M - len(matches),
        "fn": N - len(matches),
        "num_gt": N,
    }


def keypoint_matching_loss_detailed(
    logits,
    gt_coords,
    cell_size=8,
    radius=12,
    w_loc=1.0,
    w_fn=1.0,
    w_fp=0.5,
    match_mode="conf_distance",
    match_epsilon=1.0,
):
    """
    logits:    (B, 65, Hc, Wc)
    gt_coords: list[Tensor(Ni, 3)] — (x, y, conf) CNN pixels
    returns dict with scalar loss tensors, detached count aggregates, and
    per_item — list of per-image {tp, fp, fn, num_gt, matched_gt_ids}, reusable
    for KPI accumulation without re-running keypoint matching.
    fn and loc terms are weighted by the GT keypoint confidence.
    Loc/fn/fp terms are gathered across the whole batch and scored with three
    batched GPU calls instead of one Python-level call per matched cell.
    """
    B, _, Hc, Wc = logits.shape
    dustbin_idx = 64
    tp_total = fp_total = fn_total = 0
    per_item = []

    bin_yx = torch.stack([
        torch.arange(64, device=logits.device) // 8,
        torch.arange(64, device=logits.device) % 8,
    ], dim=1).float()

    loc_r, loc_c, loc_gt_xy, loc_conf = [], [], [], []
    fn_r, fn_c, fn_target, fn_conf = [], [], [], []
    fp_r, fp_c = [], []
    loc_batch_idx, fn_batch_idx, fp_batch_idx = [], [], []

    for b in range(B):
        match = _match_keypoints_single(
            logits[b], gt_coords[b], cell_size, radius, dustbin_idx,
            match_mode=match_mode, epsilon=match_epsilon,
        )
        tp_total += match["tp"]
        fp_total += match["fp"]
        fn_total += match["fn"]
        per_item.append({
            "tp": match["tp"],
            "fp": match["fp"],
            "fn": match["fn"],
            "num_gt": match["num_gt"],
            "matched_gt_ids": match["matched_gt_ids"],
        })

        for det_i, gt_j in match["matches"]:
            cr, cc = match["det_cells"][det_i]
            loc_batch_idx.append(b)
            loc_r.append(cr)
            loc_c.append(cc)
            loc_gt_xy.append(match["gt_px"][gt_j])
            loc_conf.append(gt_coords[b][gt_j, 2])

        for gt_j in range(match["num_gt"]):
            if gt_j not in match["matched_gt_ids"]:
                fn_batch_idx.append(b)
                fn_r.append(match["gt_cell_row"][gt_j])
                fn_c.append(match["gt_cell_col"][gt_j])
                fn_target.append(match["gt_bin_idx"][gt_j])
                fn_conf.append(gt_coords[b][gt_j, 2])

        for det_i in range(match["det_cells"].shape[0]):
            if det_i not in match["matched_det_ids"]:
                cr, cc = match["det_cells"][det_i]
                fp_batch_idx.append(b)
                fp_r.append(cr)
                fp_c.append(cc)

    def _gather_cells(batch_idx, rows, cols):
        idx_b = torch.as_tensor(batch_idx, device=logits.device, dtype=torch.long)
        idx_r = torch.stack(rows)
        idx_c = torch.stack(cols)
        return logits[idx_b, :, idx_r, idx_c]

    if loc_batch_idx:
        loc_logits = _gather_cells(loc_batch_idx, loc_r, loc_c)[:, :dustbin_idx]
        weights = loc_logits.softmax(dim=1)
        pred_offset = weights @ bin_yx
        cr_t = torch.stack(loc_r).float()
        cc_t = torch.stack(loc_c).float()
        pred_y = cr_t * cell_size + pred_offset[:, 0]
        pred_x = cc_t * cell_size + pred_offset[:, 1]
        gt_xy = torch.stack(loc_gt_xy)
        conf = torch.stack(loc_conf)
        loss_loc = (conf * ((pred_x - gt_xy[:, 0]) ** 2 + (pred_y - gt_xy[:, 1]) ** 2)).mean()
    else:
        loss_loc = logits.sum() * 0.0

    if fn_batch_idx:
        fn_logits = _gather_cells(fn_batch_idx, fn_r, fn_c)
        targets = torch.stack(fn_target)
        conf = torch.stack(fn_conf)
        per_term = F.cross_entropy(fn_logits, targets, reduction="none")
        loss_fn = (conf * per_term).mean()
    else:
        loss_fn = logits.sum() * 0.0

    if fp_batch_idx:
        fp_logits = _gather_cells(fp_batch_idx, fp_r, fp_c)
        targets = torch.full((fp_logits.shape[0],), dustbin_idx, device=logits.device, dtype=torch.long)
        loss_fp = F.cross_entropy(fp_logits, targets)
    else:
        loss_fp = logits.sum() * 0.0

    loss_total = w_loc * loss_loc + w_fn * loss_fn + w_fp * loss_fp

    return {
        "loss": loss_total,
        "loss_loc": loss_loc,
        "loss_fn": loss_fn,
        "loss_fp": loss_fp,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "per_item": per_item,
    }


def keypoint_matching_loss(
    logits,
    gt_coords,
    cell_size=8,
    radius=12,
    w_loc=1.0,
    w_fn=1.0,
    w_fp=0.5,
    match_mode="conf_distance",
    match_epsilon=1.0,
):
    return keypoint_matching_loss_detailed(
        logits, gt_coords, cell_size, radius, w_loc, w_fn, w_fp,
        match_mode=match_mode, match_epsilon=match_epsilon,
    )["loss"]


def encode_keypoint_labels(gt_coords, Hc, Wc, cell_size=8, dustbin_idx=64, device=None):
    """
    Original SuperPoint label encoding.

    gt_coords: list[Tensor(Ni, 3)] — (x, y, conf) in CNN pixels
    returns Y: (B, Hc, Wc) long in {0..dustbin_idx}; dustbin where no GT point
    falls in the cell. On collision the highest-conf point wins (paper keeps one
    interest point per cell).
    """
    B = len(gt_coords)
    device = device or (gt_coords[0].device if B else torch.device("cpu"))
    Y = torch.full((B, Hc, Wc), dustbin_idx, dtype=torch.long, device=device)
    for b, gt in enumerate(gt_coords):
        if gt.numel() == 0:
            continue
        xy = gt[:, :2].to(device)
        conf = gt[:, 2].to(device) if gt.shape[1] > 2 else torch.ones(gt.shape[0], device=device)
        col = torch.div(xy[:, 0], cell_size, rounding_mode="floor").long()
        row = torch.div(xy[:, 1], cell_size, rounding_mode="floor").long()
        inside = (row >= 0) & (row < Hc) & (col >= 0) & (col < Wc)
        if not bool(inside.any()):
            continue
        xy, conf, row, col = xy[inside], conf[inside], row[inside], col[inside]
        bin_row = (xy[:, 1] - row.float() * cell_size).long().clamp(0, cell_size - 1)
        bin_col = (xy[:, 0] - col.float() * cell_size).long().clamp(0, cell_size - 1)
        label = bin_row * cell_size + bin_col
        cell = row * Wc + col
        # Two stable sorts give cell-major, confidence-ascending order; taking the
        # last entry of each cell run makes the highest-conf winner deterministic,
        # which duplicate-index assignment does not guarantee on CUDA.
        order = conf.argsort(stable=True)
        order = order[cell[order].argsort(stable=True)]
        cell, label = cell[order], label[order]
        keep = torch.ones_like(cell, dtype=torch.bool)
        keep[:-1] = cell[1:] != cell[:-1]
        Y[b].view(-1)[cell[keep]] = label[keep]
    return Y


def cell_valid_from_mask(valid_mask, Hc, Wc, cell_size=8):
    """
    valid_mask: (B, H, W) or (B, 1, H, W) — 1 where pixel is usable.
    returns (B, Hc, Wc) bool — True only if every pixel in the 8×8 cell is valid.
    """
    if valid_mask.dim() == 3:
        vm = valid_mask.unsqueeze(1).float()
    else:
        vm = valid_mask.float()
    vm = F.pixel_unshuffle(vm, cell_size)
    return torch.prod(vm, dim=1) > 0.5


def cell_fov_overlap_mask(homographies, Hc, Wc, cell_size=8, side="src", valid_dst=None):
    """
    Per-cell FOV overlap under H (base→warped).

    side="src": cells in the base frame; keep if H maps them into the warped image.
    side="dst": cells in the warped frame; keep if H^{-1} maps them into the base
    image. A cell is kept only when all four of its extreme pixels map inside, so
    cells straddling the overlap boundary are rejected rather than half-counted.
    If valid_dst is set and side="src", also require every mapped corner to land on
    a valid warped pixel (rotation fill excluded).
    homographies: (B, 3, 3)
    returns (B, Hc, Wc) bool
    """
    device = homographies.device
    B = homographies.shape[0]
    out_h = Hc * cell_size
    out_w = Wc * cell_size
    gs = float(cell_size)
    r0 = torch.arange(Hc, device=device, dtype=torch.float32) * gs
    c0 = torch.arange(Wc, device=device, dtype=torch.float32) * gs
    dy, dx = torch.meshgrid(
        torch.tensor([0.0, gs - 1.0], device=device),
        torch.tensor([0.0, gs - 1.0], device=device),
        indexing="ij",
    )
    xx = c0.view(1, Wc, 1) + dx.reshape(1, 1, 4)
    yy = r0.view(Hc, 1, 1) + dy.reshape(1, 1, 4)
    xx = xx.expand(Hc, Wc, 4)
    yy = yy.expand(Hc, Wc, 4)
    pts = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).reshape(1, Hc * Wc * 4, 3)
    pts = pts.expand(B, -1, -1)
    H = homographies.to(device=device, dtype=torch.float32)
    if side == "dst":
        H = torch.linalg.inv(H)
    elif side != "src":
        raise ValueError(f"unknown side {side!r}")
    mapped = torch.bmm(pts, H.transpose(1, 2))
    xy = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-6)
    ok = (
        (xy[..., 0] >= 0)
        & (xy[..., 0] < float(out_w))
        & (xy[..., 1] >= 0)
        & (xy[..., 1] < float(out_h))
    )
    if side == "src" and valid_dst is not None:
        vm = valid_dst
        if vm.dim() == 4:
            vm = vm[:, 0]
        grid = torch.stack(
            [
                xy[..., 0] / max(float(out_w) - 1.0, 1.0) * 2.0 - 1.0,
                xy[..., 1] / max(float(out_h) - 1.0, 1.0) * 2.0 - 1.0,
            ],
            dim=-1,
        ).view(B, Hc * Wc * 4, 1, 2)
        sampled = F.grid_sample(
            vm.float().unsqueeze(1),
            grid,
            mode="nearest",
            padding_mode="zeros",
            align_corners=True,
        ).view(B, Hc * Wc * 4)
        ok = ok & (sampled > 0.5)
    return ok.view(B, Hc, Wc, 4).all(dim=-1)


def detector_ce_loss(
    logits,
    gt_coords,
    cell_size=8,
    dustbin_idx=64,
    cell_mask=None,
):
    """
    Original SuperPoint detector loss (eq 3): per-cell 65-way -log softmax
    against encoded labels. No NMS, no radius matching.

    logits:    (B, 65, Hc, Wc) raw detector head
    gt_coords: list[Tensor(Ni, 3)] — (x, y, conf) already in this frame
    cell_mask: optional (B, Hc, Wc) bool — cells outside the mask are excluded
    from the mean (FOV non-overlap / invalid warp fill).

    "loss_fn" and "loss_fp" are the same CE restricted to interest cells and to
    dustbin cells respectively; they are diagnostics, not separate penalties, and
    are already contained in "loss".
    """
    B, _, Hc, Wc = logits.shape
    Y = encode_keypoint_labels(
        gt_coords, Hc, Wc, cell_size=cell_size, dustbin_idx=dustbin_idx, device=logits.device
    )
    per_cell = F.cross_entropy(logits, Y, reduction="none")
    if cell_mask is None:
        mask = torch.ones((B, Hc, Wc), dtype=torch.bool, device=logits.device)
    else:
        mask = cell_mask.to(device=logits.device, dtype=torch.bool)
        if mask.shape != per_cell.shape:
            raise ValueError(f"cell_mask shape {tuple(mask.shape)} != {tuple(per_cell.shape)}")
    pos = mask & (Y != dustbin_idx)
    neg = mask & (Y == dustbin_idx)
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    n_mask = int(mask.sum())
    loss = per_cell[mask].mean() if n_mask else logits.sum() * 0.0
    loss_fn = per_cell[pos].mean() if n_pos else logits.sum() * 0.0
    loss_fp = per_cell[neg].mean() if n_neg else logits.sum() * 0.0
    return {
        "loss": loss,
        "loss_fn": loss_fn,
        "loss_fp": loss_fp,
        "n_pos": n_pos,
        "n_neg": n_neg,
        "n_mask": n_mask,
    }


def compute_keypoint_kpis(
    logits_he, logits_ihc, gt_coords, cell_size=8, radius=12,
    match_mode="conf_distance", match_epsilon=1.0,
):
    """
    Repeatability: GT keypoints matched on both HE and IHC / total GT.
    Precision:     sum(tp) / sum(tp + fp) across both stains.
    Recall:        sum(tp) / sum(tp + fn) across both stains.
    """
    B = logits_he.shape[0]
    tp = fp = fn = 0
    repeatable = 0
    total_gt = 0

    for b in range(B):
        match_he  = _match_keypoints_single(logits_he[b],  gt_coords[b], cell_size, radius,
                                            match_mode=match_mode, epsilon=match_epsilon)
        match_ihc = _match_keypoints_single(logits_ihc[b], gt_coords[b], cell_size, radius,
                                            match_mode=match_mode, epsilon=match_epsilon)

        tp += match_he["tp"] + match_ihc["tp"]
        fp += match_he["fp"] + match_ihc["fp"]
        fn += match_he["fn"] + match_ihc["fn"]

        gt_matched_he = match_he["matched_gt_ids"]
        gt_matched_ihc = match_ihc["matched_gt_ids"]
        repeatable += len(gt_matched_he & gt_matched_ihc)
        total_gt += match_he["num_gt"]

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    repeatability = repeatable / (total_gt + 1e-8)

    return {
        "repeatability": repeatability,
        "precision": precision,
        "recall": recall,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }

def _map_cells_to_dst(homographies, rows, cols, Hc, Wc, grid_size):
    """
    Map cell centres through H and report the destination cell index.

    rows, cols: (M,) long — source cell coordinates
    returns (dest, inside): dest (B, M) long flat index clamped into range,
    inside (B, M) bool — False when the centre lands outside the destination grid.
    """
    device = homographies.device
    gs = float(grid_size)
    xx = (cols.to(torch.float32) + 0.5) * gs
    yy = (rows.to(torch.float32) + 0.5) * gs
    pts = torch.stack([xx, yy, torch.ones_like(xx)], dim=-1).unsqueeze(0)
    pts = pts.expand(homographies.shape[0], -1, -1)
    H = homographies.to(device=device, dtype=torch.float32)
    mapped = torch.bmm(pts, H.transpose(1, 2))
    xy = mapped[..., :2] / mapped[..., 2:].clamp(min=1e-6)
    dcol = torch.div(xy[..., 0], gs, rounding_mode="floor").long()
    drow = torch.div(xy[..., 1], gs, rounding_mode="floor").long()
    inside = (drow >= 0) & (drow < Hc) & (dcol >= 0) & (dcol < Wc)
    dest = drow.clamp(0, Hc - 1) * Wc + dcol.clamp(0, Wc - 1)
    return dest, inside


def _cell_weights(valid_mask, cell_mask, batch_size, N, Hc, Wc, gs, device):
    """
    Combine an optional pixel-level valid mask with an optional cell-level mask
    into a single (B, N) float weight in {0, 1}.
    """
    w = torch.ones(batch_size, N, device=device, dtype=torch.float32)
    if valid_mask is not None:
        vm = valid_mask.unsqueeze(1).float() if valid_mask.dim() == 3 else valid_mask.float()
        vm = F.pixel_unshuffle(vm, gs)
        w = w * (torch.prod(vm, dim=1) > 0.5).to(torch.float32).view(batch_size, N)
    if cell_mask is not None:
        w = w * cell_mask.to(device=device, dtype=torch.float32).view(batch_size, N)
    return w


def descriptor_loss(
    descriptors,
    other_descriptors,
    config,
    valid_mask=None,
    homographies=None,
    cell_mask_src=None,
    cell_mask_dst=None,
):
    """
    Magicleap descriptor hinge (SuperPoint eq. 4-5) on the raw cosine similarity
    of L2-unit descriptors.

    descriptors:       (B, D, Hc, Wc) raw descriptor head in the source frame
    other_descriptors: (B, D, Hc, Wc) raw descriptor head in the destination frame
    homographies:      (B, 3, 3) source pixels → destination pixels; None means identity
    valid_mask:        (B, H, W) or (B, 1, H, W) destination-frame pixel validity
    cell_mask_src:     (B, Hc, Wc) usable source cells, e.g. FOV overlap
    cell_mask_dst:     (B, Hc, Wc) usable destination cells

    config["desc_max_cells"] caps the number of *source* cells; every destination
    cell is always scored, so a kept source never loses its correspondence.
    returns scalar
    """
    batch_size, _, Hc, Wc = descriptors.shape
    device = descriptors.device
    gs = int(config["grid_size"])
    N = Hc * Wc
    max_cells = int(config.get("desc_max_cells") or 0)
    desc = F.normalize(descriptors, p=2, dim=1).view(batch_size, -1, N)
    other = F.normalize(other_descriptors, p=2, dim=1).view(batch_size, -1, N)

    w_dst = _cell_weights(valid_mask, cell_mask_dst, batch_size, N, Hc, Wc, gs, device)
    w_src = _cell_weights(None, cell_mask_src, batch_size, N, Hc, Wc, gs, device)

    if 0 < max_cells < N:
        sel = torch.randperm(N, device=device)[:max_cells]
    else:
        sel = torch.arange(N, device=device)
    w_src = w_src[:, sel]

    s = torch.zeros(batch_size, sel.shape[0], N, device=device, dtype=torch.float32)
    if homographies is None:
        s.scatter_(2, sel.view(1, -1, 1).expand(batch_size, -1, 1), 1.0)
    else:
        rows = torch.div(sel, Wc, rounding_mode="floor")
        cols = sel % Wc
        dest, inside = _map_cells_to_dst(homographies, rows, cols, Hc, Wc, gs)
        s.scatter_(2, dest.unsqueeze(-1), inside.unsqueeze(-1).to(torch.float32))

    dots = torch.bmm(desc[:, :, sel].transpose(1, 2), other)
    positive_dist = torch.clamp(float(config["positive_margin"]) - dots, min=0.0)
    negative_dist = torch.clamp(dots - float(config["negative_margin"]), min=0.0)
    pair = float(config["lambda_d"]) * s * positive_dist + (1.0 - s) * negative_dist
    total = torch.einsum("bmn,bm,bn->", pair, w_src, w_dst)
    norm = torch.einsum("bm,bn->", w_src, w_dst)
    return total / (norm + 1e-8)