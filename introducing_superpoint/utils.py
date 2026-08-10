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

def _identity_correspondence(Hc, Wc, device):
    r = torch.arange(Hc, device=device)
    c = torch.arange(Wc, device=device)
    return (
        (r.view(Hc, 1, 1, 1) == r.view(1, 1, Hc, 1))
        & (c.view(1, Wc, 1, 1) == c.view(1, 1, 1, Wc))
    ).float()


def warp_correspondence(homographies, Hc, Wc, grid_size, device):
    """
    homographies: (B, 3, 3) mapping pixel coords in descriptors → other_descriptors.
    Returns s: (B, Hc, Wc, Hc, Wc) with 1 at the destination cell of each source cell centre.
    """
    B = homographies.shape[0]
    gs = float(grid_size)
    yy, xx = torch.meshgrid(
        (torch.arange(Hc, device=device, dtype=torch.float32) + 0.5) * gs,
        (torch.arange(Wc, device=device, dtype=torch.float32) + 0.5) * gs,
        indexing="ij",
    )
    ones = torch.ones(Hc, Wc, device=device, dtype=torch.float32)
    pts = torch.stack([xx, yy, ones], dim=-1).view(1, Hc * Wc, 3).expand(B, -1, -1)
    H = homographies.to(device=device, dtype=torch.float32)
    mapped = torch.bmm(pts, H.transpose(1, 2))
    denom = mapped[..., 2:].clamp(min=1e-6)
    xy = mapped[..., :2] / denom
    col = (xy[..., 0] / gs).long()
    row = (xy[..., 1] / gs).long()
    inside = (row >= 0) & (row < Hc) & (col >= 0) & (col < Wc)
    row = row.clamp(0, Hc - 1)
    col = col.clamp(0, Wc - 1)
    s = torch.zeros(B, Hc * Wc, Hc * Wc, device=device, dtype=torch.float32)
    src = torch.arange(Hc * Wc, device=device).view(1, -1).expand(B, -1)
    dst = row * Wc + col
    b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(src)
    s[b_idx[inside], src[inside], dst[inside]] = 1.0
    return s.view(B, Hc, Wc, Hc, Wc)


def descriptor_loss(
    descriptors,
    other_descriptors,
    config,
    valid_mask=None,
    homographies=None,
):
    batch_size, _, Hc, Wc = descriptors.shape
    device = descriptors.device
    desc = F.normalize(descriptors, p=2, dim=1)
    other_desc = F.normalize(other_descriptors, p=2, dim=1)
    desc_flat = desc.view(batch_size, -1, Hc * Wc)
    other_desc_flat = other_desc.view(batch_size, -1, Hc * Wc)
    dot_product_desc = torch.bmm(desc_flat.transpose(1, 2), other_desc_flat)
    dot_product_desc = F.relu(dot_product_desc)
    dot_product_desc = dot_product_desc.view(batch_size, Hc, Wc, Hc * Wc)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=3)
    dot_product_desc = dot_product_desc.view(batch_size, Hc * Wc, Hc, Wc)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=1)
    dot_product_desc = dot_product_desc.view(batch_size, Hc, Wc, Hc, Wc)

    if homographies is None:
        s = _identity_correspondence(Hc, Wc, device)
        if s.dim() == 4:
            s = s.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
    else:
        s = warp_correspondence(
            homographies, Hc, Wc, int(config["grid_size"]), device
        )

    positive_dist = torch.clamp(config['positive_margin'] - dot_product_desc, min=0.0)
    negative_dist = torch.clamp(dot_product_desc - config['negative_margin'], min=0.0)
    loss = config['lambda_d'] * s * positive_dist + (1 - s) * negative_dist
    if valid_mask is None:
        mask_h, mask_w = Hc * config['grid_size'], Wc * config['grid_size']
        valid_mask = torch.ones((batch_size, 1, mask_h, mask_w), dtype=torch.float32, device=device)
    elif valid_mask.dim() == 3:
        valid_mask = valid_mask.unsqueeze(1).float()
    valid_mask = F.pixel_unshuffle(valid_mask, config['grid_size'])
    valid_mask = torch.prod(valid_mask, dim=1, keepdim=True)
    valid_mask = valid_mask.view(batch_size, 1, 1, Hc, Wc)
    normalization = torch.sum(valid_mask) * (Hc * Wc)
    loss = torch.sum(valid_mask * loss) / (normalization + 1e-8)
    return loss