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
            matched_det = torch.zeros(M, dtype=torch.bool)
            matched_gt  = torch.zeros(N, dtype=torch.bool)

            if match_mode == "conf_distance":
                conf_gt    = gt[:, 2]                                    # (N,)
                score_mat  = conf_gt.unsqueeze(0) / (dist + epsilon)     # (M, N)
                flat_order = score_mat.flatten().argsort(descending=True)
                def _accept(det_i, gt_j):
                    return dist[det_i, gt_j] <= radius
            else:
                flat_order = dist.flatten().argsort()
                def _accept(det_i, gt_j):
                    return dist[det_i, gt_j] <= radius

            for idx in flat_order:
                det_i = int(idx // N)
                gt_j  = int(idx % N)
                if not _accept(det_i, gt_j):
                    if match_mode != "conf_distance":
                        break
                    continue
                if matched_det[det_i] or matched_gt[gt_j]:
                    continue
                matches.append((det_i, gt_j))
                matched_det[det_i] = True
                matched_gt[gt_j]   = True

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
    returns dict with scalar loss tensors and detached count aggregates.
    fn and loc terms are weighted by the GT keypoint confidence.
    """
    B, _, Hc, Wc = logits.shape
    dustbin_idx = 64
    loc_terms, fn_terms, fp_terms = [], [], []
    tp_total = fp_total = fn_total = 0

    bin_yx = torch.stack([
        torch.arange(64, device=logits.device) // 8,
        torch.arange(64, device=logits.device) % 8,
    ], dim=1).float()

    for b in range(B):
        match = _match_keypoints_single(
            logits[b], gt_coords[b], cell_size, radius, dustbin_idx,
            match_mode=match_mode, epsilon=match_epsilon,
        )
        tp_total += match["tp"]
        fp_total += match["fp"]
        fn_total += match["fn"]

        for det_i, gt_j in match["matches"]:
            cr, cc = match["det_cells"][det_i]
            bins = logits[b, :dustbin_idx, cr, cc]
            weights = bins.softmax(dim=0)
            pred_offset = (weights.unsqueeze(1) * bin_yx).sum(0)
            pred_y = cr.float() * cell_size + pred_offset[0]
            pred_x = cc.float() * cell_size + pred_offset[1]
            conf_j = gt_coords[b][gt_j, 2]
            loc_terms.append(conf_j * (
                (pred_x - match["gt_px"][gt_j, 0]) ** 2
                + (pred_y - match["gt_px"][gt_j, 1]) ** 2
            ))

        for gt_j in range(match["num_gt"]):
            if gt_j not in match["matched_gt_ids"]:
                cr, cc = match["gt_cell_row"][gt_j], match["gt_cell_col"][gt_j]
                target = match["gt_bin_idx"][gt_j].unsqueeze(0)
                logit_vec = logits[b, :, cr, cc].unsqueeze(0)
                conf_j = gt_coords[b][gt_j, 2]
                fn_terms.append(conf_j * F.cross_entropy(logit_vec, target))

        for det_i in range(match["det_cells"].shape[0]):
            if det_i not in match["matched_det_ids"]:
                cr, cc = match["det_cells"][det_i]
                target = torch.tensor([dustbin_idx], device=logits.device)
                logit_vec = logits[b, :, cr, cc].unsqueeze(0)
                fp_terms.append(F.cross_entropy(logit_vec, target))

    def mean_or_zero(terms):
        return torch.stack(terms).mean() if terms else logits.sum() * 0.0

    loss_loc = mean_or_zero(loc_terms)
    loss_fn = mean_or_zero(fn_terms)
    loss_fp = mean_or_zero(fp_terms)
    loss_total = w_loc * loss_loc + w_fn * loss_fn + w_fp * loss_fp

    return {
        "loss": loss_total,
        "loss_loc": loss_loc,
        "loss_fn": loss_fn,
        "loss_fp": loss_fp,
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
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

def descriptor_loss(descriptors, other_descriptors, config, valid_mask=None):
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

    grid_sigma = config.get('desc_patch_size', 1.0)
    centricity = config.get('desc_centricity', 1.0)
    gy, gx = torch.meshgrid(torch.arange(Hc, device=device, dtype=torch.float32),
                            torch.arange(Wc, device=device, dtype=torch.float32),
                            indexing='ij')
    dy = gy.unsqueeze(2).unsqueeze(3) - gy.unsqueeze(0).unsqueeze(1)
    dx = gx.unsqueeze(2).unsqueeze(3) - gx.unsqueeze(0).unsqueeze(1)
    dist_sq = dx**2 + dy**2
    s = torch.exp(-centricity * dist_sq / (2 * grid_sigma**2))
    s = s * (dist_sq <= 2).float()

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