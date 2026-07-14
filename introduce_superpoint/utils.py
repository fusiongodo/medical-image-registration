import torch
import torch.nn.functional as F

def scanner_invariant_loss(d_a, d_b, radius):
    b, c, h, w = d_a.shape
    kernel_size = 2 * radius + 1
    
    # Extract sliding local blocks from d_b
    db_unfolded = F.unfold(d_b, kernel_size=kernel_size, padding=radius)
    db_unfolded = db_unfolded.view(b, c, -1, h * w) 
    
    da_reshaped = d_a.view(b, c, 1, h * w)
    
    # Compute squared L2 distance across channels
    sq_distances = torch.sum((da_reshaped - db_unfolded) ** 2, dim=1)
    
    # Find minimum distance in the spatial neighborhood
    min_sq_distances, _ = torch.min(sq_distances, dim=1)
    
    return torch.mean(min_sq_distances)


def descriptor_loss(descriptors, warped_descriptors, homographies, config, valid_mask=None, warp_points=None):
    batch_size, _, Hc, Wc = descriptors.shape
    device = descriptors.device
    
    y, x = torch.meshgrid(torch.arange(Hc, device=device), torch.arange(Wc, device=device), indexing='ij')
    coord_cells = torch.stack([y, x], dim=-1).float()
    coord_cells = coord_cells * config['grid_size'] + config['grid_size'] // 2
    
    coord_cells_flat = coord_cells.reshape(-1, 2)
    warped_coord_cells = warp_points(coord_cells_flat, homographies)
    
    coord_cells_exp = coord_cells.view(1, 1, 1, Hc, Wc, 2)
    warped_coord_cells_exp = warped_coord_cells.view(batch_size, Hc, Wc, 1, 1, 2)
    
    cell_distances = torch.norm(coord_cells_exp - warped_coord_cells_exp, dim=-1)
    s = (cell_distances <= (config['grid_size'] - 0.5)).float()
    
    desc = F.normalize(descriptors, p=2, dim=1)
    warped_desc = F.normalize(warped_descriptors, p=2, dim=1)
    
    desc_flat = desc.view(batch_size, -1, Hc * Wc)
    warped_desc_flat = warped_desc.view(batch_size, -1, Hc * Wc)
    
    dot_product_desc = torch.bmm(desc_flat.transpose(1, 2), warped_desc_flat)
    dot_product_desc = F.relu(dot_product_desc)
    
    dot_product_desc = dot_product_desc.view(batch_size, Hc, Wc, Hc * Wc)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=3)
    
    dot_product_desc = dot_product_desc.view(batch_size, Hc * Wc, Hc, Wc)
    dot_product_desc = F.normalize(dot_product_desc, p=2, dim=1)
    
    dot_product_desc = dot_product_desc.view(batch_size, Hc, Wc, Hc, Wc)
    
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



