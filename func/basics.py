import torch
import torch.nn.functional as F

def warp_image(m, u):
    batch, channels, height, width = m.shape
    
    #meshgrid and stack are using BHWC not BCHW
    grid_y, grid_x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    p = torch.stack([grid_x, grid_y], dim=0).float().to(m.device)
    p = p.unsqueeze(0).expand(batch, -1, -1, -1)


    
    phi = p + u
    
    phi[..., 0] = 2.0 * phi[..., 0] / (width - 1) - 1.0
    phi[..., 1] = 2.0 * phi[..., 1] / (height - 1) - 1.0
    
    unwarped_m = F.grid_sample(m, phi, mode='bilinear', padding_mode='border', align_corners=True)
    
    return unwarped_m


def test_warp_identity():
    m = torch.arange(1, 10).float().view(1, 1, 3, 3)
    u = torch.zeros(1, 2, 3, 3)
    
    out = warp_image(m, u)

    #m[:,:,0,:] = 23
    
    assert torch.allclose(m, out, atol=1e-6)
    

def test_warp_translation_left():
    m = torch.tensor([[[[1., 2., 3.],
                        [4., 5., 6.],
                        [7., 8., 9.]]]])
    
    u = torch.zeros(1, 2, 3, 3)
    u[:, 0, :, :] = 1.0 
    
    expected = torch.tensor([[[[2., 3., 3.],
                               [5., 6., 6.],
                               [8., 9., 9.]]]])
                               
    out = warp_image(m, u)
    
    assert torch.allclose(out, expected, atol=1e-6)


test_warp_identity()

test_warp_translation_left()