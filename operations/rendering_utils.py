import torch
import torch.nn.functional as F


def transform_G_xyz(G, xyz, is_return_homo=False):

    assert len(G.size()) == len(xyz.size())
    if len(G.size()) == 2:
        G_B44 = G.unsqueeze(0)
        xyz_B3N = xyz.unsqueeze(0)
    else:
        G_B44 = G
        xyz_B3N = xyz
    xyz_B4N = torch.cat((xyz_B3N, torch.ones_like(xyz_B3N[:, 0:1, :])), dim=1)
    G_xyz_B4N = torch.matmul(G_B44, xyz_B4N)
    if is_return_homo:
        return G_xyz_B4N
    else:
        return G_xyz_B4N[:, 0:3, :]


def gather_pixel_by_pxpy(img, pxpy):

    with torch.no_grad():
        B, C, H, W = img.size()
        if pxpy.dtype == torch.float32:
            pxpy_int = torch.round(pxpy).to(torch.int64)
        pxpy_int = pxpy_int.to(torch.int64)
        pxpy_int[:, 0, :] = torch.clamp(pxpy_int[:, 0, :], min=0, max=W-1)
        pxpy_int[:, 1, :] = torch.clamp(pxpy_int[:, 1, :], min=0, max=H-1)
        pxpy_idx = pxpy_int[:, 0:1, :] + W * pxpy_int[:, 1:2, :]   
    rgb = torch.gather(img.view(B, C, H * W), dim=2,
                       index=pxpy_idx.repeat(1, C, 1))   
    return rgb


def uniformly_sample_disparity_from_bins(batch_size, disparity_np, device):

    assert disparity_np[0] > disparity_np[-1]
    S = disparity_np.shape[0] - 1

    B = batch_size
    bin_edges = torch.from_numpy(disparity_np).to(dtype=torch.float32, device=device)  
    interval = bin_edges[1:] - bin_edges[0:-1]   
    bin_edges_start = bin_edges[0:-1].unsqueeze(0).repeat(B, 1)   
     
    interval = interval.unsqueeze(0).repeat(B, 1)   

    random_float = torch.rand((B, S), dtype=torch.float32, device=device)  
    disparity_array = bin_edges_start + interval * random_float
    return disparity_array   


def uniformly_sample_disparity_from_linspace_bins(batch_size, num_bins, start, end, device):

    assert start > end

    B, S = batch_size, num_bins
    bin_edges = torch.linspace(start, end, num_bins+1, dtype=torch.float32, device=device)   
    interval = bin_edges[1] - bin_edges[0]   
    bin_edges_start = bin_edges[0:-1].unsqueeze(0).repeat(B, 1)   
     

    random_float = torch.rand((B, S), dtype=torch.float32, device=device)  
    disparity_array = bin_edges_start + interval * random_float
    return disparity_array   


def sample_pdf(values, weights, N_samples):

    B, N, S = weights.size(0), weights.size(2), weights.size(3)
    assert values.size() == (B, 1, N, S)

     
    bin_edges = (values[:, :, :, 1:] + values[:, :, :, :-1]) * 0.5   
    bin_edges = torch.cat((values[:, :, :, 0:1],
                           bin_edges,
                           values[:, :, :, -1:]), dim=3)   

    pdf = weights / (torch.sum(weights, dim=3, keepdim=True) + 1e-5)   
    cdf = torch.cumsum(pdf, dim=3)   
    cdf = torch.cat((torch.zeros((B, 1, N, 1), dtype=cdf.dtype, device=cdf.device),
                     cdf), dim=3)   

     
    u = torch.rand((B, 1, N, N_samples), dtype=weights.dtype, device=weights.device)   

     
    cdf_idx = torch.searchsorted(cdf, u, right=True)   
    cdf_idx_lower = torch.clamp(cdf_idx-1, min=0)   
    cdf_idx_upper = torch.clamp(cdf_idx, max=S)   

     
    cdf_idx_lower_upper = torch.cat((cdf_idx_lower, cdf_idx_upper), dim=3)   
    cdf_bounds_N2 = torch.gather(cdf, index=cdf_idx_lower_upper, dim=3)   
    cdf_bounds = torch.stack((cdf_bounds_N2[..., 0:N_samples], cdf_bounds_N2[..., N_samples:]), dim=4)
    bin_bounds_N2 = torch.gather(bin_edges, index=cdf_idx_lower_upper, dim=3)   
    bin_bounds = torch.stack((bin_bounds_N2[..., 0:N_samples], bin_bounds_N2[..., N_samples:]), dim=4)

     
    cdf_intervals = cdf_bounds[:, :, :, :, 1] - cdf_bounds[:, :, :, :, 0]  
    bin_intervals = bin_bounds[:, :, :, :, 1] - bin_bounds[:, :, :, :, 0]   
    u_cdf_lower = u - cdf_bounds[:, :, :, :, 0]   
     
    t = u_cdf_lower / torch.clamp(cdf_intervals, min=1e-5)
    t = torch.where(cdf_intervals <= 1e-4,
                    torch.full_like(u_cdf_lower, 0.5),
                    t)

    samples = bin_bounds[:, :, :, :, 0] + t*bin_intervals
    return samples
