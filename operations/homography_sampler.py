import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from utils import inverse


class HomographySample:
    def __init__(self, H_tgt, W_tgt, device=None):
        if device is None:
            self.device = torch.device("cpu")
        else:
            self.device = device

        self.Height_tgt = H_tgt
        self.Width_tgt = W_tgt
        self.meshgrid = self.grid_generation(self.Height_tgt, self.Width_tgt, self.device)
        self.meshgrid = self.meshgrid.permute(2, 0, 1).contiguous()   

        self.n = self.plane_normal_generation(self.device)

    @staticmethod
    def grid_generation(H, W, device):
        x = np.linspace(0, W-1, W)
        y = np.linspace(0, H-1, H)
        xv, yv = np.meshgrid(x, y)   
        xv = torch.from_numpy(xv.astype(np.float32)).to(dtype=torch.float32, device=device)
        yv = torch.from_numpy(yv.astype(np.float32)).to(dtype=torch.float32, device=device)
        ones = torch.ones_like(xv)
        meshgrid = torch.stack((xv, yv, ones), dim=2)   
        return meshgrid

    @staticmethod
    def plane_normal_generation(device):
        n = torch.tensor([0, 0, 1], dtype=torch.float32, device=device)
        return n

    @staticmethod
    def euler_to_rotation_matrix(x_angle, y_angle, z_angle, seq='xyz', degrees=False):

        r = Rotation.from_euler(seq,
                                [-x_angle, -y_angle, -z_angle],
                                degrees=degrees)
        rot_mtx = r.as_matrix().astype(np.float32)
        return rot_mtx


    def sample(self, src_BCHW, d_src_B,
               G_tgt_src,
               K_src_inv, K_tgt):

         
        B, channels, Height_src, Width_src = src_BCHW.size(0), src_BCHW.size(1), src_BCHW.size(2), src_BCHW.size(3)
        R_tgt_src = G_tgt_src[:, 0:3, 0:3]
        t_tgt_src = G_tgt_src[:, 0:3, 3]

        Height_tgt = self.Height_tgt
        Width_tgt = self.Width_tgt
         

        R_tgt_src = R_tgt_src.to(device=src_BCHW.device)
        t_tgt_src = t_tgt_src.to(device=src_BCHW.device)
        K_src_inv = K_src_inv.to(device=src_BCHW.device)
        K_tgt = K_tgt.to(device=src_BCHW.device)

         
        n = self.n.to(device=src_BCHW.device)
        n = n.unsqueeze(0).repeat(B, 1)   
         
         
        d_src_B33 = d_src_B.reshape(B, 1, 1).repeat(1, 3, 3)   
        R_tnd = R_tgt_src - torch.matmul(t_tgt_src.unsqueeze(2), n.unsqueeze(1)) / -d_src_B33
        H_tgt_src = torch.matmul(K_tgt,
                                 torch.matmul(R_tnd, K_src_inv))
                  
         
        with torch.no_grad():
            H_src_tgt = inverse(H_tgt_src)
             
        
        meshgrid_tgt_homo = self.meshgrid.to(src_BCHW.device)
         
        meshgrid_tgt_homo = meshgrid_tgt_homo.unsqueeze(0).expand(B, 3, Height_tgt, Width_tgt)

         
        meshgrid_tgt_homo_B3N = meshgrid_tgt_homo.view(B, 3, -1)   
        meshgrid_src_homo_B3N = torch.matmul(H_src_tgt, meshgrid_tgt_homo_B3N)   
         
        meshgrid_src_homo = meshgrid_src_homo_B3N.view(B, 3, Height_tgt, Width_tgt).permute(0, 2, 3, 1)
        meshgrid_src = meshgrid_src_homo[:, :, :, 0:2] / meshgrid_src_homo[:, :, :, 2:]   

        valid_mask_x = torch.logical_and(meshgrid_src[:, :, :, 0] < Width_src,
                                         meshgrid_src[:, :, :, 0] > -1)
        valid_mask_y = torch.logical_and(meshgrid_src[:, :, :, 1] < Height_src,
                                         meshgrid_src[:, :, :, 1] > -1)
        valid_mask = torch.logical_and(valid_mask_x, valid_mask_y)   

         
         
        meshgrid_src[:, :, :, 0] = (meshgrid_src[:, :, :, 0]+0.5) / (Width_src * 0.5) - 1
        meshgrid_src[:, :, :, 1] = (meshgrid_src[:, :, :, 1]+0.5) / (Height_src * 0.5) - 1
        tgt_BCHW = torch.nn.functional.grid_sample(src_BCHW, grid=meshgrid_src, padding_mode='border',
                                                   align_corners=False)
         
        return tgt_BCHW, valid_mask


def rotation_test():
    rx = HomographySample.euler_to_rotation_matrix(0, 0, 30,
                                                   seq='xyz',
                                                   degrees=True)
    ry = HomographySample.euler_to_rotation_matrix(0, 30, 0,
                                                   seq='xyz',
                                                   degrees=True)
    rz = HomographySample.euler_to_rotation_matrix(30, 0, 0,
                                                   seq='xyz',
                                                   degrees=True)

    rxyz = HomographySample.euler_to_rotation_matrix(30, 30, 30,
                                                     seq='xyz',
                                                     degrees=True)
    print(rxyz)
    print(np.dot(rx, np.dot(ry, rz)))
    print(np.dot(rz, np.dot(ry, rx)))


if __name__ == '__main__':
     

    img_np = cv2.imread("/your_path/", cv2.IMREAD_COLOR)
    img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2RGB)
    H, W, channels = img_np.shape
    src_BCHW = torch.from_numpy(img_np.astype(np.float32)).permute(2,0,1).unsqueeze(0)
    B = src_BCHW.size(0)

     
    R = torch.from_numpy(HomographySample.euler_to_rotation_matrix(10, 20, 20,
                                                                   seq='xyz',
                                                                   degrees=True)).unsqueeze(0).expand(B, 3, 3)
    t = torch.tensor([0, 0, -2], dtype=torch.float32).unsqueeze(0).expand(B, 3)
    d = 10

    sampler = HomographySample()
    tgt_BCHW, valid_mask = sampler.sample(src_BCHW.cuda(), d,
                                          H, W,
                                          R, t)

    tgt_np = tgt_BCHW[0].permute(1,2,0).cpu().numpy()
    tgt_np = np.clip(tgt_np, a_min=0, a_max=255).astype(np.uint8)

    plt.figure()
    plt.imshow(img_np)
    plt.figure()
    plt.imshow(tgt_np)
    plt.show()
