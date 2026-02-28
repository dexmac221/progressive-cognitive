import torch

r = 16
k = 8
d_in = 1024
d_out = 1024

A = torch.randn(r, d_in)
B = torch.randn(d_out, r)

# Original product
W = B @ A

# QR decomposition
Q_B, R_B = torch.linalg.qr(B)
Q_A, R_A = torch.linalg.qr(A.T)

C = R_B @ R_A.T
U, S, Vh = torch.linalg.svd(C)

# Keep top k
U_k = U[:, :k]
S_k = S[:k]
Vh_k = Vh[:k, :]

B_new = Q_B @ U_k @ torch.diag(torch.sqrt(S_k))
A_new = torch.diag(torch.sqrt(S_k)) @ Vh_k @ Q_A.T

# Pad to original shape
B_padded = torch.zeros_like(B)
B_padded[:, :k] = B_new

A_padded = torch.zeros_like(A)
A_padded[:k, :] = A_new

W_new = B_padded @ A_padded

print("Original rank:", torch.linalg.matrix_rank(W))
print("New rank:", torch.linalg.matrix_rank(W_new))
print("Shape B_padded:", B_padded.shape)
print("Shape A_padded:", A_padded.shape)
print("Error (Frobenius):", torch.norm(W - W_new).item())
