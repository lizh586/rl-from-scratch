import torch


def gru_cell_forward(x_t, h_prev, W_ih, W_hh, b_ih, b_hh):
    # 拆开 W_ih (input weights) 和 W_hh (hidden weights)
    W_ir, W_iz, W_in = W_ih.chunk(3, dim=0)  # 每个 [H, H_in]
    W_hr, W_hz, W_hn = W_hh.chunk(3, dim=0)  # 每个 [H, H]
    # bias 同理
    b_ir, b_iz, b_in = b_ih.chunk(3, dim=0)  # 每个 [H]
    b_hr, b_hz, b_hn = b_hh.chunk(3, dim=0)  # 每个 [H]

    

    r_t = torch.sigmoid(x_t @ W_ir.T + h_prev @ W_hr.T + b_ir + b_hr)
    z_t = torch.sigmoid(x_t @ W_iz.T + h_prev @ W_hz.T + b_iz + b_hz)
    n_t = torch.tanh(x_t @ W_in.T + b_in + r_t * (h_prev @ W_hn.T + b_hn))
    h_t =  z_t * h_prev + (1-z_t) * n_t

    return h_t
        


if __name__ == "__main__":
    B, H_in, H = 1, 4, 4

    gru = torch.nn.GRU(H_in, H, batch_first=True)
    # 获取 PyTorch 的权重
    W_ih = gru.weight_ih_l0.data  # [3H, H_in]
    W_hh = gru.weight_hh_l0.data  # [3H, H]
    b_ih = gru.bias_ih_l0.data    # [3H]
    b_hh = gru.bias_hh_l0.data    # [3H]

    x = torch.randn(B, H_in)
    h = torch.randn(B, H)

    # 你的实现
    my_h = gru_cell_forward(x, h, W_ih, W_hh, b_ih, b_hh)

    # PyTorch
    with torch.no_grad():
        pt_out, _ = gru(x.unsqueeze(1), h.unsqueeze(0))

    diff = (my_h - pt_out.squeeze()).abs().max().item()
    print(f"diff = {diff}")