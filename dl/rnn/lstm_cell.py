import torch


def lstm_cell_forward(x_t, h_prev, c_prev, W_ih, W_hh, b_ih, b_hh):
    gates = x_t @ W_ih + h_prev @ W_hh + b_ih + b_hh
    i, f, g, o = gates.chunk(4, dim=1)
    i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
    g = torch.tanh(g)
    c_t = f * c_prev + i * g
    h_t = o * torch.tanh(c_t)
    return h_t, c_t


if __name__ == "__main__":
    B, D, H = 2, 5, 3
    x = torch.randn(B, D)
    h0 = torch.randn(B, H)
    c0 = torch.randn(B, H)

    ref = torch.nn.LSTMCell(D, H)
    h_yours, c_yours = lstm_cell_forward(
        x, h0, c0,
        ref.weight_ih.data.T,   # (4H,D) → (D,4H)
        ref.weight_hh.data.T,   # (4H,H) → (H,4H)
        ref.bias_ih.data,
        ref.bias_hh.data,
    )
    h_ref, c_ref = ref(x, (h0, c0))

    print("h diff:", (h_yours - h_ref).abs().max().item())
    print("c diff:", (c_yours - c_ref).abs().max().item())