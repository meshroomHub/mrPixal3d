"""Memory-light replacement of NAF's neighborhood attention, with natten's exact semantics.

Upstream NAF (src/layers/attentions.py) nearest-upsamples keys and values to the output size, then runs
natten.na2d with dilation = upsampling factor D. At 1024x1024 with 1024 channel values (Pixal3D texture
conditioning) the upsampled values alone take 4 GiB in fp32, which does not fit next to the rest on 16 GB GPUs.

With integer D, the dilated window of an HR query only reads the K x K low-res cells
[clamp(r - K//2, 0, h - K), +K) (natten shifts windows at the borders, it does not pad), r being the low-res
cell of the query. So all the queries of one low-res cell share one window: attention is computed per low-res
cell, on the low-res keys/values, by chunks of rows. Same idea as ComfyUI's comfy/image_encoders/naf.py, which
zero-pads at the borders instead (slightly different from natten there).
"""
import torch

CHUNK_BYTES = 2 ** 29


def _windowStarts(n, kernel, device):
    return (torch.arange(n, device=device) - kernel // 2).clamp(0, n - kernel)


def neighborhoodAttention(q, k, v, kernelSize, scale, numHeads, chunkBytes=CHUNK_BYTES):
    """natten.na2d(q, nearest_upsample(k), nearest_upsample(v), kernel_size, dilation=H/h) without the upsampling.

    q: [B, nH*d, H, W] (high-res queries), k: [B, nH*d, h, w], v: [B, nH*dv, h, w] (low-res), H = D*h, W = D*w.
    Returns [B, nH*dv, H, W] in q.dtype.
    """
    B, C, H, W = q.shape
    h, w = k.shape[-2:]
    Kh, Kw = kernelSize
    Dh, Dw = H // h, W // w
    n = numHeads
    d, dv = C // n, v.shape[1] // n
    out = torch.empty((B, n * dv, H, W), device=q.device, dtype=q.dtype)
    k, v = k.to(q.dtype), v.to(q.dtype)

    colIdx = _windowStarts(w, Kw, q.device)[:, None] + torch.arange(Kw, device=q.device)  # [w, Kw]
    rowStarts = _windowStarts(h, Kh, q.device)
    # live temporaries per low-res row: k/v windows, queries, scores (fp32), outputs
    es = 4
    rowBytes = B * n * w * (Kh * Kw * (d + dv) + Dh * Dw * (d + Kh * Kw + 2 * dv)) * es
    rows = max(1, chunkBytes // rowBytes)
    for r0 in range(0, h, rows):
        r1 = min(r0 + rows, h)
        R, cells = r1 - r0, (r1 - r0) * w
        rowIdx = rowStarts[r0:r1, None] + torch.arange(Kh, device=q.device)  # [R, Kh]
        index = (rowIdx[:, None, :, None], colIdx[None, :, None, :])  # -> [R, w, Kh, Kw]
        kw = k[:, :, index[0], index[1]].view(B, n, d, cells, Kh * Kw).permute(0, 1, 3, 4, 2)  # [B, n, cells, KK, d]
        vw = v[:, :, index[0], index[1]].view(B, n, dv, cells, Kh * Kw).permute(0, 1, 3, 4, 2)  # [B, n, cells, KK, dv]
        qc = q[:, :, r0 * Dh:r1 * Dh].reshape(B, n, d, R, Dh, w, Dw) \
            .permute(0, 1, 3, 5, 4, 6, 2).reshape(B, n, cells, Dh * Dw, d)  # [B, n, cells, DD, d]
        attn = (torch.matmul(qc.float(), kw.float().transpose(-1, -2)) * scale).softmax(dim=-1).to(vw.dtype)
        o = torch.matmul(attn, vw).view(B, n, R, w, Dh, Dw, dv)
        out[:, :, r0 * Dh:r1 * Dh] = o.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, n * dv, R * Dh, W)
    return out


def patch(crossAttentionClass):
    """Route NAF's CrossAttention.forward to neighborhoodAttention when it applies (once per class)."""
    if getattr(crossAttentionClass, "_pixal3dLowmem", False):
        return
    original = crossAttentionClass.forward

    def forward(self, q, k, v, image=None, return_weights=False, **kwargs):
        (hq, wq), (hk, wk) = q.shape[-2:], k.shape[-2:]
        Kh, Kw = self.kernel_size
        if return_weights or hq % hk or wq % wk or v.shape[-2:] != k.shape[-2:] or hk < Kh or wk < Kw:
            return original(self, q, k, v, image=image, return_weights=return_weights, **kwargs)
        self.dilation = (hq // hk, wq // wk)
        return neighborhoodAttention(q, k, v, self.kernel_size, self.scale, self.num_heads)

    crossAttentionClass.forward = forward
    crossAttentionClass._pixal3dLowmem = True
    crossAttentionClass._pixal3dOriginalForward = original
