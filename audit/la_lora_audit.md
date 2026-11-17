# Evidence for Section 2: LA-LoRA's privacy mechanism

Code: https://github.com/junkangLiu0/LA-LORA, two commits

- **S** = `228579a812ac43e5c98608797de2a8aead8aac3d` (2025-09-25, "Add files via upload").
- **H** = `578e49278e731f26d12457dbc3e4bec89d017a3c` (2026-05-12, head of `main`).

Line numbers come from`git show <sha>:<file>`. Paper: arXiv 2602.19926v1, line numbers of its LaTeX source.

| # | Claim in the paper | Evidence |
|---|---|---|
| 1 | The paper describes per-example clipping to C before averaging | Paper Sec. 4.3, Eq. 9 (l.335-337): "Perform per-example $\ell_2$-norm clipping with threshold $C$ ... $g_i = 1/(bR)\sum_{j\in\mathcal{B}_i} g_{ij} + C/(bR)\cdot\mathcal{N}(0,\sigma^2)$". Batch size 16 for vision (l.494). |
| 2 | The code averages first and clips the average | Loss is `nn.CrossEntropyLoss()` (mean over the batch): [S L797](https://github.com/junkangLiu0/LA-LORA/blob/228579a812ac43e5c98608797de2a8aead8aac3d/DP_lora.py#L797), [H L854](https://github.com/junkangLiu0/LA-LORA/blob/578e49278e731f26d12457dbc3e4bec89d017a3c/DP_lora.py#L854); `loss.backward()` on that loss, then per tensor `param.grad *= min(1, args.C / norm)` with `norm = torch.norm(param.grad, 2)` of the batch-mean gradient; no per-example loop.[S L1052-1073](https://github.com/junkangLiu0/LA-LORA/blob/228579a812ac43e5c98608797de2a8aead8aac3d/DP_lora.py#L1052-L1073), [H L1138-1164](https://github.com/junkangLiu0/LA-LORA/blob/578e49278e731f26d12457dbc3e4bec89d017a3c/DP_lora.py#L1138-L1164) (LA-LoRA). `grep` for `grad_sample`, `GradSampleModule(`, `vmap`, `functorch`, `reduction=` in `DP_lora.py`: 0 hits at both commits. |
| 3 | Noise is still sized for C/\|B\| | Same sites: `torch.normal(0, args.dp_sigma * args.C / args.batch_size, ...)` added to `param.grad`. |
| 4 | One example moves the clipped average by up to 2C; with \|B\| = 16 the noise is 32 times too small per tensor | Arithmetic on rows 2-3: the clipped output has norm <= C per tensor, so two neighbouring batches can differ by 2C; 2C / (C/16) = 32. |
| 5 | C is reset at every step to the median gradient norm of the current batch, capped at 0.4 | [S L876-878](https://github.com/junkangLiu0/LA-LORA/blob/228579a812ac43e5c98608797de2a8aead8aac3d/DP_lora.py#L876-L878): `median_value = statistics.median(values)`; `args.C = min(median_value, 0.4)`. One scalar C per step: the median over the gradient norms of the tensors that have a gradient. Same at every clip site. |
| 6 | It never calls a privacy accountant | `DP_lora.py`, both commits: `from opacus import PrivacyEngine, GradSampleModule` (L7) is only an import; `initialize_dp(` and `get_dp_params(` occur only as their definitions (H L754, L766-767), never called; `get_epsilon`, `eps_computer`, `compute_rdp`: no call. `eps.py` and `eps_computer.py` are imported by no other file (`git grep -E "import eps\|from eps\|eps_computer"` at H). |
| 7 | The noise level is set by hand, as the README concedes | [H README.md L533-549](https://github.com/junkangLiu0/LA-LORA/blob/578e49278e731f26d12457dbc3e4bec89d017a3c/README.md#L533-L549), section "13. 差分隐私相关说明". L547: "当前代码没有自动把目标隐私预算 epsilon=1/2/3 反推出对应的 dp_sigma" (the code does not derive dp_sigma from the target epsilon); L549: a strict privacy report needs an added RDP accountant or a separate calibration of `--dp_sigma`. - readme was updated after publication
 |
| 8 | The same routine serves every baseline in the vision tables | The clip-and-noise pattern of rows 2-3 is in all six client update functions of `DP_lora.py` (FedAvg_per, SAM, FedAvg_LS_per, FedAvg_AL, FFA_LoRA, AR_LoRA): S L846, 896, 942, 1010, 1093, 1146; H L903, 953, 999, 1067, 1185, 1238. Clip-and-noise sites: S L883-887, 930-934, 984-988, 1069-1073, 1134-1137, 1196-1199; H L940-944, 987-991, 1041-1045, 1149-1164, 1226-1230, 1288-1292. |


