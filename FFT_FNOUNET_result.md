结果
baseline ref 取自 baseline_summary.csv。

Regime	samples	fRMSE low/mid/high	raw-shell low/mid/high (baseline ref)
M0.1_Eta0.1_128 (smooth)	1000	9.72 / 2076.10 / 119855.80	0.04896 / 0.01371 / 0.008724 (vs 0.00391 / 0.000971 / 0.000225)
M1.0_Eta0.01_128 (transitional)	1000	0.644 / 5.330 / 78.204	0.4151 / 0.08244 / 0.02928 (vs 0.1104 / 0.04019 / 0.007204)
M1.0_Eta1e-08_512 (inviscid shock)	100	0.882 / 1.363 / 5.563	1.1776 / 0.3691 / 0.05720 (vs 0.9139 / 0.2558 / 0.02555)
和 FNO 相比，U-Net 在三个 regime 的三个频带上都更差。最明显的是 smooth case，U-Net 的 fRMSE 相对 FNO 放大到 7.96x / 42.6x / 301x，说明它对中高频结构基本失守。
在 transitional case，U-Net 低频还维持在亚单位量级，但中高频仍明显落后于 FNO，fRMSE 是 3.09x / 1.76x / 5.69x，raw-shell 也比 baseline 高 2x 到 4x。
在 inviscid shock case，U-Net 的低频和中频只比 FNO 差约 1.3x，但高频仍差 5.44x。这说明它能抓住大尺度 shock 轮廓，但细尺度结构明显更糙。
如果按 raw-shell 看 baseline reproduction，U-Net 在 smooth case 偏离最严重，达到 baseline 的 12.5x 到 38.7x；另外两个 regime 也普遍高于 baseline 和 FNO。整体结论很直接：这批 U-Net rollout 的频谱质量显著弱于 FNO，尤其是中高频。
路径

Rollout root: 20260423-160214
FFT root: unet_fft
Smooth: rollout.pt , headline.csv , fft_error_spectrum.png
Transitional: rollout.pt , headline.csv , fft_error_spectrum.png
Inviscid shock: rollout.pt , headline.csv , fft_error_spectrum.png
每个 FFT 目录里都包含 band_metrics.csv、error_radial_spectrum.npy、gt_radial_spectrum.npy、pred_radial_spectrum.npy、rho.npy、headline.csv、fft_error_spectrum.png。
作业日志都在 runs。


Headline validation-set fRMSE / raw-shell (full val split)
Regime	samples	fRMSE low/mid/high	raw-shell low/mid/high (baseline ref)
M0.1_Eta0.1_128 (smooth)	1000	1.22 / 48.7 / 397.7	0.00501 / 0.00158 / 0.000322 (vs 0.00391/0.00097/0.00023)
M1.0_Eta0.01_128 (transitional)	1000	0.208 / 3.03 / 13.7	0.141 / 0.0608 / 0.0132 (vs 0.1104/0.0402/0.00720)
M1.0_Eta1e-08_512 (inviscid shock)	100	0.675 / 1.03 / 1.02	1.00 / 0.336 / 0.0484 (vs 0.914/0.256/0.0255)
Raw-shell reproduction tracks the baseline pickle within 1.1–1.9× (expected: full-plane binning corrects the legacy first-quadrant bias in metric_func, so new numbers are systematically higher). The fRMSE columns show the physically meaningful ordering: in the smooth regime, target has almost no high-k power so fRMSE_high blows up; in the inviscid-shock regime, target has substantial high-k power and fRMSE stays ~O(1) across bands.

Output paths
Rollouts ({pred, target, grid, meta}):

artifacts/fno_rollouts/20260423-153221/2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train/rollout.pt (2.6 GB)
artifacts/fno_rollouts/20260423-153221/2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train/rollout.pt (2.6 GB)
artifacts/fno_rollouts/20260423-153221/2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train/rollout.pt (4.2 GB)
FFT results (per regime: band_metrics.csv, error_radial_spectrum.npy, gt_radial_spectrum.npy, pred_radial_spectrum.npy, rho.npy, headline.csv):

artifacts/fno_fft/2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train/
artifacts/fno_fft/2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train/
artifacts/fno_fft/2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train/
Figures (log-log E_err(k) and E_target(k) per regime with normalized band markers at ρ=1/6, 1/3):

artifacts/fno_fft/2D_CFD_M0.1_Eta0.1_Zeta0.1_periodic_128_Train/fft_error_spectrum.png
artifacts/fno_fft/2D_CFD_M1.0_Eta0.01_Zeta0.01_periodic_128_Train/fft_error_spectrum.png
artifacts/fno_fft/2D_CFD_M1.0_Eta1e-08_Zeta1e-08_periodic_512_Train/fft_error_spectrum.png
Slurm logs:

Rollout: runs/fno-rollout-*-2092924.err / .out, -2092925, -2092926
FFT analysis: runs/fno-fft-*-2092991.err / .out, -2092992, -2092993