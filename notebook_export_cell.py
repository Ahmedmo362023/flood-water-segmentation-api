# ---- Paste into the LAST cell of water-segmentation.ipynb (Kaggle) and run ----
# Exports the best Week 1-2 model (selected by validation IoU) + 3 real test samples.
import json
from rasterio.transform import from_origin

all_runs = {   # val IoU of every candidate from both weeks
    'From_scratch_UNet_12ch': res.loc[res.experiment == 'E1_baseline_12ch', 'best_val_iou'].item(),
    **{r['experiment']: r['best_val_iou'] for r in p_results}}
print(all_runs)

best_p = max(p_results, key=lambda r: r['best_val_iou'])['experiment']   # P3_deeplab_r34_mean
cfg = P_EXPERIMENTS[best_p]
best_model = p_models[best_p].cpu().eval()
te_dl = DataLoader(FloodDS(X_norm, Y, te_idx, RAW), batch_size=16)
best_model.to(DEVICE); m = evaluate(best_model, te_dl, p_thr[best_p]); best_model.cpu()

OUT = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('.')
torch.save({'state_dict': best_model.state_dict(),
            'arch': cfg['arch'], 'encoder': cfg['encoder'],
            'channels': RAW, 'thr': float(p_thr[best_p]),
            'norm_lo': norm.lo[:12].copy(), 'norm_hi': norm.hi[:12].copy(),   # first 12 = raw bands
            'name': best_p, 'test_metrics': {k: float(v) for k, v in m.items()}},
           OUT / 'best_model.pt')

# 3 real test samples (raw, un-normalised 12-band GeoTIFFs) + ground-truth masks
(OUT / 'samples').mkdir(exist_ok=True)
for k, i in enumerate(te_idx[:3], 1):
    with rasterio.open(OUT / 'samples' / f'sample_{k}.tif', 'w', driver='GTiff', height=128, width=128,
                       count=12, dtype='float32', transform=from_origin(0, 0, 1, 1)) as dst:
        dst.write(X[i])
    from PIL import Image
    Image.fromarray((Y[i] * 255).astype('uint8')).save(OUT / 'samples' / f'sample_{k}_gt.png')
print('exported', best_p, m)
