# Downloads NTIRE 2026 mobile SR models (Samsung one-step diffusion, Antman RRDB), exports ONNX,
# and compares them on a public stock face. Runs on GitHub Actions.
import os, sys, json, time, glob, shutil
import numpy as np, cv2, torch, gdown
sys.path.insert(0, 'ntire')
os.makedirs('out', exist_ok=True)

FOLDER_ID = '1WvVqMqS8XxAsBRaYpPgHYTYe62agsP01'
import re, requests, html
def list_folder(fid, prefix=''):
    """Lists a public Drive folder (embedded view), recursing only into the folders we need."""
    r = requests.get(f'https://drive.google.com/embeddedfolderview?id={fid}', timeout=60)
    if r.status_code != 200:
        print('skip folder', prefix, r.status_code); return []
    out = []
    for m in re.finditer(r'<a href="(https://drive\.google\.com/[^"]+)"[^>]*>.*?<div class="flip-entry-title">([^<]+)</div>', r.text, re.S):
        url, name = html.unescape(m.group(1)), html.unescape(m.group(2)).strip()
        path = f'{prefix}/{name}' if prefix else name
        fm = re.search(r'/folders/([\w-]+)', url)
        if fm:
            if prefix or 'team11' in name or 'team06' in name:
                out += list_folder(fm.group(1), path)
            else:
                print('  (folder)', path)
        else:
            im = re.search(r'/file/d/([\w-]+)', url)
            if im: out.append((im.group(1), path))
    return out
want = list_folder(FOLDER_ID)
print('drive files:', len(want))
for fid, path in want:
    print('  file', path)
    dst = os.path.join('ntire', 'model_zoo', path)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    gdown.download(id=fid, output=dst, quiet=True)
    print('got', dst, os.path.getsize(dst) if os.path.exists(dst) else 'MISSING')

SD21_UNET = {"_class_name":"UNet2DConditionModel","act_fn":"silu","attention_head_dim":[5,10,20,20],
 "block_out_channels":[320,640,1280,1280],"center_input_sample":False,"cross_attention_dim":1024,
 "down_block_types":["CrossAttnDownBlock2D","CrossAttnDownBlock2D","CrossAttnDownBlock2D","DownBlock2D"],
 "downsample_padding":1,"dual_cross_attention":False,"flip_sin_to_cos":True,"freq_shift":0,"in_channels":4,
 "layers_per_block":2,"mid_block_scale_factor":1,"norm_eps":1e-05,"norm_num_groups":32,"num_class_embeds":None,
 "only_cross_attention":False,"out_channels":4,"sample_size":64,
 "up_block_types":["UpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D","CrossAttnUpBlock2D"],
 "upcast_attention":False,"use_linear_projection":True}

models = {}
# --- Samsung (team11)
try:
    sams_dir = glob.glob('ntire/model_zoo/**/weights.pth', recursive=True)
    sams_dir = [p for p in sams_dir if 'team11' in p]
    print('samsung weights', sams_dir)
    mdir = os.path.dirname(sams_dir[0])
    cfgdir = os.path.join(mdir, 'stable-diffusion-2-1-base', 'unet')
    if not os.path.exists(os.path.join(cfgdir, 'config.json')):
        os.makedirs(cfgdir, exist_ok=True)
        json.dump(SD21_UNET, open(os.path.join(cfgdir, 'config.json'), 'w'))
    from models.team11_SamsungAKCamera.io import load_model
    m = load_model(mdir, device=torch.device('cpu'))
    print('samsung params', sum(p.numel() for p in m.parameters()) / 1e6, 'M')
    models['samsung'] = m
except Exception as e:
    import traceback; traceback.print_exc()

# --- Antman (team06, RRDBNet)
try:
    from models.team06_Antman.model import RRDBNet
    w = [p for p in glob.glob('ntire/model_zoo/**/*.pth', recursive=True) if 'team06' in p and p.endswith('ESRGAN.pth')]
    print('antman weights', w)
    net = RRDBNet()
    sd = torch.load(w[0], map_location='cpu', weights_only=False)
    for k in ('params_ema', 'params', 'state_dict', 'model'):
        if isinstance(sd, dict) and k in sd: sd = sd[k]; break
    net.load_state_dict(sd, strict=True); net.eval()
    models['antman'] = net
except Exception as e:
    import traceback; traceback.print_exc()

def run_tiled(m, img, tile=128, pad=16):
    h, w = img.shape[:2]; S = tile + 2 * pad
    x = torch.from_numpy(img.astype(np.float32) / 255).permute(2, 0, 1)[None]
    out = np.zeros((h * 4, w * 4, 3), np.float32)
    for y0 in range(0, h, tile):
        for x0 in range(0, w, tile):
            ys = np.clip(np.arange(y0 - pad, y0 - pad + S), 0, h - 1)
            xs = np.clip(np.arange(x0 - pad, x0 - pad + S), 0, w - 1)
            t = x[:, :, ys][:, :, :, xs]
            with torch.no_grad():
                o = m(t)[0].permute(1, 2, 0).clamp(0, 1).numpy()
            vh, vw = min(tile, h - y0), min(tile, w - x0)
            out[y0*4:(y0+vh)*4, x0*4:(x0+vw)*4] = o[pad*4:(pad+vh)*4, pad*4:(pad+vw)*4]
    return (out * 255).round().astype(np.uint8)

img = cv2.cvtColor(cv2.imread('tools/ntire/test_low.jpg'), cv2.COLOR_BGR2RGB)
for name, m in models.items():
    t = time.time(); o = run_tiled(m, img); print(name, 'cpu time', round(time.time() - t, 1), 's')
    cv2.imwrite(f'out/{name}.png', cv2.cvtColor(o, cv2.COLOR_RGB2BGR))
    # ONNX export with a fixed 160x160 tile (multiple of 16)
    try:
        dummy = torch.rand(1, 3, 160, 160)
        torch.onnx.export(m, dummy, f'onnx/{name}_x4_160.onnx', input_names=['input'], output_names=['output'],
                          opset_version=17, do_constant_folding=True, dynamo=False)
        import onnxruntime as ort
        s = ort.InferenceSession(f'onnx/{name}_x4_160.onnx', providers=['CPUExecutionProvider'])
        r = s.run(None, {'input': dummy.numpy()})[0]
        with torch.no_grad(): ref = m(dummy).numpy()
        print(name, 'onnx ok', r.shape, 'maxdiff', float(np.abs(r - ref).max()), 'size MB', os.path.getsize(f'onnx/{name}_x4_160.onnx') / 1e6)
    except Exception:
        import traceback; traceback.print_exc()
cv2.imwrite('out/input_bicubic.png', cv2.resize(cv2.imread('tools/ntire/test_low.jpg'), (1024, 1024), interpolation=cv2.INTER_CUBIC))
