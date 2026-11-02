# Tries free Hugging Face Spaces on a public stock face (no personal photos).
import sys, shutil, time, traceback
from gradio_client import Client, handle_file
img = 'tools/cloudtest/face_lowq.jpg'
spaces = sys.argv[1:] or ['finegrain/finegrain-image-enhancer']
for sp in spaces:
    print('=' * 20, sp, flush=True)
    try:
        c = Client(sp)
        c.view_api(print_info=True)
    except Exception:
        traceback.print_exc(); continue
    if sp.startswith('finegrain'):
        t = time.time()
        try:
            r = c.predict(handle_file(img), "masterpiece, best quality, highres", "worst quality, low quality, normal quality",
                          42, 2, 0.6, 1.0, 6, 112, 144, 0.35, 18, "DDIM", api_name="/process")
            print('result', r, f'{time.time()-t:.1f}s')
            out = r[1] if isinstance(r, (list, tuple)) else r
            if isinstance(out, (list, tuple)): out = out[-1]
            if isinstance(out, dict): out = out.get('path') or out.get('value')
            shutil.copy(out, 'out/' + sp.replace('/', '_') + '.png')
        except Exception:
            traceback.print_exc()
