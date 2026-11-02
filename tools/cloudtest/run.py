# Tries free Hugging Face Spaces on a public stock face (no personal photos).
import sys, shutil, time, traceback
from gradio_client import Client, handle_file
import os
imgs = ['tools/cloudtest/face_lowq.jpg', 'tools/cloudtest/face_1024.jpg']
spaces = sys.argv[1:] or ['finegrain/finegrain-image-enhancer']
for sp in spaces:
    print('=' * 20, sp, flush=True)
    try:
        c = Client(sp)
        c.view_api(print_info=True)
    except Exception:
        traceback.print_exc(); continue
    if sp.startswith('finegrain'):
      for img in imgs:
       for den in (0.25, 0.35):
        t = time.time()
        try:
            r = c.predict(handle_file(img), "masterpiece, best quality, highres, detailed skin texture, sharp", "worst quality, low quality, normal quality, blurry, plastic skin",
                          42, 2, 0.6, 1.0, 6, 112, 144, den, 18, "DDIM", api_name="/process")
            print('result', r, f'{time.time()-t:.1f}s')
            out = r[1] if isinstance(r, (list, tuple)) else r
            if isinstance(out, (list, tuple)): out = out[-1]
            if isinstance(out, dict): out = out.get('path') or out.get('value')
            shutil.copy(out, 'out/' + os.path.basename(img)[:-4] + f'_d{den}.webp')
        except Exception:
            traceback.print_exc()
