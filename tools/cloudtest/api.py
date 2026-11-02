import sys, traceback
from gradio_client import Client
for sp in sys.argv[1:]:
    print('=' * 20, sp, flush=True)
    try:
        Client(sp).view_api(print_info=True)
    except Exception as e:
        print('ERROR', type(e).__name__, str(e)[:300])
