import io
import os
import sys
import time
import asyncio
import subprocess
from PIL import Image
from oscrypto import util as crypto_utils
from aiohttp import web
from io import BytesIO
from imagehash import phash
from collections import deque

SERVER_DIR_PATH = os.path.dirname(os.path.realpath(__file__))
BASE_PATH = os.path.dirname(os.path.dirname(SERVER_DIR_PATH))

VALID_LANGUAGES = {
    'CHS': 'Chinese (Simplified)',
    'CHT': 'Chinese (Traditional)',
    'CSY': 'Czech',
    'NLD': 'Dutch',
    'ENG': 'English',
    'FRA': 'French',
    'DEU': 'German',
    'HUN': 'Hungarian',
    'ITA': 'Italian',
    'JPN': 'Japanese',
    'KOR': 'Korean',
    'PLK': 'Polish',
    'PTB': 'Portuguese (Brazil)',
    'ROM': 'Romanian',
    'RUS': 'Russian',
    'ESP': 'Spanish',
    'TRK': 'Turkish',
    'UKR': 'Ukrainian',
    'VIN': 'Vietnamese',
}
VALID_DETECTORS = set(['default', 'ctd'])
VALID_DIRECTIONS = set(['auto', 'h', 'v'])
VALID_TRANSLATORS = set(['youdao', 'baidu', 'google', 'deepl', 'papago', 'offline', 'none', 'original'])

MAX_ONGOING_TASKS = 1
MAX_IMAGE_SIZE_PX = 8000**2

# Time to wait for web client to send a request to /task-state request
# before that web clients task gets removed from the queue
WEB_CLIENT_TIMEOUT = -1

# Time before finished tasks get removed from memory
FINISHED_TASK_REMOVE_TIMEOUT = 1800

# TODO: Make to dict with translator client id as key for support of multiple translator clients
ONGOING_TASKS = []
NONCE = ''
QUEUE = deque()
TASK_DATA = {}
TASK_STATES = {}
DEFAULT_TRANSLATION_PARAMS = {}

app = web.Application(client_max_size = 1024 * 1024 * 50)
routes = web.RouteTableDef()


def constant_compare(a, b):
    if isinstance(a, str):
        a = a.encode('utf-8')
    if isinstance(b, str):
        b = b.encode('utf-8')
    if not isinstance(a, bytes) or not isinstance(b, bytes):
        return False
    if len(a) != len(b):
        return False

    result = 0
    for x, y in zip(a, b):
        result |= x ^ y
    return result == 0

@routes.get("/")
async def index_async(request):
    with open(os.path.join(SERVER_DIR_PATH, 'ui.html'), 'r', encoding='utf8') as fp:
        return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/manual")
async def index_async(request):
    with open(os.path.join(SERVER_DIR_PATH, 'manual.html'), 'r', encoding='utf8') as fp:
        return web.Response(text=fp.read(), content_type='text/html')

@routes.get("/result/{taskid}")
async def result_async(request):
    im = Image.open("result/" + request.match_info.get('taskid') + "/final.png")
    stream = BytesIO()
    im.save(stream, "WebP", quality=100)
    stream.seek(0)
    return web.Response(body=stream.read(), content_type='image/webp')


@routes.get("/queue-size")
async def queue_size_async(request):
    return web.json_response({'size' : len(QUEUE)})

async def handle_post(request):
    data = await request.post()
    detection_size = None
    selected_translator = 'youdao'
    target_language = 'CHS'
    detector = 'default'
    direction = 'auto'
    ocr = '48px_ctc'
    inpainter = 'lama_mpe'
    upscaler = 'esrgan'
    upscale_ratio = None
    translator_chain = None
    det_auto_rotate = False
    det_invert = False
    det_gamma_correct = False
    inpainting_size = 2048
    unclip_ratio = 2.3
    box_threshold = 0.7
    text_threshold = 0.5
    text_mag_ratio = 1
    font_size_offset = 0
    font_size_minimum = -1
    revert_upscaling = False
    manga2eng = False
    capitalize = False
    mtpe = False

    if 'ocr' in data:
        ocr = data['ocr']
        if ocr not in ['32px', '48px_ctc']:
            ocr = None
    
    if 'inpainter' in data:
        inpainter = data['inpainter']
        if inpainter not in ['default', 'lama_mpe', 'sd', 'none', 'original']:
            inpainter = 'default'
    
    if 'upscaler' in data:
        upscaler = data['upscaler']
        if upscaler not in ['waifu2x', 'esrgan']:
            upscaler = None
            
    if 'upscale_ratio' in data:
        upscale_ratio = int(data['upscale_ratio'])
        if upscale_ratio < 1:
            upscale_ratio = None
    
    if 'det_auto_rotate' in data:
        det_auto_rotate = data['det_auto_rotate'] == 'true'
        
    if 'det_invert' in data:
        det_invert = data['det_invert'] == 'true'
        
    if 'det_gamma_correct' in data:
        det_gamma_correct = data['det_gamma_correct'] == 'true'
        
    if 'inpainting_size' in data:
        inpainting_size = int(data['inpainting_size'])
        if inpainting_size < 1:
            inpainting_size = 2048
    
    if 'unclip_ratio' in data:
        unclip_ratio = float(data['unclip_ratio'])
        if unclip_ratio < 1:
            unclip_ratio = 2.3
            
    if 'box_threshold' in data:
        box_threshold = float(data['box_threshold'])
        if box_threshold < 0:
            box_threshold = 0.7
            
    if 'text_threshold' in data:
        text_threshold = float(data['text_threshold'])
        if text_threshold < 0:
            text_threshold = 0.5
            
    if 'text_mag_ratio' in data:
        text_mag_ratio = int(data['text_mag_ratio'])
        if text_mag_ratio < 0:
            text_mag_ratio = 1
            
    if 'font_size_offset' in data:
        font_size_offset = int(data['font_size_offset'])
        if font_size_offset < 0:
            font_size_offset = 0
            
    if 'font_size_minimum' in data:
        font_size_minimum = int(data['font_size_minimum'])
        if font_size_minimum < 0:
            font_size_minimum = -1
            
    if 'revert_upscaling' in data:
        revert_upscaling = data['revert_upscaling'] == 'true'
        
    if 'manga2eng' in data:
        manga2eng = data['manga2eng'] == 'true'
    
    if 'capitalize' in data:
        capitalize = data['capitalize'] == 'true'
        
    if 'mtpe' in data:
        mtpe = data['mtpe'] == 'true'
    
    if 'translator_chain' in data:
        translator_chain = data['translator_chain']
    
    if 'detection_size' in data:
        detection_size = int(data['detection_size'])
    
    if 'tgt_lang' in data:
        target_language = data['tgt_lang'].upper()
        # TODO: move dicts to their own files to reduce load time
        if target_language not in VALID_LANGUAGES:
            target_language = 'CHS'

    if 'detector' in data:
        detector = data['detector'].lower()
        if detector not in VALID_DETECTORS:
            detector = 'default'

    if 'direction' in data:
        direction = data['direction'].lower()
        if direction not in VALID_DIRECTIONS:
            direction = 'auto'

    if 'translator' in data:
        selected_translator = data['translator'].lower()
        if selected_translator not in VALID_TRANSLATORS:
            selected_translator = 'youdao'

    if 'size' in data:
        size_text = data['size'].upper()
        if size_text == 'S':
            detection_size = 1024
        elif size_text == 'M':
            detection_size = 1536
        elif size_text == 'L':
            detection_size = 2048
        elif size_text == 'X':
            detection_size = 2560

    if 'file' in data:
        file_field = data['file']
        content = file_field.file.read()
    elif 'url' in data:
        from aiohttp import ClientSession
        async with ClientSession() as session:
            async with session.get(data['url']) as resp:
                if resp.status == 200:
                    content = await resp.read()
                else:
                    return web.json_response({'status': 'error'})
    else:
        return web.json_response({'status': 'error'})
    try:
        img = Image.open(io.BytesIO(content))
        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width * img.height > MAX_IMAGE_SIZE_PX:
            return web.json_response({'status': 'error-too-large'})
        
    except Exception:
        print('Image corrupt', file=sys.stderr)
        return web.json_response({'status': 'error-img-corrupt'})
    
    return img, detection_size, selected_translator, target_language, detector, direction, ocr, inpainter,\
            upscaler, upscale_ratio, translator_chain, det_auto_rotate, inpainting_size,\
            unclip_ratio, box_threshold, text_threshold, text_mag_ratio, font_size_offset, font_size_minimum,\
            revert_upscaling, manga2eng, capitalize, mtpe, det_invert, det_gamma_correct


@routes.post("/run")
async def run_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = f'{phash(img, hash_size = 16)}-{size}-{selected_translator}-{target_language}-{detector}-{direction}'
    print(f'New `run` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.png'):
        return web.json_response({'task_id' : task_id, 'status': 'successful'})
    # elif os.path.exists(f'result/{task_id}'):
    #     # either image is being processed or error occurred
    #     if task_id not in TASK_STATES:
    #         # error occurred
    #         return web.json_response({'state': 'error'})
    else:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        now = time.time()
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'created_at': now,
            'requested_at': now,
        }
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
    while True:
        await asyncio.sleep(0.1)
        if task_id not in TASK_STATES:
            break
        state = TASK_STATES[task_id]
        if state['finished']:
            break
    return web.json_response({'task_id': task_id, 'status': 'successful' if state['finished'] else state['info']})


@routes.get("/task-internal")
async def get_task_async(request):
    """
    Called by the translator to get a translation task.
    """
    global NONCE, ONGOING_TASKS, DEFAULT_TRANSLATION_PARAMS
    if constant_compare(request.rel_url.query.get('nonce'), NONCE):
        if len(QUEUE) > 0 and len(ONGOING_TASKS) < MAX_ONGOING_TASKS:
            task_id = QUEUE.popleft()
            if task_id in TASK_DATA:
                data = TASK_DATA[task_id]
                for p, default_value in DEFAULT_TRANSLATION_PARAMS.items():
                    current_value = data.get(p)
                    data[p] = current_value if current_value is not None else default_value
                if not TASK_DATA[task_id].get('manual', False):
                    ONGOING_TASKS.append(task_id)
                return web.json_response({'task_id': task_id, 'data': data})
            else:
                return web.json_response({})
        else:
            return web.json_response({})
    return web.json_response({})

# async def machine_trans_task(task_id, texts, translator = 'youdao', target_language = 'CHS'):
#     print('translator', translator)
#     print('target_language', target_language)
#     if task_id not in TASK_DATA:
#         TASK_DATA[task_id] = {}
#     if texts:
#         success = False
#         for _ in range(10):
#             try:
#                 TASK_DATA[task_id]['trans_result'] = await asyncio.wait_for(dispatch_translation(translator, 'auto', target_language, texts), timeout = 15)
#                 success = True
#                 break
#             except Exception as ex:
#                 continue
#         if not success:
#             TASK_DATA[task_id]['trans_result'] = 'error'
#     else:
#         TASK_DATA[task_id]['trans_result'] = []

async def manual_trans_task(task_id, texts):
    if task_id not in TASK_DATA:
        TASK_DATA[task_id] = {}
    if texts:
        TASK_DATA[task_id]['trans_request'] = [{'s': txt, 't': ''} for txt in texts]
    else:
        TASK_DATA[task_id]['trans_result'] = []
        print('manual translation complete')

@routes.post("/post-translation-result")
async def post_translation_result(request):
    rqjson = (await request.json())
    if 'trans_result' in rqjson and 'task_id' in rqjson:
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            trans_result = [r['t'] for r in rqjson['trans_result']]
            TASK_DATA[task_id]['trans_result'] = trans_result
            while True:
                await asyncio.sleep(0.1)
                if TASK_STATES[task_id]['info'].startswith('error'):
                    ret = web.json_response({'task_id': task_id, 'status': 'error'})
                    break
                if TASK_STATES[task_id]['finished']:
                    ret = web.json_response({'task_id': task_id, 'status': 'successful'})
                    break
            # remove old tasks
            del TASK_STATES[task_id]
            del TASK_DATA[task_id]
            return ret
    return web.json_response({})

@routes.post("/request-translation-internal")
async def request_translation_internal(request):
    global NONCE
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if TASK_DATA[task_id].get('manual', False):
                # manual translation
                asyncio.gather(manual_trans_task(task_id, rqjson['texts']))
            # else:
            #     # using machine translation
            #     asyncio.gather(machine_trans_task(task_id, rqjson['texts'], TASK_DATA[task_id]['translator'], TASK_DATA[task_id]['tgt']))
    return web.json_response({})

@routes.post("/get-translation-result-internal")
async def get_translation_internal(request):
    global NONCE
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_DATA:
            if 'trans_result' in TASK_DATA[task_id]:
                return web.json_response({'result': TASK_DATA[task_id]['trans_result']})
    return web.json_response({})

@routes.get("/task-state")
async def get_task_state_async(request):
    """
    Web API for getting the state of an on-going translation task from the website.

    Is periodically called from ui.html. Once it returns a finished state,
    the web client will try to fetch the corresponding image through /result/<task_id>
    """
    task_id = request.query.get('taskid')
    if task_id and task_id in TASK_STATES and task_id in TASK_DATA:
        state = TASK_STATES[task_id]
        data = TASK_DATA[task_id]
        res_dict = {
            'state': state['info'],
            'finished': state['finished'],
        }
        data['requested_at'] = time.time()
        try:
            res_dict['waiting'] = QUEUE.index(task_id) + 1
        except Exception:
            res_dict['waiting'] = 0
        res = web.json_response(res_dict)

        return res
    return web.json_response({'state': 'error'})

@routes.post("/task-update-internal")
async def post_task_update_async(request):
    """
    Lets the translator update the task state it is working on.
    """
    global NONCE, ONGOING_TASKS
    rqjson = (await request.json())
    if constant_compare(rqjson.get('nonce'), NONCE):
        task_id = rqjson['task_id']
        if task_id in TASK_STATES and task_id in TASK_DATA:
            TASK_STATES[task_id] = {
                'info': rqjson['state'],
                'finished': rqjson['finished'],
            }
            if rqjson['finished'] and not TASK_DATA[task_id].get('manual', False):
                try:
                    i = ONGOING_TASKS.index(task_id)
                    ONGOING_TASKS.pop(i)
                except ValueError:
                    pass
            print(f'Task state {task_id} to {TASK_STATES[task_id]}')
    return web.json_response({})

@routes.post("/submit")
async def submit_async(request):
    """Adds new task to the queue. Called by web client in ui.html when submitting an image."""
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction, ocr, inpainter,\
        upscaler, upscale_ratio, translator_chain, det_auto_rotate, inpainting_size,\
        unclip_ratio, box_threshold, text_threshold, text_mag_ratio, font_size_offset, font_size_minimum,\
        revert_upscaling, manga2eng, capitalize, mtpe, det_invert, det_gamma_correct = x
    else:
        return x
    task_id = f'{phash(img, hash_size = 16)}-{size}-{selected_translator}-{target_language}-{detector}-{direction}'
    now = time.time()
    print(f'New `submit` task {task_id}')
    if os.path.exists(f'result/{task_id}/final.png'):
        TASK_STATES[task_id] = {
            'info': 'saved',
            'finished': True,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'ocr': ocr,
            'inpainter': inpainter,
            'upscaler': upscaler,
            'upscale_ratio': upscale_ratio,
            'translator_chain': translator_chain,
            'det_auto_rotate': det_auto_rotate,
            'det_invert': det_invert,
            'det_gamma_correct': det_gamma_correct,
            'inpainting_size': inpainting_size,
            'unclip_ratio': unclip_ratio,
            'box_threshold': box_threshold,
            'text_threshold': text_threshold,
            'text_mag_ratio': text_mag_ratio,
            'font_size_offset': font_size_offset,
            'font_size_minimum': font_size_minimum,
            'revert_upscaling': revert_upscaling,
            'manga2eng': manga2eng,
            'capitalize': capitalize,
            'mtpe': mtpe,
            'created_at': now,
            'requested_at': now,
        }
    elif task_id not in TASK_DATA or task_id not in TASK_STATES:
        os.makedirs(f'result/{task_id}/', exist_ok=True)
        img.save(f'result/{task_id}/input.png')
        QUEUE.append(task_id)
        TASK_STATES[task_id] = {
            'info': 'pending',
            'finished': False,
        }
        TASK_DATA[task_id] = {
            'detection_size': size,
            'translator': selected_translator,
            'target_lang': target_language,
            'detector': detector,
            'direction': direction,
            'ocr': ocr,
            'inpainter': inpainter,
            'upscaler': upscaler,
            'upscale_ratio': upscale_ratio,
            'translator_chain': translator_chain,
            'det_auto_rotate': det_auto_rotate,
            'det_invert': det_invert,
            'det_gamma_correct': det_gamma_correct,
            'inpainting_size': inpainting_size,
            'unclip_ratio': unclip_ratio,
            'box_threshold': box_threshold,
            'text_threshold': text_threshold,
            'text_mag_ratio': text_mag_ratio,
            'font_size_offset': font_size_offset,
            'font_size_minimum': font_size_minimum,
            'revert_upscaling': revert_upscaling,
            'manga2eng': manga2eng,
            'capitalize': capitalize,
            'mtpe': mtpe,
            'created_at': now,
            'requested_at': now,
        }
    return web.json_response({'task_id': task_id, 'status': 'successful'})

@routes.post("/manual-translate")
async def manual_translate_async(request):
    x = await handle_post(request)
    if isinstance(x, tuple):
        img, size, selected_translator, target_language, detector, direction = x
    else:
        return x
    task_id = crypto_utils.rand_bytes(16).hex()
    print(f'New `manual-translate` task {task_id}')
    os.makedirs(f'result/{task_id}/', exist_ok=True)
    img.save(f'result/{task_id}/input.png')
    now = time.time()
    QUEUE.append(task_id)
    TASK_DATA[task_id] = {
        'detection_size': size,
        'manual': True,
        'detector': detector,
        'direction': direction,
        'created_at': now,
        'requested_at': now,
    }
    TASK_STATES[task_id] = {
        'info': 'pending',
        'finished': False,
    }
    while True:
        await asyncio.sleep(1)
        if 'trans_request' in TASK_DATA[task_id]:
            return web.json_response({'task_id' : task_id, 'status': 'pending', 'trans_result': TASK_DATA[task_id]['trans_request']})
        if TASK_STATES[task_id]['info'].startswith('error'):
            break
        if TASK_STATES[task_id]['finished']:
            # no texts detected
            return web.json_response({'task_id' : task_id, 'status': 'successful'})
    return web.json_response({'task_id' : task_id, 'status': 'error'})

app.add_routes(routes)


def generate_nonce():
    return crypto_utils.rand_bytes(16).hex()

def start_translator_client_proc(host: str, port: int, nonce: str, params: dict):
    global MAX_ONGOING_TASKS
    os.environ['MT_WEB_NONCE'] = nonce
    cmds = [
        sys.executable,
        '-m', 'manga_translator',
        '--mode', 'web_client',
        '--host', host,
        '--port', str(port),
    ]
    if params.get('use_cuda', False):
        cmds.append('--use-cuda')
    if params.get('use_cuda_limited', False):
        cmds.append('--use-cuda-limited')
    if params.get('ignore_errors', False):
        cmds.append('--ignore-errors')
    if params.get('verbose', False):
        cmds.append('--verbose')
        
        
    MAX_ONGOING_TASKS = params.get('max_ongoing_tasks', 1)

    proc = subprocess.Popen(cmds, cwd=BASE_PATH)
    return proc

async def start_async_app(host: str, port: int, nonce: str, translation_params: dict = None):
    global NONCE, DEFAULT_TRANSLATION_PARAMS
    # Secret to secure communication between webserver and translator clients
    NONCE = nonce
    DEFAULT_TRANSLATION_PARAMS = translation_params or {}
    
    # Schedule web server to run
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()
    print(f'Serving up app on http://{host}:{port}')

    return runner, site

async def dispatch(host: str, port: int, nonce: str = None, translation_params: dict = None):
    global ONGOING_TASKS

    if nonce is None:
        nonce = os.getenv('MT_WEB_NONCE', generate_nonce())

    runner, site = await start_async_app(host, port, nonce, translation_params)

    # Create client process that will execute translation tasks
    print()
    client_process = start_translator_client_proc(host, port, nonce, translation_params)

    try:
        while True:
            await asyncio.sleep(1)

            # Restart client if OOM or similar errors occured
            if client_process.poll() is not None:
                # if client_process.poll() == 0:
                #     break
                print('Restarting translator process')
                if len(ONGOING_TASKS) > 0:
                    task_id = ONGOING_TASKS.pop(0)
                    state = TASK_STATES[task_id]
                    state['info'] = 'error'
                    state['finished'] = True
                client_process = start_translator_client_proc(host, port, nonce, translation_params)

            # Filter queued and finished tasks
            now = time.time()
            to_del_task_ids = set()
            for tid, s in TASK_STATES.items():
                d = TASK_DATA[tid]
                # Remove finished tasks after 30 minutes
                if s['finished'] and now - d['created_at'] > FINISHED_TASK_REMOVE_TIMEOUT:
                    to_del_task_ids.add(tid)

                # Remove queued tasks without web client
                elif WEB_CLIENT_TIMEOUT >= 0:
                    if tid not in ONGOING_TASKS and not s['finished'] and now - d['requested_at'] > WEB_CLIENT_TIMEOUT:
                        print('REMOVING TASK', tid)
                        to_del_task_ids.add(tid)
                        try:
                            QUEUE.remove(tid)
                        except Exception:
                            pass

            for tid in to_del_task_ids:
                del TASK_STATES[tid]
                del TASK_DATA[tid]

    except:
        if client_process.poll() is None:
            # client_process.terminate()
            client_process.kill()
        await runner.cleanup()
        raise

if __name__ == '__main__':
    from ..args import parser

    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        runner, site = loop.run_until_complete(dispatch(args.host, args.port, translation_params=vars(args)))
    except KeyboardInterrupt:
        pass
