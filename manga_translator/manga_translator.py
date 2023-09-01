import asyncio
import base64
import io
import random
import string
import cv2
from omegaconf import OmegaConf
import py3langid as langid
import requests
import os
import re
import torch
import time
import logging
import numpy as np
import gradio as gr
import zipfile
import hashlib
from PIL import Image
from typing import List, Tuple
from aiohttp import web
from marshmallow import Schema, fields, ValidationError
from zipfile import BadZipFile

from manga_translator.utils.threading import Throttler

from .args import DEFAULT_ARGS
from .utils import (
    BASE_PATH,
    LANGAUGE_ORIENTATION_PRESETS,
    ModelWrapper,
    Context,
    PriorityLock,
    load_image,
    dump_image,
    replace_prefix,
    visualize_textblocks,
    add_file_logger,
    remove_file_logger,
    count_valuable_text,
    rgb2hex,
    get_color_name,
    is_url,
    natural_sort,
    sort_regions,
)

from .detection import DETECTORS, dispatch as dispatch_detection, prepare as prepare_detection
from .upscaling import UPSCALERS, dispatch as dispatch_upscaling, prepare as prepare_upscaling
from .ocr import OCRS, dispatch as dispatch_ocr, prepare as prepare_ocr
from .textline_merge import dispatch as dispatch_textline_merge
from .mask_refinement import dispatch as dispatch_mask_refinement
from .inpainting import INPAINTERS, dispatch as dispatch_inpainting, prepare as prepare_inpainting
from .translators import (
    TRANSLATORS,
    VALID_LANGUAGES,
    LanguageUnsupportedException,
    TranslatorChain,
    dispatch as dispatch_translation,
    prepare as prepare_translation,
)
from .colorization import COLORIZERS, dispatch as dispatch_colorization, prepare as prepare_colorization
from .rendering import dispatch as dispatch_rendering, dispatch_eng_render
from .save import save_result


# Will be overwritten by __main__.py if module is being run directly (with python -m)
logger = logging.getLogger('manga_translator')

def set_main_logger(l):
    global logger
    logger = l

class TranslationInterrupt(Exception):
    """
    Can be raised from within a progress hook to prematurely terminate
    the translation.
    """
    pass

class MangaTranslator():

    def __init__(self, params: dict = None):
        self._progress_hooks = []
        self._add_logger_hook()

        params = params or {}
        # Use environment variables to save thresholds for use in ocr
        self.ignore_bubble=int(params.get('ignore_bubble', 0))
        self.verbose = params.get('verbose', False)
        self.ignore_errors = params.get('ignore_errors', False)

        self.device = 'cuda' if params.get('use_cuda', False) else 'cpu'
        self._cuda_limited_memory = params.get('use_cuda_limited', False)
        if self._cuda_limited_memory and not self.using_cuda:
            self.device = 'cuda'
        if self.using_cuda and not torch.cuda.is_available():
            raise Exception('CUDA compatible device could not be found whilst --use-cuda args was set...')

        self.result_sub_folder = ''

    @property
    def using_cuda(self):
        return self.device.startswith('cuda')

    async def translate_path(self, path: str, dest: str = None, params: dict = None):
        """
        Translates an image or folder (recursively) specified through the path.
        """
        path = os.path.expanduser(path)
        # expand path patterns such as *
        if not path:
            raise FileNotFoundError(path)
        path = os.path.abspath(path)
        dest = os.path.abspath(os.path.expanduser(dest)) if dest else ''
        params = params or {}

        # Handle format
        file_ext = params.get('format') or 'png'
        if params.get('save_quality', 100) < 100:
            if not params.get('format'):
                file_ext = 'jpg'
            elif params.get('format') != 'jpg':
                raise ValueError('--save-quality of lower than 100 is only supported for .jpg files')

        if os.path.isfile(path):
            # Determine destination file path
            if not dest:
                # Use the same folder as the source
                p, _ = os.path.splitext(path)
                _dest = f'{p}-translated.{file_ext}'
            elif not os.path.basename(dest):
                p, _ = os.path.splitext(os.path.basename(path))
                # If the folders differ use the original filename from the source
                if os.path.dirname(path) != dest:
                    _dest = os.path.join(dest, f'{p}.{file_ext}')
                else:
                    _dest = os.path.join(dest, f'{p}-translated.{file_ext}')
            else:
                p, _ = os.path.splitext(dest)
                _dest = f'{p}.{file_ext}'
            await self._translate_file(path, _dest, params)

        elif os.path.isdir(path):
            # Determine destination folder path
            if path[-1] == '\\' or path[-1] == '/':
                path = path[:-1]
            _dest = dest or path + '-translated'
            if os.path.exists(_dest) and not os.path.isdir(_dest):
                raise FileExistsError(_dest)

            translated_count = 0
            for root, subdirs, files in os.walk(path):
                files = natural_sort(files)
                dest_root = replace_prefix(root, path, _dest)
                os.makedirs(dest_root, exist_ok=True)
                for f in files:
                    if f.lower() == '.thumb':
                        continue

                    file_path = os.path.join(root, f)
                    output_dest = replace_prefix(file_path, path, _dest)
                    p, _ = os.path.splitext(output_dest)
                    output_dest = f'{p}.{file_ext}'

                    if await self._translate_file(file_path, output_dest, params):
                        translated_count += 1
            if translated_count == 0:
                logger.info('No further untranslated files found. Use --overwrite to write over existing translations.')
            else:
                logger.info(f'Done. Translated {translated_count} image{"" if translated_count == 1 else "s"}')

    async def _translate_file(self, path: str, dest: str, params: dict):
        if not params.get('overwrite') and os.path.exists(dest):
            logger.info(f'Skipping as already translated: {dest}')
            await self._report_progress('saved', True)
            return True
        logger.info(f'Translating: "{path}"')

        try:
            img = Image.open(path)
            img.verify()
            img = Image.open(path)
        except Exception:
            logger.warn(f'Failed to open image: {path}')
            return False

        translation_dict = await self.translate(img, params)
        result = translation_dict.result

        # Save result
        if result:
            logger.info(f'Saving "{dest}"')
            save_result(result, dest, translation_dict)
            await self._report_progress('saved', True)

            if translation_dict.save_text or translation_dict.save_text_file or translation_dict.prep_manual:
                if translation_dict.prep_manual:
                    # Save original image next to translated
                    p, ext = os.path.splitext(dest)
                    img_filename = p + '-orig' + ext
                    img_path = os.path.join(os.path.dirname(dest), img_filename)
                    img.save(img_path, quality=translation_dict.save_quality)
                if translation_dict.text_regions:
                    self._save_text_to_file(dest, translation_dict)
        else:
            return False
        return True

    async def translate(self, image: Image.Image, params: dict = None) -> Context:
        """
        Translates a PIL image from a manga. Returns dict with result and intermediates of translation.
        Default params are taken from args.py.

        ```py
        translation_dict = await translator.translate(image)
        result = translation_dict.result
        ```
        """
        # TODO: Take list of images to speed up batch processing

        # Turn dict to context to make values also accessible through params.<property>
        params = params or {}
        ctx = Context(**params)
        self._preprocess_params(ctx)

        if ctx.gpt_config:
            from .translators import chatgpt
            chatgpt.CONFIG = OmegaConf.load(ctx.gpt_config)
        if ctx.model_dir:
            ModelWrapper._MODEL_DIR = ctx.model_dir

        ctx.input = image
        ctx.result = None

        attempts = 0
        while ctx.attempts == -1 or attempts < ctx.attempts + 1:
            if attempts > 0:
                logger.info(f'Retrying translation! Attempt {attempts}'
                            + (f' of {ctx.attempts}' if ctx.attempts != -1 else ''))
            try:
                # preload and download models (not strictly necessary, remove to lazy load)
                logger.info('Loading models')
                if ctx.upscale_ratio:
                    await prepare_upscaling(ctx.upscaler)
                await prepare_detection(ctx.detector)
                await prepare_ocr(ctx.ocr, self.device)
                await prepare_inpainting(ctx.inpainter, self.device)
                await prepare_translation(ctx.translator)
                if ctx.colorizer:
                    await prepare_colorization(ctx.colorizer)

                # translate
                return await self._translate(ctx)
            except TranslationInterrupt:
                break
            except Exception as e:
                if isinstance(e, LanguageUnsupportedException):
                    await self._report_progress('error-lang', True)
                else:
                    await self._report_progress('error', True)
                if not self.ignore_errors and not (ctx.attempts == -1 or attempts < ctx.attempts):
                    raise
                else:
                    logger.error(f'{e.__class__.__name__}: {e}',
                                 exc_info=e if self.verbose else None)
            attempts += 1
        return ctx

    def _preprocess_params(self, ctx: Context):
        # params auto completion
        for arg in DEFAULT_ARGS:
            ctx.setdefault(arg, DEFAULT_ARGS[arg])

        if 'direction' not in ctx:
            if ctx.force_horizontal:
                ctx.direction = 'h'
            elif ctx.force_vertical:
                ctx.direction = 'v'
            else:
                ctx.direction = 'auto'
        if 'alignment' not in ctx:
            if ctx.align_left:
                ctx.alignment = 'left'
            elif ctx.align_center:
                ctx.alignment = 'center'
            elif ctx.align_right:
                ctx.alignment = 'right'
            else:
                ctx.alignment = 'auto'
        if ctx.prep_manual:
            ctx.renderer = 'none'
        ctx.setdefault('renderer', 'manga2eng' if ctx.manga2eng else 'default')

        if ctx.selective_translation is not None:
            ctx.selective_translation.target_lang = ctx.target_lang
            ctx.translator = ctx.selective_translation
        elif ctx.translator_chain is not None:
            ctx.target_lang = ctx.translator_chain.langs[-1]
            ctx.translator = ctx.translator_chain
        else:
            ctx.translator = TranslatorChain(f'{ctx.translator}:{ctx.target_lang}')

        if ctx.filter_text:
            ctx.filter_text = re.compile(ctx.filter_text)

    async def _translate(self, ctx: Context) -> Context:

        if ctx.colorizer:
            await self._report_progress('colorizing')
            ctx.img_colorized = await self._run_colorizer(ctx)
        else:
            ctx.img_colorized = ctx.input

        # The default text detector doesn't work very well on smaller images, might want to
        # consider adding automatic upscaling on certain kinds of small images.
        if ctx.upscale_ratio:
            await self._report_progress('upscaling')
            ctx.upscaled = await self._run_upscaling(ctx)
        else:
            ctx.upscaled = ctx.img_colorized

        ctx.img_rgb, ctx.img_alpha = load_image(ctx.upscaled)

        await self._report_progress('detection')
        ctx.textlines, ctx.mask_raw, ctx.mask = await self._run_detection(ctx)
        if self.verbose:
            cv2.imwrite(self._result_path('mask_raw.png'), ctx.mask_raw)

        if not ctx.textlines:
            await self._report_progress('skip-no-regions', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return ctx
        if self.verbose:
            img_bbox_raw = np.copy(ctx.img_rgb)
            for txtln in ctx.textlines:
                cv2.polylines(img_bbox_raw, [txtln.pts], True, color=(255, 0, 0), thickness=2)
            cv2.imwrite(self._result_path('bboxes_unfiltered.png'), cv2.cvtColor(img_bbox_raw, cv2.COLOR_RGB2BGR))

        await self._report_progress('ocr')
        ctx.textlines = await self._run_ocr(ctx)
        if not ctx.textlines:
            await self._report_progress('skip-no-text', True)
            # If no text was found result is intermediate image product
            ctx.result = ctx.upscaled
            return ctx

        await self._report_progress('textline_merge')
        ctx.text_regions = await self._run_textline_merge(ctx)

        if self.verbose:
            bboxes = visualize_textblocks(cv2.cvtColor(ctx.img_rgb, cv2.COLOR_BGR2RGB), ctx.text_regions)
            cv2.imwrite(self._result_path('bboxes.png'), bboxes)

        await self._report_progress('translating')
        ctx.text_regions = await self._run_text_translation(ctx)

        if not ctx.text_regions:
            await self._report_progress('error-translating', True)
            return ctx

        # Delayed mask refinement to take advantage of the region filtering done after ocr and translation
        if ctx.mask is None:
            await self._report_progress('mask-generation')
            ctx.mask = await self._run_mask_refinement(ctx)

        if self.verbose:
            inpaint_input_img = await dispatch_inpainting('none', ctx.img_rgb, ctx.mask, ctx.inpainting_size, self.using_cuda, self.verbose)
            cv2.imwrite(self._result_path('inpaint_input.png'), cv2.cvtColor(inpaint_input_img, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('mask_final.png'), ctx.mask)

        await self._report_progress('inpainting')
        ctx.img_inpainted = await self._run_inpainting(ctx)

        ctx.gimp_mask = np.dstack((cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR), ctx.mask))

        if self.verbose:
            cv2.imwrite(self._result_path('inpainted.png'), cv2.cvtColor(ctx.img_inpainted, cv2.COLOR_RGB2BGR))

        await self._report_progress('rendering')
        ctx.img_rendered = await self._run_text_rendering(ctx)

        await self._report_progress('finished', True)
        ctx.result = dump_image(ctx.img_rendered, ctx.img_alpha)

        if ctx.revert_upscaling:
            await self._report_progress('downscaling')
            ctx.result = ctx.result.resize(ctx.input.size)

        return ctx

    async def _run_colorizer(self, ctx: Context):
        return await dispatch_colorization(ctx.colorizer, device=self.device, image=ctx.input, **ctx)

    async def _run_upscaling(self, ctx: Context):
        return (await dispatch_upscaling(ctx.upscaler, [ctx.img_colorized], ctx.upscale_ratio, self.device))[0]

    async def _run_detection(self, ctx: Context):
        return await dispatch_detection(ctx.detector, ctx.img_rgb, ctx.detection_size, ctx.text_threshold, ctx.box_threshold,
                                        ctx.unclip_ratio, ctx.det_invert, ctx.det_gamma_correct, ctx.det_rotate, ctx.det_auto_rotate,
                                        self.device, self.verbose)

    async def _run_ocr(self, ctx: Context):
        textlines = await dispatch_ocr(ctx.ocr, ctx.img_rgb, ctx.textlines, self.device, self.verbose, self.ignore_bubble)

        # Filter out regions by original text
        new_textlines = []
        for textline in textlines:
            text = textline.text
            if text.isnumeric() \
                or (ctx.filter_text and re.search(ctx.filter_text, text)) \
                or count_valuable_text(text) < 1 \
                or is_url(text):
                if text.strip():
                    logger.info(f'Filtered out: {text}')
            else:
                new_textlines.append(textline)
        return new_textlines

    async def _run_textline_merge(self, ctx: Context):
        text_regions = await dispatch_textline_merge(ctx.textlines, ctx.img_rgb.shape[1], ctx.img_rgb.shape[0], verbose=self.verbose)
        text_regions = [region for region in text_regions if len(''.join(region.text)) >= ctx.min_text_length]

        # Sort ctd (comic text detector) regions left to right. Otherwise right to left.
        # Sorting will improve text translation quality.
        text_regions = sort_regions(text_regions, right_to_left=True if ctx.detector != 'ctd' else False)
        return text_regions

    async def _run_text_translation(self, ctx: Context):
        translated_sentences = await dispatch_translation(ctx.translator, [region.get_text() for region in ctx.text_regions], ctx.use_mtpe,
                                                          'cpu' if self._cuda_limited_memory else self.device)

        for region, translation in zip(ctx.text_regions, translated_sentences):
            if ctx.uppercase:
                translation = translation.upper()
            elif ctx.lowercase:
                translation = translation.upper()
            region.translation = translation
            region.target_lang = ctx.target_lang
            region._alignment = ctx.alignment
            region._direction = ctx.direction

        # Filter out regions by their translations
        new_text_regions = []
        for region in ctx.text_regions:
            # TODO: Maybe print reasons for filtering
            if not ctx.translator.is_none() and (region.translation.isnumeric() \
                or (ctx.filter_text and re.search(ctx.filter_text, region.translation))
                or (region.get_text().lower().strip() == region.translation.lower().strip())):
                if region.translation.strip():
                    logger.info(f'Filtered out: {region.translation}')
            else:
                new_text_regions.append(region)
        return new_text_regions

    async def _run_mask_refinement(self, ctx: Context):
        return await dispatch_mask_refinement(ctx.text_regions, ctx.img_rgb, ctx.mask_raw, 'fit_text', self.verbose,self.ignore_bubble)

    async def _run_inpainting(self, ctx: Context):
        return await dispatch_inpainting(ctx.inpainter, ctx.img_rgb, ctx.mask, ctx.inpainting_size, self.using_cuda, self.verbose)

    async def _run_text_rendering(self, ctx: Context):
        if ctx.renderer == 'none':
            output = ctx.img_inpainted
        # manga2eng currently only supports horizontal rendering
        elif ctx.renderer == 'manga2eng' and ctx.text_regions and LANGAUGE_ORIENTATION_PRESETS.get(ctx.text_regions[0].target_lang) == 'h':
            output = await dispatch_eng_render(ctx.img_inpainted, ctx.img_rgb, ctx.text_regions, ctx.font_path)
        else:
            output = await dispatch_rendering(ctx.img_inpainted, ctx.text_regions, ctx.font_path, ctx.font_size, ctx.font_size_offset,
                                              ctx.font_size_minimum, ctx.render_mask)
        return output

    def _result_path(self, path: str) -> str:
        """
        Returns path to result folder where intermediate images are saved when using verbose flag
        or web mode input/result images are cached.
        """
        return os.path.join(BASE_PATH, 'result', self.result_sub_folder, path)

    def add_progress_hook(self, ph):
        self._progress_hooks.append(ph)

    async def _report_progress(self, state: str, finished: bool = False):
        for ph in self._progress_hooks:
            await ph(state, finished)

    def _add_logger_hook(self):
        # TODO: Pass ctx to logger hook
        LOG_MESSAGES = {
            'upscaling':            'Running upscaling',
            'detection':            'Running text detection',
            'ocr':                  'Running ocr',
            'mask-generation':      'Running mask refinement',
            'translating':          'Running text translation',
            'rendering':            'Running rendering',
            'colorizing':            'Running colorization',
            'downscaling':          'Running downscaling',
        }
        LOG_MESSAGES_SKIP = {
            'skip-no-regions':      'No text regions! - Skipping',
            'skip-no-text':         'No text regions with text! - Skipping',
            'error-translating':    'Text translator returned empty queries',
        }
        LOG_MESSAGES_ERROR = {
            # 'error-lang':           'Target language not supported by chosen translator',
        }

        async def ph(state, finished):
            if state in LOG_MESSAGES:
                logger.info(LOG_MESSAGES[state])
            elif state in LOG_MESSAGES_SKIP:
                logger.warn(LOG_MESSAGES_SKIP[state])
            elif state in LOG_MESSAGES_ERROR:
                logger.error(LOG_MESSAGES_ERROR[state])

        self.add_progress_hook(ph)

    def _save_text_to_file(self, image_path: str, ctx: Context):
        cached_colors = []

        def identify_colors(fg_rgb: List[int]):
            idx = 0
            for rgb, _ in cached_colors:
                # If similar color already saved
                if abs(rgb[0] - fg_rgb[0]) + abs(rgb[1] - fg_rgb[1]) + abs(rgb[2] - fg_rgb[2]) < 50:
                    break
                else:
                    idx += 1
            else:
                cached_colors.append((fg_rgb, get_color_name(fg_rgb)))
            return idx + 1, cached_colors[idx][1]

        s = f'\n[{image_path}]\n'
        for i, region in enumerate(ctx.text_regions):
            fore, back = region.get_font_colors()
            color_id, color_name = identify_colors(fore)

            s += f'\n-- {i+1} --\n'
            s += f'color: #{color_id}: {color_name} (fg, bg: {rgb2hex(*fore)} {rgb2hex(*back)})\n'
            s += f'text:  {region.get_text()}\n'
            s += f'trans: {region.translation}\n'
            for line in region.lines :
                s += f'coords: {list(line.ravel())}\n'
        s += '\n'

        text_output_file = ctx.text_output_file
        if not text_output_file:
            text_output_file = os.path.join(os.path.dirname(image_path), '_translations.txt')

        with open(text_output_file, 'a', encoding='utf-8') as f:
            f.write(s)


class MangaTranslatorWeb(MangaTranslator):
    """
    Translator client that executes tasks on behalf of the webserver in web_main.py.
    """
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        if self.host == '0.0.0.0':
            self.host = '127.0.0.1'
        self.port = params.get('port', 5003)
        self.nonce = params.get('nonce', '')
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None

    async def _init_connection(self):
        available_translators = []
        from .translators import MissingAPIKeyException, get_translator
        for key in TRANSLATORS:
            try:
                get_translator(key)
                available_translators.append(key)
            except MissingAPIKeyException:
                pass

        data = {
            'nonce': self.nonce,
            'capabilities': {
                'translators': available_translators,
            },
        }
        requests.post(f'http://{self.host}:{self.port}/connect-internal', json=data)

    async def _send_state(self, state: str, finished: bool):
        # wait for translation to be saved first (bad solution?)
        finished = finished and not state == 'finished'
        while True:
            try:
                data = {
                    'task_id': self._task_id,
                    'nonce': self.nonce,
                    'state': state,
                    'finished': finished,
                }
                requests.post(f'http://{self.host}:{self.port}/task-update-internal', json=data, timeout=20)
                break
            except Exception:
                # if translation is finished server has to know
                if finished:
                    continue
                else:
                    break

    def _get_task(self):
        try:
            rjson = requests.get(f'http://{self.host}:{self.port}/task-internal?nonce={self.nonce}', timeout=3600).json()
            return rjson.get('task_id'), rjson.get('data')
        except Exception:
            return None, None

    async def listen(self, translation_params: dict = None):
        """
        Listens for translation tasks from web server.
        """
        logger.info('Waiting for translation tasks')

        await self._init_connection()
        self.add_progress_hook(self._send_state)

        while True:
            self._task_id, self._params = self._get_task()
            if self._params and 'exit' in self._params:
                break
            if not (self._task_id and self._params):
                await asyncio.sleep(0.1)
                continue

            self.result_sub_folder = self._task_id
            logger.info(f'Processing task {self._task_id}')
            if translation_params is not None:
                # Combine default params with params chosen by webserver
                for p, default_value in translation_params.items():
                    current_value = self._params.get(p)
                    self._params[p] = current_value if current_value is not None else default_value
            if self.verbose:
                # Write log file
                log_file = self._result_path('log.txt')
                add_file_logger(log_file)

            # final.jpg will be renamed if format param is set
            await self.translate_path(self._result_path('input.jpg'), self._result_path('final.jpg'), params=self._params)
            print()

            if self.verbose:
                remove_file_logger(log_file)
            self._task_id = None
            self._params = None
            self.result_sub_folder = ''

    async def _run_text_translation(self, ctx: Context):
        # Run machine translation as reference for manual translation (if `--translator=none` is not set)
        text_regions = await super()._run_text_translation(ctx)

        if ctx.get('manual', False):
            logger.info('Waiting for user input from manual translation')
            requests.post(f'http://{self.host}:{self.port}/request-manual-internal', json={
                'task_id': self._task_id,
                'nonce': self.nonce,
                'texts': [r.get_text() for r in ctx.text_regions],
                'translations': [r.translation for r in ctx.text_regions],
            }, timeout=20)

            # wait for at most 1 hour for manual translation
            wait_until = time.time() + 3600
            while time.time() < wait_until:
                ret = requests.post(f'http://{self.host}:{self.port}/get-manual-result-internal', json={
                    'task_id': self._task_id,
                    'nonce': self.nonce
                }, timeout=20).json()
                if 'result' in ret:
                    manual_translations = ret['result']
                    if isinstance(manual_translations, str):
                        if manual_translations == 'error':
                            return []
                    for region, translation in zip(text_regions, manual_translations):
                        region.translation = translation
                        region.target_lang = ctx.translator.langs[-1]
                    break
                await asyncio.sleep(0.1)
        return text_regions


class MangaTranslatorWS(MangaTranslator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.url = params.get('ws_url')
        self.secret = params.get('ws_secret', os.getenv('WS_SECRET', ''))
        self.ignore_errors = params.get('ignore_errors', True)

        self._task_id = None
        self._websocket = None

    async def listen(self, translation_params: dict = None):
        from threading import Thread
        import io
        import aioshutil
        from aiofiles import os
        import websockets
        from .server import ws_pb2

        self._server_loop = asyncio.new_event_loop()
        self.task_lock = PriorityLock()
        self.counter = 0

        async def _send_and_yield(websocket, msg):
            # send message and yield control to the event loop (to actually send the message)
            await websocket.send(msg)
            await asyncio.sleep(0)

        send_throttler = Throttler(0.2)
        send_and_yield = send_throttler.wrap(_send_and_yield)

        async def sync_state(state, finished):
            if self._websocket is None:
                return
            msg = ws_pb2.WebSocketMessage()
            msg.status.id = self._task_id
            msg.status.status = state
            self._server_loop.call_soon_threadsafe(
                asyncio.create_task,
                send_and_yield(self._websocket, msg.SerializeToString())
            )

        self.add_progress_hook(sync_state)

        async def translate(task_id, websocket, image, params):
            async with self.task_lock((1<<31) - params['ws_count']):
                self._task_id = task_id
                self._websocket = websocket
                result = await self.translate(image, params)
                self._task_id = None
                self._websocket = None
            return result
        
        async def server_send_status(websocket, task_id, status):
            msg = ws_pb2.WebSocketMessage()
            msg.status.id = task_id
            msg.status.status = status
            await websocket.send(msg.SerializeToString())
            await asyncio.sleep(0)

        async def server_process_inner(main_loop, logger_task, session, websocket, task) -> Tuple[bool, bool]:
            logger_task.info(f'-- Processing task {task.id}')
            await server_send_status(websocket, task.id, 'pending')

            if self.verbose:
                await aioshutil.rmtree(f'result/{task.id}', ignore_errors=True)
                await os.makedirs(f'result/{task.id}', exist_ok=True)

            params = {
                'target_lang': task.target_language,
                'detector': task.detector,
                'direction': task.direction,
                'translator': task.translator,
                'size': task.size,
                'ws_event_loop': asyncio.get_event_loop(),
                'ws_count': self.counter,
            }
            self.counter += 1

            logger_task.info(f'-- Downloading image from {task.source_image}')
            await server_send_status(websocket, task.id, 'downloading')
            async with session.get(task.source_image) as resp:
                if resp.status == 200:
                    source_image = await resp.read()
                else:
                    msg = ws_pb2.WebSocketMessage()
                    msg.status.id = task.id
                    msg.status.status = 'error-download'
                    await websocket.send(msg.SerializeToString())
                    await asyncio.sleep(0)
                    return False, False

            logger_task.info(f'-- Translating image')
            if translation_params:
                for p, default_value in translation_params.items():
                    current_value = params.get(p)
                    params[p] = current_value if current_value is not None else default_value

            image = Image.open(io.BytesIO(source_image))

            (ori_w, ori_h) = image.size
            if max(ori_h, ori_w) > 1200 :
                params['upscale_ratio'] = 1

            await server_send_status(websocket, task.id, 'preparing')
            # translation_dict = await self.translate(image, params)
            translation_dict = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    translate(task.id, websocket, image, params),
                    main_loop
                )
            )
            await send_throttler.flush()
            
            output: Image.Image = translation_dict.result
            if output is not None:
                await server_send_status(websocket, task.id, 'saving')

                output = output.resize((ori_w, ori_h), resample = Image.LANCZOS)

                img = io.BytesIO()
                output.save(img, format='PNG')
                if self.verbose:
                    output.save(self._result_path('ws_final.png'))

                img_bytes = img.getvalue()
                logger_task.info(f'-- Uploading result to {task.translation_mask}')
                await server_send_status(websocket, task.id, 'uploading')
                async with session.put(task.translation_mask, data=img_bytes) as resp:
                    if resp.status != 200:
                        logger_task.error(f'-- Failed to upload result:')
                        logger_task.error(f'{resp.status}: {resp.reason}')
                        msg = ws_pb2.WebSocketMessage()
                        msg.status.id = task.id
                        msg.status.status = 'error-upload'
                        await websocket.send(msg.SerializeToString())
                        await asyncio.sleep(0)
                        return False, False

            return True, output is not None

        async def server_process(main_loop, session, websocket, task) -> bool:
            logger_task = logger.getChild(f'{task.id}')
            try:
                (success, has_translation_mask) = await server_process_inner(main_loop, logger_task, session, websocket, task)
            except Exception as e:
                logger_task.error(f'-- Task failed with exception:')
                logger_task.error(f'{e.__class__.__name__}: {e}', exc_info=e if self.verbose else None)
                (success, has_translation_mask) = False, False
            finally:
                result = ws_pb2.WebSocketMessage()
                result.finish_task.id = task.id
                result.finish_task.success = success
                result.finish_task.has_translation_mask = has_translation_mask
                await websocket.send(result.SerializeToString())
                await asyncio.sleep(0)
                logger_task.info(f'-- Task finished')

        async def async_server_thread(main_loop):
            from aiohttp import ClientSession, ClientTimeout
            timeout = ClientTimeout(total=30)
            async with ClientSession(timeout=timeout) as session:
                logger_conn = logger.getChild('connection')
                if self.verbose:
                    logger_conn.setLevel(logging.DEBUG)
                async for websocket in websockets.connect(
                    self.url,
                    extra_headers={
                        'x-secret': self.secret,
                    },
                    max_size=1_000_000,
                    logger=logger_conn
                ):
                    bg_tasks = set()
                    try:
                        logger.info('-- Connected to websocket server')

                        async for raw in websocket:
                            # logger.info(f'Got message: {raw}')
                            msg = ws_pb2.WebSocketMessage()
                            msg.ParseFromString(raw)
                            if msg.WhichOneof('message') == 'new_task':
                                task = msg.new_task
                                bg_task = asyncio.create_task(server_process(main_loop, session, websocket, task))
                                bg_tasks.add(bg_task)
                                bg_task.add_done_callback(bg_tasks.discard)

                    except Exception as e:
                        logger.error(f'{e.__class__.__name__}: {e}', exc_info=e if self.verbose else None)

                    finally:
                        logger.info('-- Disconnected from websocket server')
                        for bg_task in bg_tasks:
                            bg_task.cancel()

        def server_thread(future, main_loop, server_loop):
            asyncio.set_event_loop(server_loop)
            try:
                server_loop.run_until_complete(async_server_thread(main_loop))
            finally:
                future.set_result(None)

        future = asyncio.Future()
        Thread(
            target=server_thread,
            args=(future, asyncio.get_running_loop(), self._server_loop),
            daemon=True
        ).start()

        # create a future that is never done
        await future

    async def _run_text_translation(self, ctx: Context):
        coroutine = super()._run_text_translation(ctx)
        if ctx.translator.has_offline():
            return await coroutine
        else:
            task_id = self._task_id
            websocket = self._websocket
            await self.task_lock.release()
            result = await asyncio.wrap_future(
                asyncio.run_coroutine_threadsafe(
                    coroutine,
                    ctx.ws_event_loop
                )
            )
            await self.task_lock.acquire((1<<30) - ctx.ws_count)
            self._task_id = task_id
            self._websocket = websocket
            return result

    async def _run_text_rendering(self, ctx: Context):
        render_mask = (ctx.mask >= 127).astype(np.uint8)[:, :, None]

        output = await super()._run_text_rendering(ctx)
        render_mask[np.sum(ctx.img_rgb != output, axis=2) > 0] = 1
        ctx.render_mask = render_mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_render_in.png'), cv2.cvtColor(ctx.img_rgb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_render_out.png'), cv2.cvtColor(output, cv2.COLOR_RGB2BGR))
            cv2.imwrite(self._result_path('ws_mask.png'), render_mask * 255)

        # only keep sections in mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_inmask.png'), cv2.cvtColor(ctx.img_rgb, cv2.COLOR_RGB2BGRA) * render_mask)
        output = cv2.cvtColor(output, cv2.COLOR_RGB2RGBA) * render_mask
        if self.verbose:
            cv2.imwrite(self._result_path('ws_output.png'), cv2.cvtColor(output, cv2.COLOR_RGBA2BGRA) * render_mask)

        return output


# Experimental. May be replaced by a refactored server/web_main.py in the future.
class MangaTranslatorAPI(MangaTranslator):
    def __init__(self, params: dict = None):
        import nest_asyncio
        nest_asyncio.apply()
        super().__init__(params)
        self.host = params.get('host', '127.0.0.1')
        self.port = params.get('port', '5003')
        self.log_web = params.get('log_web', False)
        self.ignore_errors = params.get('ignore_errors', True)
        self._task_id = None
        self._params = None
        self.params = params

    async def get_file(self, image, base64Images, url) -> Image:
        if image is not None:
            content = image.file.read()
        elif base64Images is not None:
            base64Images = base64Images
            if base64Images.__contains__('base64,'):
                base64Images = base64Images.split('base64,')[1]
            content = base64.b64decode(base64Images)
        elif url is not None:
            from aiohttp import ClientSession
            async with ClientSession() as session:
                async with session.get(url) as resp:
                    if resp.status == 200:
                        content = await resp.read()
                    else:
                        return web.json_response({'status': 'error'})
        else:
            raise ValidationError("donest exist")
        img = Image.open(io.BytesIO(content))

        img.verify()
        img = Image.open(io.BytesIO(content))
        if img.width * img.height > 8000**2:
            raise ValidationError("to large")
        return img

    async def listen(self, translation_params: dict = None):
        self.params = translation_params
        app = web.Application(client_max_size=1024 * 1024 * 50)
        routes = web.RouteTableDef()
        run_until_state = ''

        def hook(state, finished):
            if run_until_state and run_until_state == state and not finished:
                raise TranslationInterrupt()
        self.add_progress_hook(hook)

        @routes.post("/get_text")
        async def text_api(req):
            nonlocal run_until_state
            run_until_state = 'ocr'
            return await self.err_handling(self.texts_exec, req, self.format_translate)

        @routes.post("/translate")
        async def translate_api(req):
            nonlocal run_until_state
            run_until_state = 'translating'
            return await self.err_handling(self.translate_exec, req, self.format_translate)

        @routes.post("/inpaint_translate")
        async def inpaint_translate_api(req):
            nonlocal run_until_state
            run_until_state = 'inpainting'
            return await self.err_handling(self.inpaint_translate_exec, req, self.format_translate)

        # #@routes.post("/file")
        # async def file_api(req):
        #     #TODO: return file
        #     return await self.err_handling(self.file_exec, req, None)

        app.add_routes(routes)
        web.run_app(app, host=self.host, port=self.port)

    async def texts_exec(self, translation_params, img):
        return await self.translate(img, translation_params)

    async def translate_exec(self, translation_params, img):
        return await self.translate(img, translation_params)

    async def inpaint_translate_exec(self, translation_params, img):
        return await self.translate(img, translation_params)

    async def file_exec(self, translation_params, img):
        return await self.translate(img, translation_params)

    async def err_handling(self, func, req, format):
        try:
            if req.content_type == 'application/json' or req.content_type == 'multipart/form-data':
                if req.content_type == 'application/json':
                    d = await req.json()
                else:
                    d = await req.post()
                schema = self.PostSchema()
                data = schema.load(d)
                if data.get('image') is None and data.get('base64Images') is None and data.get('url') is None:
                    return web.json_response({'error': "Missing input", 'status': 422})
                fil = await self.get_file(data.get('image'), data.get('base64Images'), data.get('url'))
                if 'image' in data:
                    del data['image']
                if 'base64Images' in data:
                    del data['base64Images']
                if 'url' in data:
                    del data['url']
                loaded_data = await func(dict(self.params, **data), fil)
                return format(loaded_data)
            else:
                return web.json_response({'error': "Wrong content type: " + req.content_type, 'status': 415},
                                         status=415)
        except ValueError:
            return web.json_response({'error': "Wrong input type", 'status': 422}, status=422)

        except ValidationError as e:
            print(e)
            return web.json_response({'error': "Input invalid", 'status': 422}, status=422)


    def format_translate(self, ctx: Context):
        text_regions = ctx.text_regions
        inpaint = ctx.img_inpainted
        results = []
        for i, blk in enumerate(text_regions):
            minX, minY, maxX, maxY = blk.xyxy
            text = text_regions[i].get_text()
            trans = text_regions[i].translation
            overlay = inpaint[minY:maxY, minX:maxX]
            retval, buffer = cv2.imencode('.jpg', overlay)
            jpg_as_text = base64.b64encode(buffer)
            color1, color2 = text_regions[i].get_font_colors()
            background = jpg_as_text.decode("utf-8")
            results.append({
                'originalText': text,
                'minX': int(minX),
                'minY': int(minY),
                'maxX': int(maxX),
                'maxY': int(maxY),
                'language': langid.classify(text)[0],
                'translatedText': trans,
                'textColor': {
                    'fg': color1.tolist(),
                    'bg': color2.tolist()
                },
                'background': "data:image/jpg;base64," + background
            })
        return web.json_response({'images': [results]})

    class PostSchema(Schema):
        size = fields.Str(required=False, validate=lambda a: a.upper() not in ['S', 'M', 'L', 'X'])
        translator = fields.Str(required=False,
                                validate=lambda a: a.lower() not in TRANSLATORS)
        target_language = fields.Str(required=False,
                                     validate=lambda a: a.upper() not in VALID_LANGUAGES)
        detector = fields.Str(required=False, validate=lambda a: a.lower() not in DETECTORS)
        direction = fields.Str(required=False,
                               validate=lambda a: a.lower() not in set(['auto', 'h', 'v']))
        inpainter = fields.Str(required=False,
                               validate=lambda a: a.lower() not in INPAINTERS)
        ocr = fields.Str(required=False, validate=lambda a: a.lower() not in OCRS)
        upscale_ratio = fields.Integer(required=False)
        text_threshold = fields.Float(required=False)
        box_threshold = fields.Float(required=False)
        unclip_ratio = fields.Float(required=False)
        inpainting_size = fields.Integer(required=False)
        font_size_offset = fields.Integer(required=False)
        text_mag_ratio = fields.Integer(required=False)
        det_rearrange_max_batches = fields.Integer(required=False)
        manga2eng = fields.Boolean(required=False)
        base64Images = fields.Raw(required=False)
        image = fields.Raw(required=False)
        url = fields.Raw(required=False)
        fingerprint = fields.Raw(required=False)
        clientUuid = fields.Raw(required=False)


class MangaTranslatorWeb2(MangaTranslator):
    def __init__(self, params: dict = None):
        super().__init__(params)
        self.translator = None
        self.host = params.get('host', '127.0.0.1')
        self.port = params.get('port', '7860')
        self.share = params.get('share', False)
       
    async def process_zip_file(self, zip_file, translator_params=None, progress=gr.Progress()):
        self.fixBadZipfile(zip_file.name)
        zip_file_name = os.path.splitext(os.path.basename(zip_file.name))[0].strip()
        output_text = ""
        output_files = []
        params_hash = translator_params.get('params_hash', '')
        try:
            with zipfile.ZipFile(zip_file.name, "r") as zf:
                for file_info in progress.tqdm(zf.infolist(), desc="Processing Files", unit="files"):
                    file_extension = os.path.splitext(file_info.filename)[1].lower()[1:]
                    
                    if file_extension in ["jpg", "jpeg", "png", "bmp", "gif", "webp"]:
                        temp_file_name = os.path.join(BASE_PATH, 'result', zip_file_name, file_info.filename.split('/')[-1])
                        os.makedirs(os.path.dirname(temp_file_name), exist_ok=True)
                        with zf.open(file_info.filename) as file:
                            with open(temp_file_name, "wb") as temp_file:
                                temp_file.write(file.read())

                        out_file_name = await self.process_image(temp_file, translator_params)
                        # check if output file path exist
                        if not os.path.exists(out_file_name):
                            # remove "-translated" from filename
                            out_file_name = out_file_name.replace(f"-{params_hash}-translated", "")
                        output_files.append({"name":out_file_name, "data": None})
                    else:
                        raise ValueError("Unsupported file format. Please upload a zip file containing text files in .txt format.")
            
            # Create a new zip file
            # create path for zip file
            # os.makedirs(os.path.join(BASE_PATH, 'result', zip_file_name), exist_ok=True)
            with zipfile.ZipFile(os.path.join(BASE_PATH, 'result', zip_file_name, f'{zip_file_name}-{params_hash}-translated.zip'), 'w') as zip_file:
                # Add the translated text file to the zip file
                for file in progress.tqdm(output_files, desc="Creating Zip File", unit="files"):
                    if file["data"] is not None:
                        zip_file.writestr(file["name"], file["data"])
                    else:
                        arcname = os.path.basename(file["name"])
                        zip_file.write(file["name"], arcname)
            return output_text, os.path.join(BASE_PATH, 'result', zip_file_name, f'{zip_file_name}-{params_hash}-translated.zip')
        
        except BadZipFile:
            raise ValueError("The provided file is not a valid zip file. Please upload a valid zip file containing text files.")
        
        
    async def process_image(self, image_file=None, params={}):
        params = params.copy()
        if image_file:
            params_hash = params.get('params_hash', '')
            dir_name = os.path.split(os.path.split(image_file.name)[0])[1].strip()
            file_name = os.path.splitext(os.path.split(image_file.name)[1])[0] + f"-{params_hash}-translated." + params.get('format', 'jpg')
            dest = os.path.join(BASE_PATH, 'result', dir_name, file_name)
            
            if not os.path.exists(dest):
                os.makedirs(os.path.dirname(dest), exist_ok=True)

            if params.get('device') == "cuda_limited":
                params['use_cuda_limited'] = True
                
            self.device = 'cuda' if params.get('device', False) else 'cpu'
            self._cuda_limited_memory = params.get('use_cuda_limited', False)
            if self._cuda_limited_memory and not self.using_cuda:
                self.device = 'cuda'
            if self.using_cuda and not torch.cuda.is_available():
                raise Exception('CUDA compatible device could not be found whilst --use-cuda args was set...')

            del params['device']
            
            await self.translate_path(path=image_file.name, dest=dest, params=params)
            return dest
        else:
            raise ValueError("Unsupported file format. Please upload an image file.")
        
    def process_image_sync(self, image_file=None, translator="offline", target_lang="ENG",
                           device="cpu", image_detector="default", image_inpainter="default",
                           image_upscaler="esrgan", image_upscale_ratio=0, image_detection_size=2048,
                           image_colorizer="None", misc_attempts=0, misc_skip_errors=False,
                           image_revert_upscaling=False, image_det_rotate=False, image_det_auto_rotate=False,
                           image_det_invert=False, image_det_gamma_correct=False, image_unclip_ratio=2.3,
                           image_box_threshold=0.7, image_text_threshold=0.5, image_inpainting_size=2048,
                           image_colorization_size=576, image_denoise_sigma=30, image_save_quality=85,
                           image_save_file_type="jpg", text_min_text_length=0, text_font_size=None,
                           text_font_size_offset=0, text_font_size_minimum=-1, text_force_render_orientation="auto",
                           text_alignment="auto", text_case="sentence", text_manga2eng=False,
                           text_filter_text=None, text_gimp_font="Sans-serif", text_font_path=None,
                           misc_overwrite=False, image_ignore_bubble=0
                           ):
        
        params = {
            'translator': translator,
            'target_lang': target_lang,
            'device': device,
            'detector': image_detector,
            'inpainter': image_inpainter,
            'upscaler': image_upscaler,
            'upscale_ratio': image_upscale_ratio,
            'detection_size': image_detection_size,
            'colorizer': image_colorizer if image_colorizer != "None" else None,
            'attempts': misc_attempts,
            'skip_errors': misc_skip_errors,
            'revert_upscaling': image_revert_upscaling,
            'det_rotate': image_det_rotate,
            'det_auto_rotate': image_det_auto_rotate,
            'det_invert': image_det_invert,
            'det_gamma_correct': image_det_gamma_correct,
            'unclip_ratio': image_unclip_ratio,
            'box_threshold': image_box_threshold,
            'text_threshold': image_text_threshold,
            'inpainting_size': image_inpainting_size,
            'colorization_size': image_colorization_size,
            'denoise_sigma': image_denoise_sigma,
            'save_quality': image_save_quality,
            'format': image_save_file_type,
            'min_text_length': text_min_text_length,
            'font_size': text_font_size if text_font_size > 0 else None,
            'font_size_offset': text_font_size_offset,
            'font_size_minimum': text_font_size_minimum,
            'manga2eng': text_manga2eng,
            'filter_text': text_filter_text,
            'gimp_font': text_gimp_font,
            'font_path': text_font_path,
            'overwrite': misc_overwrite,
            'ignore_bubble': image_ignore_bubble
        }
        
        if (text_force_render_orientation == 'horizontal'):
            params['force_horizontal'] = True
        elif (text_force_render_orientation == 'vertical'):
            params['force_vertical'] = True
            
        if (text_alignment == 'left'):
            params['align_left'] = True
        elif (text_alignment == 'right'):
            params['align_right'] = True
        elif (text_alignment == 'center'):
            params['align_center'] = True
            
        if (text_case == 'uppercase'):
            params['uppercase'] = True
            print("upper: " + str(params['uppercase']))
        elif (text_case == 'lowercase'):
            params['lowercase'] = True
            
        file_translated = self.file_already_exists(image_file, params)
        if file_translated and not misc_overwrite:
            return file_translated, "File already exists. Skipping translation."
       
        try:
            startTime = time.time()
            result = asyncio.run(self.process_image(image_file, params))
            endTime = time.time()
            totalTime = round(endTime - startTime, 2)
            status = "Successfully translated image.\n" + "Time taken: " + str(totalTime) + " seconds" 
        except Exception as e:
            result = None
            status = "Failed to translate image:\n" + str(e)
        
        return result, status
        
    def process_image_zip(self, image_zip_file=None, translator="offline", target_lang="ENG",
                          device="cpu", image_detector="default", image_inpainter="default",
                          image_upscaler="esrgan", image_upscale_ratio=0, image_detection_size=2048, 
                          image_colorizer="None", misc_attempts=0, misc_skip_errors=False,
                          image_revert_upscaling=False, image_det_rotate=False, image_det_auto_rotate=False,
                          image_det_invert=False, image_det_gamma_correct=False, image_unclip_ratio=2.3,
                          image_box_threshold=0.7, image_text_threshold=0.5, image_inpainting_size=2048,
                          image_colorization_size=576, image_denoise_sigma=30, image_save_quality=85,
                          image_save_file_type="jpg", text_min_text_length=0, text_font_size=None,
                          text_font_size_offset=0, text_font_size_minimum=-1, text_force_render_orientation="auto",
                          text_alignment="auto", text_case="sentence", text_manga2eng=False,
                          text_filter_text=None, text_gimp_font="Sans-serif", text_font_path=None,
                          misc_overwrite=False, image_ignore_bubble=0,
                          progress=gr.Progress()):
        
        if image_zip_file:
            params = {
                'translator': translator,
                'target_lang': target_lang,
                'device': device,
                'detector': image_detector,
                'inpainter': image_inpainter,
                'upscaler': image_upscaler,
                'upscale_ratio': image_upscale_ratio,
                'detection_size': image_detection_size,
                'colorizer': image_colorizer if image_colorizer != "None" else None,
                'attempts': misc_attempts,
                'skip_errors': misc_skip_errors,
                'revert_upscaling': image_revert_upscaling,
                'det_rotate': image_det_rotate,
                'det_auto_rotate': image_det_auto_rotate,
                'det_invert': image_det_invert,
                'det_gamma_correct': image_det_gamma_correct,
                'unclip_ratio': image_unclip_ratio,
                'box_threshold': image_box_threshold,
                'text_threshold': image_text_threshold,
                'inpainting_size': image_inpainting_size,
                'colorization_size': image_colorization_size,
                'denoise_sigma': image_denoise_sigma,
                'save_quality': image_save_quality,
                'format': image_save_file_type,
                'min_text_length': text_min_text_length,
                'font_size': text_font_size if text_font_size > 0 else None,
                'font_size_offset': text_font_size_offset,
                'font_size_minimum': text_font_size_minimum,
                'manga2eng': text_manga2eng,
                'filter_text': text_filter_text,
                'gimp_font': text_gimp_font,
                'font_path': text_font_path,
                'overwrite': misc_overwrite,
                'ignore_bubble': image_ignore_bubble
            }
            
            if (text_force_render_orientation == 'horizontal'):
                params['force_horizontal'] = True
            elif (text_force_render_orientation == 'vertical'):
                params['force_vertical'] = True
                
            if (text_alignment == 'left'):
                params['align_left'] = True
            elif (text_alignment == 'right'):
                params['align_right'] = True
            elif (text_alignment == 'center'):
                params['align_center'] = True
                
            if (text_case == 'upper'):
                params['uppercase'] = True
            elif (text_case == 'lower'):
                params['lowercase'] = True
                
            file_translated = self.zipfile_already_exists(image_zip_file, params)
            if file_translated and not misc_overwrite:
                return file_translated, "File already exists. Skipping translation."
                
            try:
                startTime = time.time()
                task = self.process_zip_file(image_zip_file, translator_params=params, progress=progress)
                text, dest = asyncio.run(task)
                endTime = time.time()
                totalTime = round(endTime - startTime, 2)
                status = "Successfully translated zip file.\n" + "Time taken: " + str(totalTime) + " seconds"
            except Exception as e:
                dest = None
                status = "Failed to translate zip file:\n" + str(e)
                
            return dest, status
        else:
            raise ValueError("Unsupported file format. Please upload an image file.")
        
    def fixBadZipfile(self, zipFile):  
        f = open(zipFile, 'r+b')  
        data = f.read()  
        pos = data.find(b'\x50\x4b\x05\x06') # End of central directory signature  
        if (pos > 0):  
            # print("Trancating file at location " + str(pos + 22)+ ".")  
            f.seek(pos + 22)   # size of 'ZIP end of central directory record' 
            f.truncate()  
            f.close()  
        else:  
            # raise error, file is truncated
            raise ValueError("The provided file is not a valid zip file. Please upload a valid zip file containing text files.")
        
    def zipfile_already_exists(self, file_path, params):
        params_hash = self.generate_md5_signature(params)
        params['params_hash'] = params_hash
        dir_name = os.path.splitext(os.path.basename(file_path.name))[0].strip()
        translated_file = os.path.join(BASE_PATH, 'result', dir_name, f'{dir_name}-{params_hash}-translated.zip')
        if os.path.exists(translated_file):
            return translated_file
        return None
    
    def file_already_exists(self, file_path, params):
        params_hash = self.generate_md5_signature(params)
        params['params_hash'] = params_hash
        dir_name = os.path.splitext(os.path.basename(file_path.name))[0].strip()
        file_name = os.path.splitext(os.path.basename(file_path.name))[0] + f"-{params_hash}-translated." + params.get('format', 'jpg')
        translated_file = os.path.join(BASE_PATH, 'result', dir_name, file_name)
        if os.path.exists(translated_file):
            return translated_file
        return None
    
    def generate_md5_signature(self, params):
        params_copy = params.copy()
        del params_copy['overwrite']
        del params_copy['attempts']
        del params_copy['skip_errors']
        input_string = str(params_copy)
        m = hashlib.md5()
        m.update(input_string.encode('utf-8'))
        return m.hexdigest()[-6:]
        
    async def start(self):
        colorizer_list = ['None']
        colorizer_list.extend(COLORIZERS.keys())
        with gr.Blocks() as interface:
            gr.Markdown("Manga Translation")
            with gr.Tab("Single"):
                with gr.Row():
                    with gr.Column(min_width=600):
                        image_file_input = gr.inputs.File(label="Upload Image File")
                        image_submit_button = gr.Button("Submit")
                    with gr.Column(min_width=600):
                        with gr.Row():
                            image_output_file = gr.Image(label="Download Translated Image File")
                        with gr.Row():
                            image_output_text = gr.Textbox(label="Status", lines=2)
            with gr.Tab("Batch/ZIP"):
                with gr.Row():
                    with gr.Column(min_width=600):
                        image_zip_file_input = gr.inputs.File(type="file", label="Batch Image(Zip)")
                        image_zip_submit_button = gr.Button("Submit")
                    with gr.Column(min_width=600):
                        with gr.Row():
                            image_zip_output_file = gr.outputs.File(label="Download Zip File")
                        with gr.Row():
                            image_zip_output_text = gr.Textbox(label="Status", lines=2)
                        
            with gr.Column():
                gr.Markdown("Translator Settings")
                with gr.Row():
                    translator_translator = gr.inputs.Dropdown(list(TRANSLATORS.keys()), label="Translator", default="offline")
                    translator_target_lang = gr.inputs.Dropdown(list(VALID_LANGUAGES.keys()), label="Target Language", default="ENG")
                    translator_device = gr.inputs.Radio(["cpu", "cuda", "cuda_limited"], label="Device", default="cpu")
                        
            with gr.Column():
                gr.Markdown("Image Settings")
                with gr.Row():
                    image_detector = gr.inputs.Dropdown(list(DETECTORS.keys()), label="Image Detector", default="default")
                    image_inpainter = gr.inputs.Dropdown(list(INPAINTERS.keys()), label="Image Inpainter", default="default")
                    image_upscaler = gr.inputs.Dropdown(list(UPSCALERS.keys()), label="Image Upscaler", default="esrgan")
                    image_upscale_ratio = gr.inputs.Slider(minimum=0, maximum=32, step=1, label="Image Upscale Ratio", default=0)
                    image_detection_size = gr.inputs.Slider(minimum=0, maximum=2560, step=1, label="Image Detection Size", default=2048)
                    image_revert_upscaling = gr.inputs.Checkbox(label="Revert Upscaling", default=False)
                with gr.Row():
                    image_colorizer = gr.inputs.Dropdown(colorizer_list, label="Image Colorizer", default="None", optional=True)
                    image_det_rotate = gr.inputs.Checkbox(label="Rotate", default=False)
                    image_det_auto_rotate = gr.inputs.Checkbox(label="Auto Rotate", default=False)
                    image_det_invert = gr.inputs.Checkbox(label="Invert", default=False)
                    image_det_gamma_correct = gr.inputs.Checkbox(label="Gamma Correct", default=False)
                with gr.Row():
                    image_unclip_ratio = gr.inputs.Slider(minimum=0.1, maximum=20, step=0.01, label="Unclip Ratio", default=2.3)
                    image_box_threshold = gr.inputs.Slider(minimum=0.1, maximum=5, step=0.01, label="Box Threshold", default=0.7)
                    image_text_threshold = gr.inputs.Slider(minimum=0.1, maximum=5, step=0.01, label="Text Threshold", default=0.5)
                    image_inpainting_size = gr.inputs.Slider(minimum=0, maximum=4096, step=1, label="Inpainting Size", default=2048)
                    image_colorization_size = gr.inputs.Slider(minimum=-1, maximum=4096, step=1, label="Colorization Size", default=576)
                    
                with gr.Row():
                    image_ignore_bubble = gr.inputs.Slider(minimum=0, maximum=50, step=1, label="Ignore Bubble", default=0)
                    image_denoise_sigma = gr.inputs.Slider(minimum=0, maximum=100, step=0.1, label="Denoise Sigma", default=30)
                    image_save_quality = gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Save Quality", default=85)
                    image_save_file_type = gr.inputs.Dropdown(["jpg", "png", "webp"], label="Save File Type", default="jpg")
                    
            with gr.Column():
                gr.Markdown("Text Settings")
                with gr.Row():
                    text_min_text_length = gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Min Text Length", default=0)
                    text_font_size = gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Font Size", default=None)
                    text_font_size_offset = gr.inputs.Slider(minimum=0, maximum=100, step=1, label="Font Size Offset", default=0)
                    text_font_size_minimum = gr.inputs.Slider(minimum=-1, maximum=100, step=1, label="Font Size Minimum", default=-1)
                    text_force_render_orientation = gr.inputs.Dropdown(["auto", "horizontal", "vertical"], label="Force Render Text Orientation", default="auto")
                with gr.Row():
                    text_alignment = gr.inputs.Dropdown(["auto", "left", "center", "right"], label="Text Alignment", default="auto")
                    text_case = gr.inputs.Dropdown(["sentence", "uppercase", "lowercase"], label="Text Case", default="sentence")
                    text_manga2eng = gr.inputs.Checkbox(label="Manga2Eng", default=False)
                    text_filter_text = gr.inputs.Textbox(label="Filter Text", default=None)
                    text_gimp_font = gr.inputs.Textbox(label="GIMP Font", default="Sans-serif")
                with gr.Row():
                    text_font_path = gr.inputs.File(label="Font Path", optional=True)
                    
            with gr.Column():
                gr.Markdown("Misc")
                with gr.Row():
                    misc_attempts = gr.inputs.Slider(minimum=0, maximum=10, step=1, label="Attempts", default=0)
                    misc_skip_error = gr.inputs.Checkbox(label="Skip Error", default=False)
                    misc_overwrite = gr.inputs.Checkbox(label="Overwrite", default=False)
                    
            default_params = [
                translator_translator,
                translator_target_lang,
                translator_device,
                image_detector,
                image_inpainter,
                image_upscaler,
                image_upscale_ratio,
                image_detection_size,
                image_colorizer,
                misc_attempts,
                misc_skip_error,
                image_revert_upscaling,
                image_det_rotate,
                image_det_auto_rotate,
                image_det_invert,
                image_det_gamma_correct,
                image_unclip_ratio,
                image_box_threshold,
                image_text_threshold,
                image_inpainting_size,
                image_colorization_size,
                image_denoise_sigma,
                image_save_quality,
                image_save_file_type,
                text_min_text_length,
                text_font_size,
                text_font_size_offset,
                text_font_size_minimum,
                text_force_render_orientation,
                text_alignment,
                text_case,
                text_manga2eng,
                text_filter_text,
                text_gimp_font,
                text_font_path,
                misc_overwrite,
                image_ignore_bubble
            ]
            
            image_submit = [
                image_file_input
            ]
            image_submit.extend(default_params)
                        
            image_zip_submit = [
                image_zip_file_input,
            ]
            image_zip_submit.extend(default_params)
            
        
            image_submit_button.click(self.process_image_sync, inputs=image_submit,
                outputs=[image_output_file, image_output_text])
            image_zip_submit_button.click(self.process_image_zip, inputs=image_zip_submit,
                outputs=[image_zip_output_file, image_zip_output_text])
        
        interface.queue(concurrency_count=1).launch(server_name=self.host, debug=True, share=self.share, server_port=self.port)