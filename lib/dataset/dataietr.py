import os
import re

import torch

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import ast
import os.path
import random

import numpy as np
import pandas as pd
import soundfile as sf

import audiomentations as A
from lib.utils.logger import logger
from train_config import config as cfg

_SOUNDSCAPE_TIME_RE = re.compile(r'_(\d+)_(\d+)\.ogg$')


def get_primary_sampling_label(label_value):
    label_str = str(label_value).strip()
    if not label_str or label_str == '10086':
        return ''
    return label_str.split(';')[0].strip()


def build_sample_weights(df, alpha=0.4, min_weight=1.0, max_weight=4.0,
                         soundscape_boost=1.0, target_count=40,
                         taxonomy_file=None, class_boost=None,
                         max_per_label=None):
    """Build per-sample weights for WeightedRandomSampler.

    Args:
        max_per_label: If set, labels with more than this many samples get
            down-weighted so each label effectively contributes at most
            max_per_label equivalent samples (weight = max_per_label / count).
            Set to None or 0 to disable.
    """
    sample_labels = df['primary_label'].astype(str).map(get_primary_sampling_label)
    label_counts = sample_labels.value_counts()
    valid_counts = label_counts[label_counts > 0]
    if valid_counts.empty:
        return np.ones(len(df), dtype=np.float32)

    class_dict = None
    if class_boost and taxonomy_file and os.path.exists(taxonomy_file):
        taxonomy = pd.read_csv(taxonomy_file)
        class_to_label = {
            'Insecta': 0,
            'Amphibia': 1,
            'Mammalia': 2,
            'Aves': 3,
            'Reptilia': 4,
        }
        taxonomy['class_label'] = taxonomy['class_name'].apply(
            lambda name: class_to_label[name])
        class_dict = dict(zip(
            taxonomy['primary_label'].astype(str),
            taxonomy['class_label'].astype(int)))

    weights = []
    for _, row in df.iterrows():
        label_key = get_primary_sampling_label(row['primary_label'])
        count = max(int(label_counts.get(label_key, 1)), 1)
        is_downsampled = (max_per_label and count > max_per_label)

        if count < target_count:
            weight = float(target_count) / float(count)
        elif is_downsampled:
            weight = float(max_per_label) / float(count)
        else:
            weight = min_weight

        # For downsampled labels, allow weight below min_weight.
        # Otherwise clip to [min_weight, max_weight].
        if is_downsampled:
            weight = min(weight, max_weight)
        else:
            weight = float(np.clip(weight, min_weight, max_weight))

        if 'start' in row.index:
            weight *= soundscape_boost
        if class_dict is not None:
            label_str = str(row['primary_label']).strip()
            cls = class_dict.get(label_str)
            if cls is not None and cls in class_boost:
                weight *= class_boost[cls]
        weights.append(weight)
    return np.asarray(weights, dtype=np.float32)

class AlaskaDataIter():
    def __init__(self, df, nm2cls,
                 audio_dir='../bd26/train_audio',
                 extra_audio_dir=None,
                 taxonomy_file='../bd26/taxonomy.csv',
                 training_flag=True, 
                 shuffle=True, 
                 use_wave=True,
                 zero_fallback=False):

        self.use_wave = use_wave
        self.training_flag = training_flag
        self.shuffle = shuffle
        self.extra_audio_dir = extra_audio_dir
        self.zero_fallback = zero_fallback

        self.raw_data_set_size = None

        self.df = df.copy()
        self.df = self.df.fillna(10086)

        self.classes = self.df['primary_label'].unique()

        self.random_gain = A.Gain(min_gain_db=-4, max_gain_db=4, p=0.8)

        self.nm2cls = nm2cls
        self.num_classes = len(nm2cls)
        self.audio_dir = audio_dir

        taxonomy = pd.read_csv(taxonomy_file)
        class_to_label = {
            'Insecta': 0,
            'Amphibia': 1,
            'Mammalia': 2,
            'Aves': 3,
            'Reptilia': 4,
        }
        taxonomy['class_label'] = taxonomy['class_name'].apply(lambda name: class_to_label[name])
        self.class_dict = dict(zip(taxonomy['primary_label'].astype(str), taxonomy['class_label'].astype(int)))

        self._build_soundscape_lookup()

    def _build_soundscape_lookup(self):
        self.sc_lookup = {}
        for idx, row in self.df.iterrows():
            fn = str(row['filename'])
            if 'soundscapes/' not in fn:
                continue
            m = _SOUNDSCAPE_TIME_RE.search(fn)
            if not m:
                continue
            start_sec = int(m.group(1))
            base = fn[:m.start()]
            if base not in self.sc_lookup:
                self.sc_lookup[base] = {}
            self.sc_lookup[base][start_sec] = idx

    def __getitem__(self, item):

        return self.single_map_func(self.df.iloc[item], self.training_flag)

    def __len__(self):

        return len(self.df)

    def read_random_segment(self, audio_path,
                            valid_length=5 * 32000,
                            random_read=True):
        with sf.SoundFile(audio_path) as sound_file:
            L = sound_file.frames

            if L > valid_length:
                if random_read:
                    start_p = random.randint(0, L - valid_length)
                else:
                    start_p = 0
                frames = valid_length
            else:
                start_p = 0
                frames = L
            end_p = start_p + frames - 1

            frames = sound_file._prepare_read(start_p, stop=None, frames=frames)
            data = sound_file.read(frames, dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1, dtype=np.float32)
        return data, sound_file.samplerate

    def read_multicontext_segment(self, audio_path, num_context=4,
                                  clip_length=5 * 32000, random_read=True):
        total_length = num_context * clip_length
        with sf.SoundFile(audio_path) as sound_file:
            L = sound_file.frames
            if L >= total_length:
                if random_read:
                    start_p = random.randint(0, L - total_length)
                else:
                    start_p = 0
                frames = sound_file._prepare_read(start_p, stop=None, frames=total_length)
                data = sound_file.read(frames, dtype='float32')
            else:
                start_p = 0
                data = sound_file.read(dtype='float32')
            if data.ndim > 1:
                data = data.mean(axis=1, dtype=np.float32)
        return data, sound_file.samplerate, start_p

    def safe_pad(self, waves, valid_lenth=32000 * 5, repeat=False):
        L = waves.shape[0]

        if L < valid_lenth:
            if repeat and L > 0:
                reps = valid_lenth // L + 1
                waves = np.tile(waves, reps)[:valid_lenth].astype(np.float32)
                return waves
            padded_array = np.zeros(valid_lenth, dtype=np.float32)
            padded_array[:L] = waves
            return padded_array
        else:
            return waves[:valid_lenth].astype(np.float32)

    def parse_secondary_labels(self, value):
        if value in (None, '', 10086, '10086'):
            return []
        if isinstance(value, list):
            return [str(x) for x in value if str(x)]
        if isinstance(value, str):
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                logger.warning('Failed to parse secondary_labels: %s', value)
                return []
            if isinstance(parsed, list):
                return [str(x) for x in parsed if str(x)]
        return []

    def _make_label(self, row):
        slp = row.get('soft_label_path', None)
        if slp and isinstance(slp, str) and slp not in ('10086',) and os.path.exists(slp):
            return np.load(slp).astype(np.float32)
        label_container = np.zeros(self.num_classes, dtype=np.float32)
        label_str = get_primary_sampling_label(row['primary_label'])
        if label_str in self.nm2cls:
            all_labels = [label_str] + self.parse_secondary_labels(row.get('secondary_labels', []))
            all_labels = [str(x) for x in all_labels if x != '']
            for item in all_labels:
                if item in self.nm2cls:
                    label_container[self.nm2cls[item]] = 1
        return label_container

    def _is_soundscape(self, filename):
        return 'soundscapes/' in str(filename)

    def _get_soundscape_context(self, dp, num_context):
        fn = str(dp['filename'])
        m = _SOUNDSCAPE_TIME_RE.search(fn)
        if not m:
            return None, None
        start_sec = int(m.group(1))
        base = fn[:m.start()]
        time_map = self.sc_lookup.get(base, {})

        sc_path = dp.get('soundscape_path', None)
        if isinstance(sc_path, str) and sc_path in ('10086',):
            sc_path = None

        clip_length = 32000 * 5
        clips = []
        labels = []
        context_found = []
        for c in range(num_context):
            t = start_sec + c * 5
            found = False
            if t in time_map:
                row_idx = time_map[t]
                row = self.df.iloc[row_idx]
                ogg = os.path.join(self.audio_dir, str(row['filename']))
                if os.path.exists(ogg):
                    data, sr = sf.read(ogg, dtype='float32')
                    if data.ndim > 1:
                        data = data.mean(axis=1, dtype=np.float32)
                    data = self.safe_pad(data, clip_length)
                    clips.append(data)
                    labels.append(self._make_label(row))
                    found = True
                    context_found.append(found)
                    continue
                row_sc = row.get('soundscape_path', sc_path)
                if isinstance(row_sc, str) and row_sc not in ('10086',) and os.path.exists(row_sc):
                    with sf.SoundFile(row_sc) as f:
                        f.seek(t * 32000)
                        data = f.read(clip_length, dtype='float32')
                    if data.ndim > 1:
                        data = data.mean(axis=1, dtype=np.float32)
                    data = self.safe_pad(data, clip_length)
                    clips.append(data)
                    labels.append(self._make_label(row))
                    found = True
                    context_found.append(found)
                    continue
            clips.append(np.zeros(clip_length, dtype=np.float32))
            labels.append(np.zeros(self.num_classes, dtype=np.float32))
            context_found.append(found)

        waves = np.concatenate(clips)
        mc_labels = np.stack(labels)
        return waves, mc_labels

    def single_map_func(self, dp, is_training):

        ogg_path = os.path.join(self.audio_dir, dp['filename'])
        clip_length = 32000 * 5
        num_context = cfg.MODEL.get('num_context', 1)

        if num_context > 1 and self._is_soundscape(dp['filename']):
            waves, label_container = self._get_soundscape_context(dp, num_context)
            if waves is not None:
                data = waves.astype(np.float32, copy=False)
                if is_training:
                    data = self.random_gain(data, sample_rate=32000)
                    data = data.astype(np.float32, copy=False)
                return data, label_container

        if not os.path.exists(ogg_path):
            if self.extra_audio_dir:
                ogg_path = os.path.join(self.extra_audio_dir, dp['filename'])
            if not os.path.exists(ogg_path):
                sc_path = dp.get('soundscape_path', None)
                start_col = dp.get('start', None)
                if (sc_path and isinstance(sc_path, str) and sc_path not in ('10086',)
                        and os.path.exists(sc_path)
                        and start_col is not None and str(start_col) not in ('10086', 'nan')):
                    start_sample = int(float(start_col)) * 32000
                    valid_length = num_context * clip_length if num_context > 1 else clip_length
                    with sf.SoundFile(sc_path) as f:
                        f.seek(start_sample)
                        data = f.read(valid_length, dtype='float32')
                    if data.ndim > 1:
                        data = data.mean(axis=1, dtype=np.float32)
                    data = self.safe_pad(data, valid_length, repeat=(num_context > 1))
                    label_container = self._make_label(dp)
                    if num_context > 1:
                        label_container = np.tile(label_container, (num_context, 1))
                    if is_training:
                        data = self.random_gain(data, sample_rate=32000)
                        data = data.astype(np.float32, copy=False)
                    return data, label_container
                valid_length = num_context * clip_length if num_context > 1 else clip_length
                data = np.zeros(valid_length, dtype=np.float32)
                if num_context > 1:
                    label_container = np.zeros((num_context, self.num_classes), dtype=np.float32)
                else:
                    label_container = np.zeros(self.num_classes, dtype=np.float32)
                return data, label_container

        if num_context > 1:
            valid_length = num_context * clip_length
            waves, sr, start_sample = self.read_multicontext_segment(
                ogg_path, num_context=num_context,
                clip_length=clip_length, random_read=is_training)
            data = self.safe_pad(waves, valid_length, repeat=True)
        else:
            valid_length = clip_length
            start_sample = 0
            if is_training:
                waves, sr = self.read_random_segment(ogg_path, random_read=True)
            else:
                waves, sr = self.read_random_segment(ogg_path, random_read=False)
            data = self.safe_pad(waves, valid_length)

        data = data.astype(np.float32, copy=False)

        label_container = self._make_label(dp)

        if num_context > 1:
            label_container = np.tile(label_container, (num_context, 1))

        if is_training:
            data = self.random_gain(data, sample_rate=sr)
            data = data.astype(np.float32, copy=False)

        return data, label_container
