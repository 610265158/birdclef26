# -*-coding:utf-8-*-
import math
import random

import sklearn.metrics
import cv2
import time
import os
import pandas as pd
from torch.utils.data import DataLoader, WeightedRandomSampler

from lib.dataset.dataietr import AlaskaDataIter, build_sample_weights

from train_config import config as cfg
# from lib.dataset.dataietr import DataIter
from timm.utils.model_ema import ModelEmaV3

from lib.core.base_trainer.metric import *
import torch

from lib.core.base_trainer.model import Net

from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from lib.core.model.mix.mix import mixup, mixup_criterion


#
class BCEFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds, targets):
        bce_loss = nn.BCEWithLogitsLoss(reduction='none')(preds, targets)
        probas = torch.sigmoid(preds)
        loss = targets * self.alpha * \
               (1. - probas) ** self.gamma * bce_loss + \
               (1. - targets) * probas ** self.gamma * bce_loss
        loss = loss.mean()
        return loss


class Train(object):
    """Train class.
    """

    def __init__(self,
                 train_df,
                 val_df,
                 fold,
                 nm2cls):

        self.ddp = False

        if self.ddp:
            torch.distributed.init_process_group(backend="nccl")
            self.train_generator = AlaskaDataIter(train_df, nm2cls=nm2cls,
                                                  audio_dir=cfg.DATA.audio_dir,
                                                  taxonomy_file=cfg.DATA.taxonomy_file,
                                                  training_flag=True, shuffle=False)
            self.train_ds = DataLoader(self.train_generator,
                                       cfg.TRAIN.batch_size,
                                       num_workers=cfg.TRAIN.process_num,
                                       sampler=DistributedSampler(self.train_generator,
                                                                  shuffle=True))

            self.val_generator = AlaskaDataIter(val_df, nm2cls=nm2cls,
                                                 audio_dir=cfg.DATA.audio_dir,
                                                 extra_audio_dir=cfg.DATA.get('extra_audio_dir', None),
                                                 taxonomy_file=cfg.DATA.taxonomy_file,
                                                 training_flag=False, shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,
                                     sampler=DistributedSampler(self.val_generator,
                                                                shuffle=False))
            local_rank = torch.distributed.get_rank()
            torch.cuda.set_device(local_rank)
            self.device = torch.device("cuda", local_rank)


        else:
            self.train_generator = AlaskaDataIter(train_df, nm2cls=nm2cls,
                                                  audio_dir=cfg.DATA.audio_dir,
                                                  extra_audio_dir=cfg.DATA.get('extra_audio_dir', None),
                                                  taxonomy_file=cfg.DATA.taxonomy_file,
                                                  training_flag=True, shuffle=False)
            train_sampler = None
            train_shuffle = True
            if cfg.TRAIN.use_balanced_sampler:
                sample_weights = build_sample_weights(
                    train_df,
                    alpha=cfg.TRAIN.sampler_alpha,
                    min_weight=cfg.TRAIN.sampler_min_weight,
                    max_weight=cfg.TRAIN.sampler_max_weight,
                    soundscape_boost=cfg.TRAIN.sampler_soundscape_boost,
                    target_count=cfg.TRAIN.sampler_target_count,
                )
                train_sampler = WeightedRandomSampler(
                    weights=torch.as_tensor(sample_weights, dtype=torch.double),
                    num_samples=len(sample_weights),
                    replacement=True,
                )
                train_shuffle = False
            pse_data_file = cfg.DATA.get('pse_data_file', None)
            main_batch_size = cfg.TRAIN.batch_size

            self.train_ds = DataLoader(self.train_generator,
                                       main_batch_size,
                                       num_workers=cfg.TRAIN.process_num,
                                       shuffle=train_shuffle,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)

            if pse_data_file and os.path.exists(pse_data_file):
                pse_df = pd.read_csv(pse_data_file, low_memory=False).fillna(10086)
                self.pse_generator = AlaskaDataIter(pse_df, nm2cls=nm2cls,
                                                    audio_dir=cfg.DATA.audio_dir,
                                                    taxonomy_file=cfg.DATA.taxonomy_file,
                                                    training_flag=True, shuffle=False)
                pse_sampler = None
                pse_shuffle = True
                if 'sample_weight' in pse_df.columns:
                    pse_weights = pse_df['sample_weight'].astype(float).values
                    pse_weights[pse_weights <= 0] = 0.01
                    pse_sampler = WeightedRandomSampler(
                        weights=torch.as_tensor(pse_weights, dtype=torch.double),
                        num_samples=len(pse_weights),
                        replacement=True,
                    )
                    pse_shuffle = False
                    logger.info('PSE using WeightedRandomSampler, effective samples: %d', int(pse_weights.sum()))
                pse_batch_size = cfg.TRAIN.get('pse_batch_size', cfg.TRAIN.batch_size // 2)
                self.pse_ds = DataLoader(self.pse_generator,
                                         pse_batch_size,
                                         num_workers=cfg.TRAIN.process_num,
                                         shuffle=pse_shuffle,
                                         sampler=pse_sampler,
                                         pin_memory=True,
                                         persistent_workers=True)
                logger.info('PSE dataloader ready: %d clips, batch_size=%d', len(pse_df), pse_batch_size)
            else:
                self.pse_ds = None

            self.val_generator = AlaskaDataIter(val_df, nm2cls=nm2cls,
                                                audio_dir=cfg.DATA.audio_dir,
                                                extra_audio_dir=cfg.DATA.get('extra_audio_dir', None),
                                                taxonomy_file=cfg.DATA.taxonomy_file,
                                                training_flag=False, shuffle=False)

            self.val_ds = DataLoader(self.val_generator,
                                     cfg.TRAIN.validatiojn_batch_size,
                                     num_workers=cfg.TRAIN.process_num,
                                     shuffle=False,
                                     pin_memory=True,
                                     persistent_workers=True)

            self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        self.fold = fold

        self.init_lr = cfg.TRAIN.init_lr
        self.warup_step = cfg.TRAIN.warmup_step
        self.epochs = cfg.TRAIN.epoch
        self.batch_size = cfg.TRAIN.batch_size
        self.l2_regularization = cfg.TRAIN.weight_decay_factor

        self.early_stop = cfg.MODEL.early_stop

        self.accumulation_step = cfg.TRAIN.accumulation_batch_size // cfg.TRAIN.batch_size

        self.gradient_clip = cfg.TRAIN.gradient_clip

        self.save_dir = cfg.MODEL.model_path
        #### make the device

        self.model = Net(num_classes=cfg.DATA.num_classes).to(self.device)
        self.load_weight()

        if cfg.TRAIN.get('compile', False):
            self.model = torch.compile(self.model, mode='default')
            logger.info('torch.compile enabled')

        if 'Adamw' in cfg.TRAIN.opt:

            self.optimizer = self.configure_optimizers()
        else:
            raise NotImplementedError

        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model,
                                                                   device_ids=[local_rank],
                                                                   output_device=local_rank,
                                                                   find_unused_parameters=True)
        elif torch.cuda.device_count() > 1:
            self.model = nn.DataParallel(self.model)


        ###control vars
        self.iter_num = 0

        # unified warmup + cosine schedule at step level
        # total steps = epochs * steps_per_epoch; estimated at init with dataset size
        steps_per_epoch = max(1, len(self.train_generator) // cfg.TRAIN.batch_size)
        total_steps = self.epochs * steps_per_epoch
        warmup_steps = self.warup_step

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                # linear warmup from 0 to 1
                return float(current_step) / float(max(1, warmup_steps))
            # cosine annealing from 1 to eta_min/init_lr after warmup
            progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            eta_min_ratio = 1.e-6 / self.init_lr
            return eta_min_ratio + (1.0 - eta_min_ratio) * cosine_decay

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.criterion = nn.BCEWithLogitsLoss()

        self.scaler = torch.cuda.amp.GradScaler()

    def configure_optimizers(self, ):
        optimizer = torch.optim.AdamW(self.model.named_parameters(), self.init_lr)
        return optimizer

    def custom_loop(self):
        """Custom training and testing loop.
        Args:
          train_dist_dataset: Training dataset created using strategy.
          test_dist_dataset: Testing dataset created using strategy.
          strategy: Distribution strategy.
        Returns:
          train_loss, train_accuracy, test_loss, test_accuracy
        """

        def distributed_train_epoch(epoch_num):

            summary_loss = AverageMeter()

            self.model.train()

            pse_iter = iter(self.pse_ds) if self.pse_ds is not None else None

            for batch in self.train_ds:

                waves, label = batch

                main_size = waves.size(0)
                if pse_iter is not None:
                    try:
                        pse_batch = next(pse_iter)
                    except StopIteration:
                        pse_iter = iter(self.pse_ds)
                        pse_batch = next(pse_iter)
                    pse_waves, pse_label = pse_batch[0], pse_batch[1]
                    waves = torch.cat([waves, pse_waves], dim=0)
                    label = torch.cat([label, pse_label], dim=0)

                start = time.time()

                data = waves.to(self.device).float()
                label = label.to(self.device).float()
                batch_size = data.shape[0]

                if random.uniform(0, 1) < 1:
                    data, label, mix_indices = mixup(data, label, 2,
                                                     main_size=main_size if self.pse_ds is not None else None)
                    with torch.cuda.amp.autocast(enabled=cfg.TRAIN.mix_precision,dtype=torch.bfloat16):
                        prediction, clip_wise_pre = self.model(data)
                    if prediction.dim() == 3:
                        prediction = prediction.reshape(-1, prediction.size(-1))
                        clip_wise_pre = clip_wise_pre.reshape(-1, clip_wise_pre.size(-1))
                        label[0] = label[0].reshape(-1, label[0].size(-1))
                        label[1] = label[1].reshape(-1, label[1].size(-1))

                    current_loss = (mixup_criterion(prediction, label, self.criterion) +
                                    mixup_criterion(clip_wise_pre, label, self.criterion)) / 2.
                else:
                    with torch.cuda.amp.autocast(enabled=cfg.TRAIN.mix_precision):
                        predictions, _ = self.model(data)
                    if predictions.dim() == 3:
                        predictions = predictions.reshape(-1, predictions.size(-1))
                        label = label.reshape(-1, label.size(-1))

                    current_loss = self.criterion(predictions, label)

                summary_loss.update(current_loss.detach().item(), batch_size)

                self.scaler.scale(current_loss).backward()

                if ((self.iter_num + 1) % self.accumulation_step) == 0:
                    self.scaler.unscale_(self.optimizer)
                    if self.gradient_clip > 0:
                        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip, norm_type=2)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    self.scheduler.step()

                self.iter_num += 1
                time_cost_per_batch = time.time() - start

                images_per_sec = cfg.TRAIN.batch_size / time_cost_per_batch

                if self.iter_num % cfg.TRAIN.log_interval == 0:
                    log_message = '[fold %d], ' \
                                  'Train Step %d, ' \
                                  'summary_loss: %.6f, ' \
                                  'time: %.6f, ' \
                                  'speed %d images/persec' % (
                                      self.fold,
                                      self.iter_num,
                                      summary_loss.avg,
                                      time.time() - start,
                                      images_per_sec)
                    logger.info(log_message)

            return summary_loss

        def distributed_test_epoch(loader):

            summary_loss = AverageMeter()
            summary_rocauc = ROCAUCMeter()
            self.model.eval()

            oof_pre = []
            oof_gt = []
            with torch.no_grad():
                for batch in tqdm(loader):
                    waves, labels = batch

                    data = waves.to(self.device).float()
                    labels = labels.to(self.device).float()
                    batch_size = data.shape[0]

                    prediction, _ = self.model(data)

                    if prediction.dim() == 3:
                        prediction = prediction.reshape(-1, prediction.size(-1))
                        labels = labels.reshape(-1, labels.size(-1))

                    current_loss = self.criterion(prediction, labels)
                    summary_rocauc.update(labels, prediction)
                    oof_pre.append(prediction.detach().cpu().numpy())
                    oof_gt.append(labels.detach().cpu().numpy())

                    summary_loss.update(current_loss.detach().item(), batch_size)

            return summary_loss, summary_rocauc, oof_pre, oof_gt

        best_distance = 0
        not_improvement = 0
        for epoch in range(self.epochs):

            for param_group in self.optimizer.param_groups:
                lr = param_group['lr']
            logger.info('learning rate: [%f]' % (lr))
            t = time.time()

            summary_loss = distributed_train_epoch(epoch)
            train_epoch_log_message = '[fold %d], ' \
                                      '[RESULT]: TRAIN. Epoch: %d,' \
                                      ' summary_loss: %.5f,' \
                                      ' time:%.5f' % (
                                          self.fold,
                                          epoch,
                                          summary_loss.avg,
                                          (time.time() - t))
            logger.info(train_epoch_log_message)

            if epoch % cfg.TRAIN.test_interval == 0:
                summary_loss, summary_rocauc, oof_pre, oof_gt = distributed_test_epoch(self.val_ds)

                val_epoch_log_message = '[fold %d], ' \
                                        '[RESULT]: VAL. Epoch: %d,' \
                                        ' val_loss: %.5f,' \
                                        ' val_rocauc: %.5f,' \
                                        ' time:%.5f' % (
                                            self.fold,
                                            epoch,
                                            summary_loss.avg,
                                            summary_rocauc.avg,
                                            (time.time() - t))
                logger.info(val_epoch_log_message)

            #### save model
            if not os.access(cfg.MODEL.model_path, os.F_OK):
                os.mkdir(cfg.MODEL.model_path)
            ###save the best auc model

            #### save the model every end of epoch
            current_model_saved_name = self.save_dir + '/fold%d_epoch_%d_val_loss_%.6f_val_auc%.6f.pth' % (self.fold,
                                                                                                           epoch,
                                                                                                           summary_loss.avg,
                                                                                                           summary_rocauc.avg
                                                                                                           )

            logger.info('A model saved to %s' % current_model_saved_name)
            raw_sd = getattr(self.model, 'module', self.model).state_dict()
            clean_sd = {k.replace('_orig_mod.', ''): v for k, v in raw_sd.items()}
            torch.save(clean_sd, current_model_saved_name)

            # save_checkpoint({
            #           'state_dict': self.model.state_dict(),
            #           },iters=epoch,tag=current_model_saved_name)

            if summary_rocauc.avg > best_distance:
                best_distance = summary_rocauc.avg
                logger.info(' best metric score update as %.6f' % (best_distance))
                logger.info(' bestmodel update as %s' % (current_model_saved_name))
                not_improvement = 0
                self.oof_pre = np.concatenate(oof_pre, axis=0)
                self.oof_gt = np.concatenate(oof_gt, axis=0)


            else:
                not_improvement += 1

            if not_improvement >= self.early_stop and self.early_stop > -1:
                logger.info(' best metric score not improvement for %d, break' % (self.early_stop))
                break

            torch.cuda.empty_cache()

    def load_weight(self):
        if cfg.MODEL.pretrained_model is not None:
            state_dict = torch.load(cfg.MODEL.pretrained_model, map_location=self.device)
            self.model.load_state_dict(state_dict, strict=False)
