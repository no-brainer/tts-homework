import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts_hw.alignment import GraphemeAligner, PrecomputedAligner
from tts_hw.base import BaseTrainer
from tts_hw.utils import inf_loop, MetricTracker, get_mask_from_lengths


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            optimizer,
            config,
            device,
            vocoder,
            aligner,
            featurizer,
            data_loader,
            valid_data_loader=None,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
            sr=16000
    ):
        super().__init__(model, criterion, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 10
        self.sr = sr

        self.vocoder = vocoder
        self.aligner = aligner
        self.featurizer = featurizer

        self.train_metrics = MetricTracker(
            "loss", "duration loss", "mel loss", "grad norm", writer=self.writer
        )
        self.valid_metrics = MetricTracker(
            "loss", "duration loss", "mel loss", writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the GPU
        """
        for tensor_for_gpu in ["melspec", "text_encoded", "durations"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def update_batch(self, batch):
        durations = self.aligner(**batch)

        if isinstance(self.aligner, PrecomputedAligner):
            batch["durations"] = durations
            return batch

        full_duration = durations.sum(axis=1)
        durations /= full_duration[:, None]

        batch["waveform_lengths"] = (batch["waveform_lengths"].double() * full_duration).long()
        batch["waveform"] *= get_mask_from_lengths(batch["waveform_lengths"], batch["waveform"].size(1))

        melspec = self.featurizer(batch["waveform"])
        batch["melspec_lengths"] = (batch["waveform_lengths"] / self.featurizer.hop_length).long()
        batch["melspec"] = melspec.transpose(-1, -2)

        batch["durations"] = durations * batch["melspec_lengths"][:, None]

        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.data_loader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_media(**batch)
                self._log_scalars(self.train_metrics)
                self.train_metrics.reset()

            if batch_idx > self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{"val_" + k: v for k, v in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.update_batch(batch)
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)

        batch["melspec_preds"] = outputs[0]
        batch["melspec_pred_lengths"] = outputs[1]
        batch["log_durations_pred"] = outputs[2]

        loss, loss_parts = self.criterion(**batch)
        batch["loss"] = loss
        batch["duration loss"] = loss_parts["duration_loss"]
        batch["mel loss"] = loss_parts["mel_loss"]

        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        for key in ["loss", "duration loss", "mel loss"]:
            metrics.update(key, batch[key].item())

        return batch

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(self.valid_data_loader),
                    desc="validation",
                    total=len(self.valid_data_loader),
            ):
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.valid_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, "valid")
            self._log_scalars(self.valid_metrics)
            self._log_media(**batch)

        # # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.data_loader, "n_samples"):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_media(self, *args, **kwargs):
        transcripts = kwargs.get("text")[0]
        self.writer.add_text("transcripts", transcripts)

        melspec_preds = kwargs.get("melspec_preds")[0]
        melspec_preds = melspec_preds[:kwargs.get("melspec_pred_lengths")[0].item()]
        self.writer.add_image("predicted spectrograms", melspec_preds)

        melspec_target = kwargs.get("melspec")[0]
        melspec_target = melspec_target[:kwargs.get("melspec_lengths")[0].item()]
        self.writer.add_image("true spectrograms", melspec_target)

        waveform_pred = self.vocoder.inference(melspec_preds.transpose(-2, -1).unsqueeze(0)).squeeze(0)
        self.writer.add_audio("predicted audio", waveform_pred.cpu(), self.sr)

        waveform_true = kwargs.get("waveform")[0]
        waveform_true = waveform_true[:kwargs.get("waveform_lengths")[0].item()]
        self.writer.add_audio("true audio", waveform_true.cpu(), self.sr)

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
