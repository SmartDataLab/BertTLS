import os

import numpy as np
import torch
from tensorboardX import SummaryWriter

import distributed

# import onmt
from models.reporter import ReportMgr
from models.stats import Statistics
from others.logging import logger
from others.utils import test_rouge, rouge_results_to_str, cal_rouge_tls, cal_date_f1


def get_clss_split(clss, msk_cls, start, end):
    selector = (clss >= start) * (clss < end)
    return (clss - start)[selector].unsqueeze(0), msk_cls[selector].unsqueeze(0)


def _tally_parameters(model):
    n_params = sum([p.nelement() for p in model.parameters()])
    return n_params


def build_trainer(args, device_id, model, optim):
    """
    Simplify `Trainer` creation based on user `opt`s*
    Args:
        opt (:obj:`Namespace`): user options (usually from argument parsing)
        model (:obj:`onmt.models.NMTModel`): the model to train
        fields (dict): dict of fields
        optim (:obj:`onmt.utils.Optimizer`): optimizer used during training
        data_type (str): string describing the type of data
            e.g. "text", "img", "audio"
        model_saver(:obj:`onmt.models.ModelSaverBase`): the utility object
            used to save the model
    """
    device = "cpu" if args.visible_gpus == "-1" else "cuda"

    grad_accum_count = args.accum_count
    n_gpu = args.world_size

    if device_id >= 0:
        gpu_rank = int(args.gpu_ranks[device_id])
    else:
        gpu_rank = 0
        n_gpu = 0

    print("gpu_rank %d" % gpu_rank)

    tensorboard_log_dir = args.model_path

    writer = SummaryWriter(tensorboard_log_dir, comment="Unmt")

    report_manager = ReportMgr(
        args.report_every, start_time=-1, tensorboard_writer=writer
    )

    trainer = Trainer(
        args, model, optim, grad_accum_count, n_gpu, gpu_rank, report_manager
    )

    # print(tr)
    if model:
        n_params = _tally_parameters(model)
        logger.info("* number of parameters: %d" % n_params)

    return trainer


class Trainer(object):
    """
    Class that controls the training process.

    Args:
            model(:py:class:`onmt.models.model.NMTModel`): translation model
                to train
            train_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            valid_loss(:obj:`onmt.utils.loss.LossComputeBase`):
               training loss computation
            optim(:obj:`onmt.utils.optimizers.Optimizer`):
               the optimizer responsible for update
            trunc_size(int): length of truncated back propagation through time
            shard_size(int): compute loss in shards of this size for efficiency
            data_type(string): type of the source input: [text|img|audio]
            norm_method(string): normalization methods: [sents|tokens]
            grad_accum_count(int): accumulate gradients this many times.
            report_manager(:obj:`onmt.utils.ReportMgrBase`):
                the object that creates reports, or None
            model_saver(:obj:`onmt.models.ModelSaverBase`): the saver is
                used to save a checkpoint.
                Thus nothing will be saved if this parameter is None
    """

    def __init__(
        self,
        args,
        model,
        optim,
        grad_accum_count=1,
        n_gpu=1,
        gpu_rank=1,
        report_manager=None,
    ):
        # Basic attributes.
        self.args = args
        self.save_checkpoint_steps = args.save_checkpoint_steps
        self.model = model
        self.optim = optim
        self.grad_accum_count = grad_accum_count
        self.n_gpu = n_gpu
        self.gpu_rank = gpu_rank
        self.report_manager = report_manager

        self.loss = torch.nn.BCELoss(reduction="none")
        assert grad_accum_count > 0
        # Set model in training mode.
        if model:
            self.model.train()
        print("log_id", self.args.db_logger.log_id)

    def train(self, train_iter_fct, train_steps, valid_iter_fct=None, valid_steps=-1):
        """
        The main training loops.
        by iterating over training data (i.e. `train_iter_fct`)
        and running validation (i.e. iterating over `valid_iter_fct`

        Args:
            train_iter_fct(function): a function that returns the train
                iterator. e.g. something like
                train_iter_fct = lambda: generator(*args, **kwargs)
            valid_iter_fct(function): same as train_iter_fct, for valid data
            train_steps(int):
            valid_steps(int):
            save_checkpoint_steps(int):

        Return:
            None
        """
        logger.info("Start training...")

        # step =  self.optim._step + 1
        step = self.optim._step + 1
        true_batchs = []
        accum = 0
        normalization = 0
        train_iter = train_iter_fct()

        total_stats = Statistics()
        report_stats = Statistics()
        self._start_report_manager(start_time=total_stats.start_time)

        while step <= train_steps:

            reduce_counter = 0
            for i, batch in enumerate(train_iter):
                if self.n_gpu == 0 or (i % self.n_gpu == self.gpu_rank):

                    true_batchs.append(batch)
                    normalization += batch.batch_size
                    accum += 1
                    if accum == self.grad_accum_count:
                        reduce_counter += 1
                        if self.n_gpu > 1:
                            normalization = sum(
                                distributed.all_gather_list(normalization)
                            )

                        self._gradient_accumulation(
                            true_batchs, normalization, total_stats, report_stats
                        )

                        report_stats = self._maybe_report_training(
                            step, train_steps, self.optim.learning_rate, report_stats
                        )

                        true_batchs = []
                        accum = 0
                        normalization = 0
                        if (
                            step % self.save_checkpoint_steps == 0
                            and self.gpu_rank == 0
                        ):
                            self._save(step)

                        step += 1
                        if step > train_steps:
                            break

            self.args.db_logger.add_attr("step_%s" % step, report_stats.xent(), "train")
            train_iter = train_iter_fct()

        self.args.db_logger.insert_into_db("train")
        return total_stats

    def validate(self, valid_iter, step=0):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        # self.model.eval()
        stats = Statistics()

        with torch.no_grad():
            for batch in valid_iter:

                src = batch.src
                labels = batch.labels
                segs = batch.segs
                clss = batch.clss
                mask = batch.mask
                mask_cls = batch.mask_cls

                sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

                loss = self.loss(sent_scores, labels.float())
                loss = (loss * mask.float()).sum()
                batch_stats = Statistics(float(loss.cpu().data.numpy()), len(labels))
                stats.update(batch_stats)
            self._report_step(0, step, valid_stats=stats)
            return stats

    def test(self, test_iter, step, cal_lead=False, cal_oracle=False, is_tls=True):
        """Validate model.
            valid_iter: validate data iterator
        Returns:
            :obj:`nmt.Statistics`: validation loss statistics
        """
        # Set model in validating mode.
        def _get_ngrams(n, text):
            ngram_set = set()
            text_length = len(text)
            max_index_ngram_start = text_length - n
            for i in range(max_index_ngram_start + 1):
                ngram_set.add(tuple(text[i : i + n]))
            return ngram_set

        def _block_tri(c, p):
            tri_c = _get_ngrams(3, c.split())
            for s in p:
                tri_s = _get_ngrams(3, s.split())
                if len(tri_c.intersection(tri_s)) > 0:
                    return True
            return False

        # if not cal_lead and not cal_oracle:
        #     self.model.eval()
        stats = Statistics()

        if is_tls:
            tls_rouge_list = []
        can_path = "%s_step%d.candidate" % (self.args.result_path, step)
        gold_path = "%s_step%d.gold" % (self.args.result_path, step)
        with open(can_path, "w") as save_pred:
            with open(gold_path, "w") as save_gold:
                with torch.no_grad():
                    for batch in test_iter:
                        src = batch.src
                        labels = batch.labels
                        segs = batch.segs
                        clss = batch.clss
                        mask = batch.mask
                        mask_cls = batch.mask_cls
                        if is_tls:
                            src_date = batch.src_date
                            tgt_date = batch.tgt_date

                        gold = []
                        pred = []

                        if cal_lead:
                            selected_ids = [
                                list(range(batch.clss.size(1)))
                            ] * batch.batch_size
                        elif cal_oracle:
                            selected_ids = [
                                [
                                    j
                                    for j in range(batch.clss.size(1))
                                    if labels[i][j] == 1
                                ]
                                for i in range(batch.batch_size)
                            ]
                        else:
                            sent_scores_list = []
                            mask_loss_list = []
                            for i in range(segs.size()[1] // 512):
                                start, end = i * 512, (i * 512 + 512)
                                clss_, mask_cls_ = get_clss_split(
                                    clss, mask_cls, start, end
                                )
                                tmp_score, mask_loss_tmp = self.model(
                                    src[:, start:end, ...],
                                    segs[:, start:end, ...],
                                    clss_,
                                    mask[:, start:end, ...],
                                    mask_cls_,
                                )
                                print(i, tmp_score)
                                sent_scores_list.append(tmp_score)
                                mask_loss_list.append(mask_loss_tmp)
                            if i == 0:
                                sent_scores = sent_scores_list[0]
                                mask_loss = mask_loss_list[0]
                            else:
                                sent_scores = torch.cat(sent_scores_list, dim=1)
                                mask_loss = torch.cat(mask_loss_list, dim=1)
                            # print(i, sent_scores, labels)
                            loss = self.loss(sent_scores, labels.float())
                            loss = (loss * mask_loss.float()).sum()
                            batch_stats = Statistics(
                                float(loss.cpu().data.numpy()), len(labels)
                            )
                            stats.update(batch_stats)

                            sent_scores = sent_scores + mask_loss.float()
                            sent_scores = sent_scores.cpu().data.numpy()
                            selected_ids = np.argsort(-sent_scores, 1)
                        # selected_ids = np.sort(selected_ids,1)
                        for i, idx in enumerate(selected_ids):
                            _pred = []
                            if is_tls:
                                _date = []
                            if len(batch.src_str[i]) == 0:
                                continue
                            for j in selected_ids[i][: len(batch.src_str[i])]:
                                if j >= len(batch.src_str[i]):
                                    continue
                                candidate = batch.src_str[i][j].strip()
                                if is_tls:
                                    can_date = src_date[i][j]
                                if self.args.block_trigram:
                                    if not _block_tri(candidate, _pred):
                                        _pred.append(candidate)
                                        if is_tls:
                                            _date.append(can_date)
                                else:
                                    _pred.append(candidate)
                                    if is_tls:
                                        _date.append(can_date)

                                if (
                                    (not cal_oracle)
                                    and (not self.args.recall_eval)
                                    and len(_pred) == 3
                                ):
                                    break

                            if is_tls:
                                rouge1, rouge2 = cal_rouge_tls(
                                    _pred,
                                    _date,
                                    batch.tgt_str[i].split("<q>"),
                                    tgt_date[i],
                                )
                                date_f1 = cal_date_f1(_date, tgt_date[i])["f1"]
                                tls_rouge_list.append((rouge1, rouge2, date_f1))
                            else:
                                _pred = "<q>".join(_pred)
                                if self.args.recall_eval:
                                    _pred = " ".join(
                                        _pred.split()[: len(batch.tgt_str[i].split())]
                                    )

                                pred.append(_pred)
                                gold.append(batch.tgt_str[i])
                        if not is_tls:
                            for i in range(len(gold)):
                                save_gold.write(gold[i].strip() + "\n")
                            for i in range(len(pred)):
                                save_pred.write(pred[i].strip() + "\n")
        if step != -1 and self.args.report_rouge:
            if is_tls:
                rouge1_list = [x[0] for x in tls_rouge_list]
                rouge2_list = [x[1] for x in tls_rouge_list]
                date_f1_list = [x[2] for x in tls_rouge_list]
                logger.info(
                    "Rouges at step %d \n rouge1 : %s, rouge2: %s, date_f1: %s"
                    % (
                        step,
                        sum(rouge1_list) / len(tls_rouge_list),
                        sum(rouge2_list) / len(tls_rouge_list),
                        sum(date_f1_list) / len(date_f1_list),
                    )
                )
                self.args.db_logger.add_attr("rouge1", rouge1_list, "test")
                self.args.db_logger.add_attr("rouge2", rouge2_list, "test")
                self.args.db_logger.add_attr("date_f1", date_f1_list, "test")
                self.args.db_logger.insert_into_db("test")
            else:
                rouges = test_rouge(self.args.temp_dir, can_path, gold_path)
                logger.info(
                    "Rouges at step %d \n%s" % (step, rouge_results_to_str(rouges))
                )
        self._report_step(0, step, valid_stats=stats)

        return stats

    def _gradient_accumulation(
        self, true_batchs, normalization, total_stats, report_stats
    ):
        if self.grad_accum_count > 1:
            self.model.zero_grad()

        for batch in true_batchs:
            if self.grad_accum_count == 1:
                self.model.zero_grad()

            src = batch.src
            labels = batch.labels
            segs = batch.segs
            clss = batch.clss
            mask = batch.mask
            mask_cls = batch.mask_cls

            sent_scores_list = []
            mask_loss_list = []
            count = 0
            for _ in range(segs.size()[1] // 512):

                start, end = count * 512, (count * 512 + 512)
                clss_, mask_cls_ = get_clss_split(clss, mask_cls, start, end)
                tmp_score, mask_loss_tmp = self.model(
                    src[:, start:end, ...],
                    segs[:, start:end, ...],
                    clss_,
                    mask[:, start:end, ...],
                    mask_cls_,
                )
                sent_scores_list.append(tmp_score)
                mask_loss_list.append(mask_loss_tmp)
                count += 1
            if count == 0:
                return
            elif count == 1:
                sent_scores = sent_scores_list[0]
                mask_loss = mask_loss_list[0]
            else:
                sent_scores = torch.cat(sent_scores_list, dim=1)
                mask_loss = torch.cat(mask_loss_list, dim=1)
            # sent_scores, mask = self.model(src, segs, clss, mask, mask_cls)

            print("sent_score", sent_scores, "label", labels)
            loss = self.loss(sent_scores, labels.float())
            loss = (loss * mask_loss.float()).sum()
            (loss / loss.numel()).backward()
            # loss.div(float(normalization)).backward()

            # TODO(sujinhua): save to db base
            batch_stats = Statistics(float(loss.cpu().data.numpy()), normalization)

            total_stats.update(batch_stats)
            report_stats.update(batch_stats)

            # 4. Update the parameters and statistics.
            if self.grad_accum_count == 1:
                # Multi GPU gradient gather
                if self.n_gpu > 1:
                    grads = [
                        p.grad.data
                        for p in self.model.parameters()
                        if p.requires_grad and p.grad is not None
                    ]
                    distributed.all_reduce_and_rescale_tensors(grads, float(1))
                self.optim.step()

        # in case of multi step gradient accumulation,
        # update only after accum batches
        if self.grad_accum_count > 1:
            if self.n_gpu > 1:
                grads = [
                    p.grad.data
                    for p in self.model.parameters()
                    if p.requires_grad and p.grad is not None
                ]
                distributed.all_reduce_and_rescale_tensors(grads, float(1))
            self.optim.step()

    def _save(self, step):
        real_model = self.model
        # real_generator = (self.generator.module
        #                   if isinstance(self.generator, torch.nn.DataParallel)
        #                   else self.generator)

        model_state_dict = real_model.state_dict()
        # generator_state_dict = real_generator.state_dict()
        args_d = {
            key: value for key, value in self.args._get_kwargs() if key != "db_logger"
        }
        checkpoint = {
            "model": model_state_dict,
            # 'generator': generator_state_dict,
            "opt": args_d,
            "optim": self.optim,
        }
        checkpoint_path = os.path.join(
            self.args.model_path,
            "%s_model_%s_step_%d.pt"
            % (
                self.args.bert_data_path.split("/")[-1],
                self.args.db_logger.log_id,
                step,
            ),
        )
        logger.info("Saving checkpoint %s" % checkpoint_path)
        # checkpoint_path = '%s_step_%d.pt' % (FLAGS.model_path, step)
        if not os.path.exists(checkpoint_path):
            torch.save(checkpoint, checkpoint_path)
            return checkpoint, checkpoint_path

    def _start_report_manager(self, start_time=None):
        """
        Simple function to start report manager (if any)
        """
        if self.report_manager is not None:
            if start_time is None:
                self.report_manager.start()
            else:
                self.report_manager.start_time = start_time

    def _maybe_gather_stats(self, stat):
        """
        Gather statistics in multi-processes cases

        Args:
            stat(:obj:onmt.utils.Statistics): a Statistics object to gather
                or None (it returns None in this case)

        Returns:
            stat: the updated (or unchanged) stat object
        """
        if stat is not None and self.n_gpu > 1:
            return Statistics.all_gather_stats(stat)
        return stat

    def _maybe_report_training(self, step, num_steps, learning_rate, report_stats):
        """
        Simple function to report training stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_training` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_training(
                step, num_steps, learning_rate, report_stats, multigpu=self.n_gpu > 1
            )

    def _report_step(self, learning_rate, step, train_stats=None, valid_stats=None):
        """
        Simple function to report stats (if report_manager is set)
        see `onmt.utils.ReportManagerBase.report_step` for doc
        """
        if self.report_manager is not None:
            return self.report_manager.report_step(
                learning_rate, step, train_stats=train_stats, valid_stats=valid_stats
            )

    def _maybe_save(self, step):
        """
        Save the model if a model saver is set
        """
        if self.model_saver is not None:
            self.model_saver.maybe_save(step)
