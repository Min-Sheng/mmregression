import copy
from pathlib import Path

import numpy as np
from torch.utils.data import Dataset

from mmcls.core.evaluation import precision_recall_f1, support
from mmcls.models.losses import accuracy
from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class HemaAutoClsDataset(Dataset):
    """Hema Auto dataset

    Args:
        root_dir (str): the root of data dir
        data_prefix (str): the prefix of data path
        ann_file (str | None): the annotation file. When ann_file is str, the data list is expected to read from the ann_file.
        pipeline (list): a list of dict, where each element represents
            a operation defined in `mmcls.datasets.pipelines`
        test_mode (bool): in train mode or test mode
    """

    CLASSES = None

    def __init__(self,
                 root_dir,
                 data_prefix,
                 ann_file,
                 pipeline,
                 valid_thresholds=(10, 20),
                 test_mode=False):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.valid_thresholds = valid_thresholds
        self.pipeline = Compose(pipeline)
        self.data_infos = self.load_annotations()
        self.test_mode = test_mode

    def load_annotations(self):

        with open(self.ann_file) as f:
            data_list = [x.strip() for x in f.readlines()]

        self.data_list = data_list

        data_infos = []
        for filename in self.data_list:
            img_path = str(Path(self.root_dir) / ('images/' + filename + '.png'))
            label_path = str(Path(self.root_dir) / ('annotations/' + filename + '.npy'))
            label = np.load(label_path)
            info = {'img_prefix': self.data_prefix}
            info['img_info'] = {'filename': img_path}
            gt_label = copy.deepcopy(label[..., 1])
            class_id = 1
            for i, valid_threshold in enumerate(self.valid_thresholds):
                if i == 0:
                    gt_label = np.where(gt_label > valid_threshold, class_id, 0)
                else:
                    gt_label = np.where(gt_label > valid_threshold, class_id, gt_label)
                    
                class_id = class_id + 1

            info['gt_label'] = gt_label.astype(np.uint8)
            data_infos.append(info)
        return data_infos

    def get_gt_labels(self):
        """Get all ground-truth labels (categories).

        Returns:
            np.ndarray: categories for all images.
        """

        gt_labels = np.array([data['gt_label'] for data in self.data_infos])
        return gt_labels
    
    def prepare_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        return self.pipeline(results)

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        return self.prepare_data(idx)

    def evaluate(self,
                 results,
                 metric='accuracy',
                 metric_options=None,
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `accuracy`.
            metric_options (dict, optional): Options for calculating metrics.
                Allowed keys are 'topk', 'thrs' and 'average_mode'.
                Defaults to None.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if metric_options is None:
            metric_options = {'topk': (1, 5)}
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'accuracy', 'precision', 'recall', 'f1_score', 'support'
        ]
        eval_results = {}
        results = np.array(results)
        num_classes = results.shape[-1]
        results = results.reshape(-1, num_classes)
        gt_labels = self.get_gt_labels()
        gt_labels = gt_labels.flatten()

        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        topk = metric_options.get('topk', (1, 5))
        thrs = metric_options.get('thrs')
        average_mode = metric_options.get('average_mode', 'macro')

        if 'accuracy' in metrics:
            if thrs is not None:
                acc = accuracy(results, gt_labels, topk=topk, thrs=thrs)
            else:
                acc = accuracy(results, gt_labels, topk=topk)
            if isinstance(topk, tuple):
                eval_results_ = {
                    f'accuracy_top-{k}': a
                    for k, a in zip(topk, acc)
                }
            else:
                eval_results_ = {'accuracy': acc}
            if isinstance(thrs, tuple):
                for key, values in eval_results_.items():
                    eval_results.update({
                        f'{key}_thr_{thr:.2f}': value.item()
                        for thr, value in zip(thrs, values)
                    })
            else:
                eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})

        if 'support' in metrics:
            support_value = support(
                results, gt_labels, average_mode=average_mode)
            eval_results['support'] = support_value

        precision_recall_f1_keys = ['precision', 'recall', 'f1_score']
        if len(set(metrics) & set(precision_recall_f1_keys)) != 0:
            if thrs is not None:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode, thrs=thrs)
            else:
                precision_recall_f1_values = precision_recall_f1(
                    results, gt_labels, average_mode=average_mode)
            for key, values in zip(precision_recall_f1_keys,
                                   precision_recall_f1_values):
                if key in metrics:
                    if isinstance(thrs, tuple):
                        eval_results.update({
                            f'{key}_thr_{thr:.2f}': value
                            for thr, value in zip(thrs, values)
                        })
                    else:
                        eval_results[key] = values

        return eval_results
