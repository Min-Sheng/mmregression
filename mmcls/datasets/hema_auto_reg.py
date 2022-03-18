import copy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from .builder import DATASETS
from .pipelines import Compose


@DATASETS.register_module()
class HemaAutoRegDataset(Dataset):
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
                 max_total=150,
                 test_mode=False):
        super(Dataset, self).__init__()
        self.root_dir = root_dir
        self.data_prefix = data_prefix
        self.ann_file = ann_file
        self.max_total = max_total
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
            label[..., 0 ] = np.clip(label[..., 0 ] / self.max_total, a_min=0, a_max=1)
            label[..., 1] = label[..., 1] / (label[..., 0] + 1e-8)
            info['gt_label'] = label.astype(np.float32)
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
                 metric='mae',
                 indices=None,
                 logger=None):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
                Default value is `mae`.
            indices (list, optional): The indices of samples corresponding to
                the results. Defaults to None.
            logger (logging.Logger | str, optional): Logger used for printing
                related information during evaluation. Defaults to None.
        Returns:
            dict: evaluation results
        """
        if isinstance(metric, str):
            metrics = [metric]
        else:
            metrics = metric
        allowed_metrics = [
            'mse', 'mae'
        ]
        eval_results = {}
        results = torch.from_numpy(np.array(results).transpose(0, 2, 3, 1))
        gt_labels = torch.from_numpy(self.get_gt_labels())
        
        if indices is not None:
            gt_labels = gt_labels[indices]
        num_imgs = len(results)
        assert len(gt_labels) == num_imgs, 'dataset testing results should '\
            'be of the same length as gt_labels.'

        invalid_metrics = set(metrics) - set(allowed_metrics)
        if len(invalid_metrics) != 0:
            raise ValueError(f'metric {invalid_metrics} is not supported.')

        if 'mse' in metrics:
            mse = F.mse_loss(results, gt_labels)
            mse = mse.cpu().detach()
            eval_results_ = {'mse': mse}

        if 'mae' in metrics:
            mae = F.l1_loss(results, gt_labels)
            mae = mae.cpu().detach()
            eval_results_ = {'mae': mae}
        
        eval_results.update(
                    {k: v.item()
                     for k, v in eval_results_.items()})
        return eval_results

