from mmdet.registry import DATASETS
from .coco import CocoDataset


@DATASETS.register_module()
class CarDDCocoDataset(CocoDataset):
    """Dataset for CarDD_COCO."""

    METAINFO = {
        'classes':
        ('dent', 'scratch', 'crack', 'glass shatter', 'lamp broken', 'tire flat'),
        # palette is a list of color tuples, which is used for visualization.
        'palette':
        [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
         (0, 60, 100)]
    }

