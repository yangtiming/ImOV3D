import os
import sys
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, MagicMock
import numpy as np
import torch
import pytest

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test files."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def temp_file():
    """Create a temporary file for testing."""
    fd, path = tempfile.mkstemp()
    os.close(fd)
    yield Path(path)
    try:
        os.unlink(path)
    except OSError:
        pass


@pytest.fixture
def sample_config():
    """Sample configuration dictionary for testing."""
    return {
        'dataset_config': {
            'train_dataset': 'scannet',
            'val_dataset': 'scannet',
            'data_path': '/tmp/data',
            'batch_size': 8,
            'num_workers': 4,
        },
        'model_config': {
            'num_class': 18,
            'num_heading_bin': 12,
            'num_size_cluster': 18,
            'mean_size_arr': [[0.76966727, 0.8116021, 0.92573744]] * 18,
            'input_feature_dim': 0,
            'width': 1,
            'bn_momentum': 0.1,
            'sync_bn': False,
        },
        'training_config': {
            'learning_rate': 0.001,
            'weight_decay': 0.0001,
            'max_epoch': 100,
            'lr_decay_steps': [80, 90],
            'lr_decay_rates': [0.1, 0.1],
        }
    }


@pytest.fixture
def mock_torch_device():
    """Mock torch device for testing."""
    with pytest.mock.patch('torch.cuda.is_available', return_value=True):
        with pytest.mock.patch('torch.device') as mock_device:
            mock_device.return_value = 'cuda:0'
            yield mock_device


@pytest.fixture
def sample_point_cloud():
    """Generate a sample point cloud for testing."""
    np.random.seed(42)
    num_points = 1000
    point_cloud = np.random.rand(num_points, 3) * 10.0  # 3D coordinates
    colors = np.random.rand(num_points, 3) * 255  # RGB colors
    return {
        'points': point_cloud,
        'colors': colors,
        'num_points': num_points
    }


@pytest.fixture
def sample_bbox_3d():
    """Generate sample 3D bounding boxes for testing."""
    return {
        'center': np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 2.0]]),
        'size': np.array([[2.0, 1.0, 1.8], [1.5, 2.5, 1.2]]),
        'heading_angle': np.array([0.1, 0.5]),
        'semantic_label': np.array([0, 1]),
        'instance_label': np.array([1, 2])
    }


@pytest.fixture
def sample_image_features():
    """Generate sample image features for testing."""
    batch_size = 2
    num_views = 4
    feature_dim = 256
    height = 32
    width = 32
    
    return torch.randn(batch_size, num_views, feature_dim, height, width)


@pytest.fixture
def mock_dataset():
    """Mock dataset for testing."""
    mock_ds = Mock()
    mock_ds.__len__ = Mock(return_value=100)
    mock_ds.__getitem__ = Mock(return_value={
        'point_clouds': torch.randn(20000, 3),
        'image_features': torch.randn(4, 256, 32, 32),
        'bboxes_3d': torch.randn(10, 7),
        'class_labels': torch.randint(0, 18, (10,)),
        'scan_idx': 0
    })
    return mock_ds


@pytest.fixture
def mock_dataloader(mock_dataset):
    """Mock dataloader for testing."""
    from torch.utils.data import DataLoader
    return DataLoader(mock_dataset, batch_size=2, shuffle=False)


@pytest.fixture
def sample_camera_params():
    """Sample camera parameters for testing."""
    return {
        'intrinsic': np.array([
            [525.0, 0.0, 320.0],
            [0.0, 525.0, 240.0],
            [0.0, 0.0, 1.0]
        ]),
        'extrinsic': np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]),
        'width': 640,
        'height': 480
    }


@pytest.fixture
def mock_wandb():
    """Mock wandb for testing."""
    mock_wandb = Mock()
    mock_wandb.init = Mock()
    mock_wandb.log = Mock()
    mock_wandb.finish = Mock()
    return mock_wandb


@pytest.fixture
def mock_model():
    """Mock model for testing."""
    model = Mock()
    model.train = Mock()
    model.eval = Mock()
    model.cuda = Mock(return_value=model)
    model.parameters = Mock(return_value=[torch.randn(10, 10, requires_grad=True)])
    model.forward = Mock(return_value={
        'objectness_scores': torch.randn(2, 256, 2),
        'center': torch.randn(2, 256, 3),
        'heading_scores': torch.randn(2, 256, 12),
        'size_scores': torch.randn(2, 256, 18),
        'sem_cls_scores': torch.randn(2, 256, 18)
    })
    return model


@pytest.fixture
def mock_optimizer():
    """Mock optimizer for testing."""
    optimizer = Mock()
    optimizer.zero_grad = Mock()
    optimizer.step = Mock()
    optimizer.state_dict = Mock(return_value={})
    optimizer.load_state_dict = Mock()
    return optimizer


@pytest.fixture
def mock_scheduler():
    """Mock learning rate scheduler for testing."""
    scheduler = Mock()
    scheduler.step = Mock()
    scheduler.state_dict = Mock(return_value={})
    scheduler.load_state_dict = Mock()
    return scheduler


@pytest.fixture
def sample_loss_dict():
    """Sample loss dictionary for testing."""
    return {
        'objectness_loss': torch.tensor(0.5),
        'center_loss': torch.tensor(0.3),
        'heading_cls_loss': torch.tensor(0.2),
        'size_cls_loss': torch.tensor(0.4),
        'sem_cls_loss': torch.tensor(0.6),
        'total_loss': torch.tensor(2.0)
    }


@pytest.fixture
def sample_metrics():
    """Sample evaluation metrics for testing."""
    return {
        'mAP@0.25': 0.65,
        'mAP@0.5': 0.45,
        'AR@100': 0.70,
        'per_class_ap': {
            'cabinet': 0.60,
            'bed': 0.75,
            'chair': 0.55,
            'sofa': 0.80
        }
    }


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path, monkeypatch):
    """Set up test environment variables and paths."""
    # Set temporary paths for testing
    monkeypatch.setenv('TMPDIR', str(tmp_path))
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    
    # Mock heavy imports that might not be available in test environment
    sys.modules['pointnet2._ext'] = Mock()
    
    # Ensure deterministic behavior
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)


@pytest.fixture
def capture_logs():
    """Capture logs for testing."""
    import logging
    from io import StringIO
    
    log_capture = StringIO()
    handler = logging.StreamHandler(log_capture)
    logger = logging.getLogger()
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    
    yield log_capture
    
    logger.removeHandler(handler)


# Test markers for easy test selection
pytest.mark.unit = pytest.mark.unit
pytest.mark.integration = pytest.mark.integration  
pytest.mark.slow = pytest.mark.slow