import os
import unittest
import numpy as np
from unittest import TestCase
from mlwrapper import MLWrapper


class MLWrapperTest(TestCase):
    def setUp(self):
        self.test_image_1 = np.ones(shape=(3, 40, 40, 1))
        self.test_image_1[0,:20,:,:] = 0.
        self.test_image_1[1,:,:,:] = 0.
        self.test_image_1[2,20:,:,:] = 0.

        self.test_image_2 = np.ones(shape=(3, 1, 40, 40))
        self.test_image_2[0,:, 20:,:] = 0.
        self.test_image_2[1,:,:,:] = 0.
        self.test_image_2[2,:, :20,:] = 0.

        self.test_kwargs = {
            "experiment parameter": 42,
            "additional": "information",
            }

    def log_values(self, logger, result=42):
        logger.log_args(**{"run param": "value"})
        for step in range(0, 50):
            logger.log_scalar("test_loss", value=100 - step * 1.5, step=step)
            logger.log_scalar("test_acc", value=0.00 + step * 0.01, step=step)
        logger.log_images("test_image", self.test_image_1, 1)
        logger.log_images("test_image", self.test_image_2, 2, channel_first=True)
        logger.log_metric("result", result)

    def test_logger_decorator(self):
        @MLWrapper(mlflow_dir="/tmp/mlruns/")
        def test(**kwargs):
            logger = kwargs["logger"]
            self.log_values(logger)
            return logger.exp_id
    
        # Check multiple calls of same function
        experiment_id = test(**self.test_kwargs)
        assert(experiment_id == test(**self.test_kwargs))

        @MLWrapper(mlflow_dir="/tmp/mlruns/", append_to_experiment_id=experiment_id)
        def test2(**kwargs):
            logger = kwargs["logger"]
            self.log_values(logger)
            return logger.exp_id

        # Check appending to exising experiment
        assert(experiment_id == test2(**self.test_kwargs))

    def test_logger_contex_decorator_one_instance(self):
        # all runs should be stored in one experiment directory (same exp_ids)
        exp_ids = []
        logger = MLWrapper(mlflow_dir="/tmp/mlruns/", **self.test_kwargs)
        for _ in range(5):
            with logger as run_logger:
                self.log_values(run_logger)
                exp_ids.append(run_logger.exp_id)
        assert len(set(exp_ids)) == 1

    def test_logger_contex_decorator_multiple_instances(self):
        # all runs should be in different directories (different exp_ids)
        exp_ids = []
        for _ in range(5):
            with MLWrapper(mlflow_dir="/tmp/mlruns/", **self.test_kwargs) as logger:
                self.log_values(logger)
                exp_ids.append(logger.exp_id)

        assert len(set(exp_ids)) == 5

    def test_logger_logging_files(self):
        storage = ""
        with MLWrapper(mlflow_dir="/tmp/mlruns/", **self.test_kwargs) as logger:
                self.log_values(logger)
                storage = logger.storage
        files = os.listdir(storage)
        assert "logs.txt" in files
        assert len(files) == 2

    def test_logger_args(self):
        storage = ""
        with MLWrapper(mlflow_dir="/tmp/mlruns/", **self.test_kwargs) as logger:
                self.log_values(logger)
                storage = logger.storage
        param_dir = os.path.join("/", *storage.split("/")[:-1], "params")
        files = os.listdir(param_dir)
        for k, v in self.test_kwargs.items():
            assert k in files
            fname = os.path.join(param_dir, k)
            with open(fname, 'r') as f:
                assert str(f.read()) == str(v)