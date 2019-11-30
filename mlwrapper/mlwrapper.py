import os
import uuid
import mlflow
import logging
import numpy as np
import tensorflow as tf
from numbers import Number
from functools import wraps
from mlflow import get_artifact_uri, log_param, log_metric
from mlflow.exceptions import MlflowException
from contextlib import ContextDecorator


class MLWrapper(ContextDecorator):
    def __init__(self,
            mlflow_dir:str,
            exp_name:str = None,
            append_to_experiment_id:int = None,
            verbose:bool = True,
            logging_level:int = logging.ERROR,
            **kwargs,
            ):
        self.running = False
        self.mlflow_dir = mlflow_dir
        self.exp_id = append_to_experiment_id
        self.exp_name = exp_name
        self.logging_level = logging_level
        self.verbose = verbose
        self.kwargs = kwargs
        mlflow.set_tracking_uri(self.mlflow_dir)

    def __enter__(self):
        self._start_run()
        self.storage = get_artifact_uri()
        self._setup_logger()
        self.file_writer = tf.summary.create_file_writer(self.storage)
        self.running = True
        self.logger.info(f"MLWraper started with name {self.exp_name} and id {self.exp_id}")

        self.log_args()
        return self

    def __exit__(self, *exc):
        self.running = False
        mlflow.end_run()
        self.logger.info("MLWraper run finished")
        self.logger = None
        self.storage = None
        self.file_writer = None
        return False

    def __call__(self, func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            self.kwargs = kwargs
            with self as logger:
                return func(*args, **kwargs, **{"logger": logger})
        return wrapper

    def _start_run(self):
        if self.exp_id is None:
            self._gen_new_exp_id()
        try:
            mlflow.start_run(experiment_id=self.exp_id)
        except MlflowException:
            self._gen_new_exp_id()
            mlflow.start_run(experiment_id=self.exp_id)

    def _gen_new_exp_id(self):
        try:
            self.exp_id = mlflow.create_experiment(self.exp_name)
        except MlflowException:
            self.exp_name = str(uuid.uuid4())
            self.exp_id = mlflow.create_experiment(self.exp_name)

    def _setup_logger(self):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(self.logging_level)

        logging_file = os.path.join(self.storage, "logs.txt")
        file_handler = logging.FileHandler(logging_file)
        file_handler.setLevel(self.logging_level)
        self.logger.addHandler(file_handler)
        if self.verbose:
            stream_handler = logging.StreamHandler()
            self.logger.addHandler(stream_handler)

    def log_args(self, **kwargs):
        assert self.running, "Attempt to log before entering context meneger"
        self.logger.debug("Attempt to log arguments")
        for k, v in self.kwargs.items():
            log_param(k, v)
        for k, v in kwargs.items():
            log_param(k, v)

    def log_images(self, description:str, img:np.array, step:int, channel_first:bool=False):
        assert self.running, "Attempt to log before entering context meneger"
        self.logger.debug("Attempt to log image/s")
        assert(len(img.shape) == 4), \
            "expected shape [b, c, h, w] or [b, h, w, c] if channel_first==True"
        if isinstance(img, np.ndarray):
            if channel_first:
                img = np.swapaxes(img, 1, 3)
                img = np.swapaxes(img, 1, 2)
            with self.file_writer.as_default():
                tf.summary.image(description, img, step=step)
                self.file_writer.flush()
        else:
            raise NotImplementedError("Functionality not implemented for type {} yet".format(type(img)))

    def log_scalar(self, tag:str, value:Number, step:int):
        assert self.running, "Attempt to log before entering context meneger"
        self.logger.debug("Attempt to log scalar")
        with self.file_writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)

    def log_metric(self, name:str, value:Number):
        assert self.running, "Attempt to log before entering context meneger"
        self.logger.debug("Attempt to log metric")
        log_metric(name, value)
