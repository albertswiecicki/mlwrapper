# MLWrapper v0.1

MLwrapper is a context manager that helps you store experiment results.

Context manager \_\_enter\_\_ creates mlflow run and store logged values inside. It allows logging following stuff:

- script arguments
- images
- scalars
- metrics

## Quick start

```python

        # data to log
        kwargs = {
            "experiment parameter": 42,
        }
        test_image_1 = np.ones(shape=(3, 40, 40, 1))
        test_image_1[0,:20,:,:] = 0.
        test_image_1[1,:,:,:] = 0.
        test_image_1[2,20:,:,:] = 0.

        test_image_2 = np.ones(shape=(3, 1, 40, 40))
        test_image_2[0,:, 20:,:] = 0.
        test_image_2[1,:,:,:] = 0.
        test_image_2[2,:, :20,:] = 0.

        def test(logger):
            logger.log_args(**{"run param": "value"})
            for step in range(0, 50):
                logger.log_scalar("test_loss", value=100 - step * 1.5, step=step)
                logger.log_scalar("test_acc", value=0.00 + step * 0.01, step=step)
            logger.log_images("test_image", test_image_1, 1)
            logger.log_images("test_image", test_image_2, 2, channel_first=True)
            logger.log_metric("result", result)

        # approach 1
        Experiment = MLWrapper(mlflow_dir="/tmp/mlruns/", **kwargs)
        with Experiment as logger:
            test(logger)

        # approach 2
        with MLWrapper(mlflow_dir="/tmp/mlruns/", **kwargs) as logger:
            test(logger)

        # approach 3
        Experiment = MLWrapper(mlflow_dir="/tmp/mlruns/", **kwargs)
        wrapped_test = Experiment(test) # func needs to accept "logger" or "**kwarg"
        wrapped_test()
```

## Testing

Testing will create files under /tmp directory. Those files are not deleted automatically.

```bash
python3 -m unittest discover .
```

## References

[mlflow](https://mlflow.org/)

[tensorboard](https://www.tensorflow.org/tensorboard)