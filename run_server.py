import hydra
import mlflow
from omegaconf import DictConfig


@hydra.main(config_path="configs", config_name="infer", version_base="1.2")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)

    onnx_model = mlflow.pyfunc.load_model(cfg.model_uri)
    print(onnx_model)


if __name__ == "__main__":
    main()
