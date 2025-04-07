from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Literal, List, Optional, Dict, Any
import os
import uuid
import numpy as np
from pathlib import Path
import onnxruntime as ort
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_regression, make_classification, make_blobs
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from openvino import Core, Model, convert_model, save_model
from sklearnex import patch_sklearn
import json
import logging

patch_sklearn()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ML Model Training and Serving API",
              description="A service for training and serving ML models using scikit-learn, ONNX, and OpenVINO")

MODEL_REPO = Path("./model_repo")
ONNX_DIR = MODEL_REPO / "onnx"
IR_DIR = MODEL_REPO / "ir"
MODEL_CONFIG = MODEL_REPO / "model_config.json"

ONNX_DIR.mkdir(parents=True, exist_ok=True)
IR_DIR.mkdir(parents=True, exist_ok=True)

core = Core()


class ModelConfig:
    def __init__(self):
        self.config_file = MODEL_CONFIG
        self.config = self._load_config()

    def _load_config(self) -> Dict:
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                return json.load(f)
        return {}

    def save_model_info(self, algorithm: str, version: str, features: int, supports_openvino: bool):
        if algorithm not in self.config:
            self.config[algorithm] = {}

        self.config[algorithm][version] = {
            'features': features,
            'supports_openvino': supports_openvino,
            'onnx_path': str(ONNX_DIR / f"{algorithm}_{version}.onnx"),
            'ir_path': str(IR_DIR / algorithm / version) if supports_openvino else None
        }

        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)


model_config = ModelConfig()


class TrainRequest(BaseModel):
    algorithm: Literal['linear_regression', 'svm', 'kmeans']
    version: Optional[str] = None
    hyperparameters: Optional[Dict[str, Any]] = None


class InferenceRequest(BaseModel):
    algorithm: Literal['linear_regression', 'svm', 'kmeans']
    version: str
    input_data: List[List[float]]


@app.post("/train", response_class=JSONResponse)
async def train_model(req: TrainRequest):
    try:
        model_name = req.algorithm
        version = req.version or str(uuid.uuid4())
        hyperparameters = req.hyperparameters or {}

        X, y = None, None
        model = None
        n_features = 4

        if model_name == 'linear_regression':
            X, y = make_regression(n_samples=100, n_features=n_features, noise=0.1)
            model = LinearRegression(**hyperparameters)
        elif model_name == 'svm':
            X, y = make_classification(n_samples=100, n_features=n_features)
            model = SVC(probability=True, **hyperparameters)
        elif model_name == 'kmeans':
            X, _ = make_blobs(n_samples=100, n_features=n_features, centers=3)
            model = KMeans(n_clusters=3, **hyperparameters)
        model.fit(X, y if y is not None else X)

        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        onx = convert_sklearn(model, initial_types=initial_type)

        onnx_path = ONNX_DIR / f"{model_name}_{version}.onnx"
        with open(onnx_path, "wb") as f:
            f.write(onx.SerializeToString())

        supports_openvino = False
        try:
            if model_name == 'kmeans':
                ir_path = IR_DIR / model_name / version
                ir_path.mkdir(parents=True, exist_ok=True)
                save_model(convert_model(onnx_path), str(ir_path / "model.xml"))
                supports_openvino = True
                logger.info(f"Successfully converted {model_name} to IR format")
        except Exception as e:
            logger.warning(f"Could not convert {model_name} to IR format: {str(e)}")

        model_config.save_model_info(
            algorithm=model_name,
            version=version,
            features=n_features,
            supports_openvino=supports_openvino
        )

        return JSONResponse({
            "status": "success",
            "message": f"Model {model_name} version {version} trained successfully",
            "version": version,
            "supports_openvino": supports_openvino
        })

    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_class=JSONResponse)
async def predict(req: InferenceRequest):
    try:
        if not req.input_data:
            raise HTTPException(status_code=400, detail="No input data provided")

        model_info = model_config.config.get(req.algorithm, {}).get(req.version)
        if not model_info:
            raise HTTPException(status_code=404, detail=f"Model {req.algorithm} version {req.version} not found")

        input_data = np.array(req.input_data, dtype=np.float32)
        if input_data.shape[1] != model_info['features']:
            raise HTTPException(
                status_code=400,
                detail=f"Expected {model_info['features']} features, got {input_data.shape[1]}"
            )

        print(model_info)
        if model_info['supports_openvino']:
            ir_path = Path(model_info['ir_path']) / "model.xml"
            model = core.read_model(ir_path)
            compiled_model = core.compile_model(model)
            results = compiled_model(input_data)[0]
        else:
            onnx_path = model_info['onnx_path']
            session = ort.InferenceSession(onnx_path)
            input_name = session.get_inputs()[0].name
            results = session.run(None, {input_name: input_data})[0]

        return JSONResponse({
            "predictions": results.tolist()
        })

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models", response_class=JSONResponse)
async def list_models():
    return JSONResponse(model_config.config)
    model = KMeans(n_clusters=3)


    model.fit(X, y)
    initial_type = [('input', FloatTensorType([None, 4]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    model_dir = os.path.join(MODEL_REPO, model_name, version)
    os.makedirs(model_dir, exist_ok=True)
    onnx_path = os.path.join(model_dir, 'model.onnx')

    with open(onnx_path, "wb") as f:
        f.write(onnx_model.SerializeToString())

    convert_model(onnx_path, output=model_dir)

    return {
        "status": "success",
        "model": model_name,
        "version": version,
        "onnx_path": onnx_path,
        "model_ir_path": os.path.join(model_dir, "model.xml")
    }

