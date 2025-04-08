# POC for OpenVINO Model Serve Idea

This POC uses a FastAPI service to give a brief demo of the proposal.  
The `/train` endpoint → accepts models from input with hyperparameters and stores it as an ONNX file, and if OpenVINO-supported, then also as an IR model.  
The `/predict` endpoint → calls the model for inference with a payload.

## ▶️ Run

```bash
pip install -r requirements.txt
uvicorn main:app --reload
```

## Example curl:
## For Train API:
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "kmeans"
}'
```

For Predict API:
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "kmeans",
    "version": "322edaf9-762b-4d9a-b84b-def2a9f2d7e9",
    "input_data": [
      [1.0, 2.0, 3.0, 4.0],
      [4.0, 3.0, 2.0, 1.0]
    ]
}'
```
