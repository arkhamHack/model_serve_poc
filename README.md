POC for OpenVINO Model Serve Idea
This poc uses a fastapi service to give a brief demo of the proposal
The /train endpoint -> accepts models from input with hyperparameters and stores it as an onnx file and if openvino supported then as an ir model.
The /predict endpoint -> calls model for inference with a payload 

Run:
pip install -r requirements.txt
uvicorn main:app --reload

Example curl:
For Train API:
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{
    "algorithm": "kmeans"
}'

For Kmeans API:
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
