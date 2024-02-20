from onnx_opcounter import calculate_params
import onnx

model = onnx.load_model('/home/khuy/Recognition-System/face_recognition/arcface/weights/arcface_r100.onnx')
params = calculate_params(model)

print('Number of params:', params)