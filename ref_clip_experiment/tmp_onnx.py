import onnx

# モデルのパスを指定
model_path = '/workspace/osada_ws/ref_clip_experiment/models/clip-vit-base-patch32-visual-float16.onnx'

# モデルの読み込みと検証
model = onnx.load(model_path)
onnx.checker.check_model(model)
print("ONNXモデルは正常です。")
