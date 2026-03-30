from openvino.runtime import Core
import wandb

run = wandb.init()
ie = Core()
artifact = run.use_artifact('florian_wirtz_problem/florian-wirtz-problem/wirtz-tracking-model:v0', type='model')
artifact_dir = artifact.download()

# The artifact folder will contain both the .pt and the openvino_version
model = ie.read_model(f"{artifact_dir}\\openvino_version\\best.xml")
compiled_model = ie.compile_model(model, "CPU")

print("OpenVINO model loads fine")