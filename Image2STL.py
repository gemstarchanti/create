import os
import sys
import importlib
import torch
import time
import rembg
import numpy as np
import os
import pymeshlab
from tsr.utils import save_video # Import save_video
from tsr.utils import remove_background, resize_foreground
from tsr.system import TSR

# Define the path to the TripoSR main directory
triposr_root_path = '/content/TripoSR'

# Clone TripoSR if it doesn't exist
if not os.path.exists(triposr_root_path):
    # Ensure we are in /content before cloning to place TripoSR correctly
    current_dir_before_clone = os.getcwd()
    if current_dir_before_clone != '/content':
        %cd /content
    !git clone https://github.com/VAST-AI-Research/TripoSR.git

# Change directory to TripoSR for installing requirements and running other commands
%cd {triposr_root_path}

# Install requirements
!pip install -r requirements.txt

# Install onnxruntime explicitly, as it's a dependency of rembg and might not be pulled automatically
!pip install onnxruntime

# Install pymeshlab
!pip install pymeshlab

# Explicitly install a compatible version of huggingface-hub
# transformers==4.35.0 requires huggingface-hub<1.0,>=0.16.4 and uses functionality from >=0.19.0
# accelerate 1.12.0 requires huggingface_hub>=0.21.0
!pip install huggingface_hub==0.21.0 --force-reinstall

# Add TripoSR's root directory to sys.path for module discovery
# This ensures that 'tsr' module within TripoSR is discoverable
if triposr_root_path not in sys.path:
    sys.path.insert(0, triposr_root_path)

# Create an empty __init__.py in the tsr directory if it doesn't exist
# This ensures Python recognizes 'tsr' as a package.
ts_init_path = os.path.join(triposr_root_path, 'tsr', '__init__.py')
if not os.path.exists(ts_init_path):
    print(f"[DEBUG] Creating empty __init__.py at: {ts_init_path}")
    with open(ts_init_path, 'w') as f:
        pass

# Debugging statements
print(f"[DEBUG] Current Working Directory: {os.getcwd()}")
print(f"[DEBUG] sys.path contains: {sys.path}")
print(f"[DEBUG] Path to tsr/system.py: {os.path.join(triposr_root_path, 'tsr', 'system.py')}")
print(f"[DEBUG] Does tsr/system.py exist? {os.path.exists(os.path.join(triposr_root_path, 'tsr', 'system.py'))}")
print(f"[DEBUG] Does tsr/__init__.py exist? {os.path.exists(ts_init_path)}")

# Invalidate import caches (keep this as it can help with fresh imports)
importlib.invalidate_caches()

# Removed: del sys.modules['tsr'] and del sys.modules['tsr.system'] as they might be causing problems for subsequent cells.
# Removed: Direct imports of torch, time, PIL, numpy, TSR, etc. as they should be done in the cells that need them.

class Timer:
    def __init__(self):
        self.items = {}
        self.time_scale = 1000.0  # ms
        self.time_unit = "ms"
    def start(self, name: str) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.items[name] = time.time()
    def end(self, name: str) -> float:
        if name not in self.items:
            return
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start_time = self.items.pop(name)
        delta = time.time() - start_time
        t = delta * self.time_scale
        print(f"{name} finished in {t:.2f}{self.time_unit}.")
timer = Timer()

# Define the missing variables
pretrained_model_name_or_path = "stabilityai/TripoSR"
chunk_size = 2048 # A common default value

timer.start("Initializing model")
model = TSR.from_pretrained(
    pretrained_model_name_or_path,
    config_name="config.yaml",
    weight_name="model.ckpt",
)
model.renderer.set_chunk_size(chunk_size)
model.to(device)
timer.end("Initializing model")
timer.start("Processing images")
images = []

# Define missing variables
foreground_ratio = 0.85  # Common default value for foreground ratio
output_dir = "output"  # Define output directory
os.makedirs(output_dir, exist_ok=True)

rembg_session = rembg.new_session()
image = remove_background(original_image, rembg_session)
image = resize_foreground(image, foreground_ratio)

if image.mode == "RGBA":
    image = np.array(image).astype(np.float32) / 255.0
    image = image[:, :, :3] * image[:, :, 3:4] + (1 - image[:, :, 3:4]) * 0.5
    image = Image.fromarray((image * 255.0).astype(np.uint8))

image_dir = os.path.join(output_dir, str(0))
os.makedirs(image_dir, exist_ok=True)
image.save(os.path.join(image_dir, "input.png"))
images.append(image)
timer.end("Processing images")


render = True  # Set to False if you don't want to render videos and images
model_save_format = "obj" # or "glb"

for i, image in enumerate(images):
    print(f"Running image {i + 1}/{len(images)} ...")
    timer.start("Running model")
    with torch.no_grad():
        scene_codes = model([image], device=device)
    timer.end("Running model")

    if render:
        timer.start("Rendering")
        render_images = model.render(scene_codes, n_views=30, return_type="pil")
        for ri, render_image in enumerate(render_images[0]):
            render_image.save(os.path.join(output_dir, str(i), f"render_{ri:03d}.png"))
        save_video(
            render_images[0], os.path.join(output_dir, str(i), "render.mp4"), fps=30
        )
        timer.end("Rendering")

    timer.start("Exporting mesh")
    meshes = model.extract_mesh(scene_codes, has_vertex_color=False)
    mesh_file = os.path.join(output_dir, str(i), f"mesh.{model_save_format}")
    meshes[0].export(mesh_file)
    timer.end("Exporting mesh")
print("Processing complete.")


obj_file = "/content/TripoSR/output/0/mesh.obj"
# Load the .obj mesh
ms = pymeshlab.MeshSet()
ms.load_new_mesh(obj_file)
mesh = ms.current_mesh()
# Convert to .stl format
stl_file = 'model.stl'
ms.save_current_mesh(stl_file)
