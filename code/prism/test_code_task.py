import openai
import os
import re
import time
import base64
import imageio
import traceback
import numpy as np
from datetime import datetime
from PIL import Image
import io
import threading

from arguments import get_config
from interfaces import setup_LMP
from visualizers import ValueMapVisualizer
from envs.rlbench_env import VoxPoserRLBench
from utils import set_lmp_objects
from rlbench import tasks
import concurrent.futures
import ast
import sys
import json


from multiprocessing import Process


from grounded_sam2_hf_model_drawer import detect_objects





# print(result)


# result = detect_objects(
#     image_path="/path/to/image.png",
#     sam2_checkpoint="/path/to/sam2_checkpoint.pt",
#     sam2_config="/path/to/sam2_config.yaml"
# )

# annotate_and_save("/path/to/image.png", result, "annotated_output.jpg")



def load_api_key(filepath='key.json'):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get('openai_api_key')
    except Exception as e:
        print("Error loading API key:", e)
        sys.exit(1)




def main():
  
    openai.api_key = load_api_key()

    # Unique save directory per run
    base_dir = "/PRISM/data/test"
    save_directory = os.path.join(base_dir, f"episode_{1}/recordings")
    os.makedirs(save_directory, exist_ok=True)

    config = get_config('rlbench')
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer, save_directory=save_directory)
    env._lock = threading.Lock()
    env.image_counter = 1

    lmps, lmps_2, lmp_env = setup_LMP(env, config, debug=False)
    env.load_task(tasks.StackCups)
    descriptions, obs = env.reset()
    save_directory = os.path.join(base_dir, f"episode_{1}/front")
    os.makedirs(save_directory, exist_ok=True)
    initial_image = obs.front_rgb_0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(save_directory, f"front_{timestamp}.png")
    print(f'{image_path}')
    imageio.imwrite(image_path, initial_image)


    text_prompt = ""
    obs_dict = detect_objects(img_path=image_path)

    # Remove 'score' key from each object
    for obj in obs_dict:
        obj.pop('score', None)


    print(obs_dict)



    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"



    

    set_lmp_objects(lmps, env.get_object_names())
    set_lmp_objects(lmps_2, env.get_object_names())

    

    gripper_ui = lmps['composer_ui']
    gripper_ui_2 = lmps_2['composer_ui_2']

#     turn 90 degree along the x-axis
# move to 20 cm right to handle top center
# grasp the handle top
# move 20cm right to the handle top

    # gripper_ui_2("grasp the yellow ring")
    # # gripper_ui_2("move to 10cm top of the center of the green spoke")
    # # gripper_ui_2("move to 5 cm top of the center of the green spoke")
    # # gripper_ui_2("open gripper")
    # # gripper_ui_2("move to 10cm top of the center of the green spoke")
    # # gripper_ui_2("move to default pose")


    gripper_ui_2("move ee down by 40cm")
    gripper_ui("grasp the red cup")
    # gripper_ui("move to handle top center")
    # gripper_ui("close gripper")
    # gripper_ui("move 20cm right to the handle top")
    # # gripper_ui("open gripper")
    # gripper_ui("move to 10cm top of the center of red spoke")
    # gripper_ui("move to default pose")

    # gripper_ui("open gripper")
    # gripper_ui("move to default position")

    # gripper_ui_2("grasp the green pepper")
    # gripper_ui_2("move to 10 cm top of the center of the gray container")
    # gripper_ui_2("open gripper")
    # gripper_ui_2("move to default position")


    # gripper_ui("grasp the yellow pepper")
    # gripper_ui("move to 10 cm top of the center of the gray container")
    # gripper_ui("open gripper")
    # gripper_ui("move to default position")
    
if __name__ == "__main__":
    main()
