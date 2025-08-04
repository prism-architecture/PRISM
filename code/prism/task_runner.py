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
import shutil
from PIL import Image
import numpy as np
from multiprocessing import Process
import yaml

import warnings
warnings.simplefilter("error", RuntimeWarning)


description_history = []


# import task configuration
from task_configs import task_registry, collab_plan_prompt_path, subtask_prompt_path
from synchronization_methods import *

from grounded_sam2_hf_model import detect_objects




########################################## convert image ########################################################################


def get_image_base64(image_np):
    buffered = io.BytesIO()
    img = Image.fromarray(image_np)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")



########################################## save-history #####################################


def save_description_history(save_dir, filename="description_history.json"):
    """Save description_history to JSON file."""
    filepath = os.path.join(save_dir, filename)
    try:
        with open(filepath, "w") as f:
            json.dump(description_history, f, indent=2)
        print(f"[wait] 💾 Saved description history to {filepath}")
    except Exception as e:
        print(f"[wait] ❌ Failed to save description history: {e}")

###################################################################################################################################
################################################################ get_plan_from_the_vlm ############################################
###################################################################################################################################


def load_visual_and_instruction(yaml_path: str) -> tuple[str, str]:
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data["visual_prompt"], data["task_instruction"]


def get_plan_from_vlm(image_url, obs_dict, query, object_list, prompt_path):

    """
    Generate collaborative robot plans in [blocking_condition]:[plan] format,
    using visual layout and object affordances to plan a synchronized stacking task.
    """

    with open(task_registry.collab_plan_prompt_path, 'r') as f:
        collab_examples = f.read().strip()

    PLAN_PROMPT_SUFFIX = """

    Detected objects:
    {object_list}
    Bounding boxes:
    {obs_dict}
    
    Format:
    gripper: [natural language plan]
    gripper1: [natural language plan]

    Below are additional examples from other tasks:
    {collab_examples}

    Now generate high-level robot plans for this task: {query}
    """
    
    # task-specific visual and plan prompt
    # task-specific visual and plan prompt
    visual_prompt, task_instruction = load_visual_and_instruction(prompt_path)

    plan_prompt = (
        task_instruction.strip()
        + "\n\n"
        + PLAN_PROMPT_SUFFIX.strip().format(
            collab_examples=collab_examples,
            query=query
        )
    )

    plan_prompt = (
        task_instruction.strip()
        + "\n\n"
        + PLAN_PROMPT_SUFFIX.strip().format(
            object_list=object_list,
            obs_dict=obs_dict,
            collab_examples=collab_examples,
            query=query
        )
    )

    # VLM response
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in collaborative robot task planning. "
                    "You will coordinate actions between two robot arms to avoid collisions and ensure synchronized execution."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": visual_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    },
                    {"type": "text", "text": plan_prompt}
                ]
            }
        ],
        temperature=0.2,
        max_tokens=300,
    )

    response_text = response['choices'][0]['message']['content']
    print("Raw GPT Plan:\n", response_text)

    plan_dict = {}

    # Extract lines like: gripper: pick the red block and stack it...
    for line in response_text.strip().splitlines():
        line = line.strip()
        if line.startswith("gripper:") or line.startswith("gripper1:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                robot = parts[0].strip()
                plan = parts[1].strip()
                plan_dict[robot] = plan

    return plan_dict


############################################################################################################################
###################################### Get Subtask Response ################################################################
############################################################################################################################

import re
import openai


def get_subtask_response(env, gripper_query, gripper1_query):
    import openai
    import re

    WAIT_METHOD_NAMES = [
        "is_stacked_and_gripper_moved",
        "is_drawer_open_and_gripper_moved",
        "is_lid_open_and_gripper_moved",
        "is_ring_inserted_and_gripper_moved",
        "is_box_on_target",
        "is_handover_pose_done",
        "is_grasped",
        "is_gripper_opened",
        "wait_and_guess_the_cup"
    ]

    # Load example subtask patterns
    with open(task_registry.subtask_prompt_path, 'r') as f:
        subtask_examples = f.read().strip()

    system_content = (
        "You are a helpful assistant that writes clean Python-style subtask lists for controlling "
        "two robot arms (gripper and gripper1) in a tabletop manipulation setting.\n\n"
        "Output format must strictly follow:\n"
        "gripper:\n- subtask 1\n- subtask 2\n...\n- done\n\n"
        "gripper1:\n- subtask 1\n- subtask 2\n...\n- done\n"
    )

    user_prompt = f"""
        You will be provided with:
        1. A list of example subtasks
        2. A task query for each gripper
        3. A list of valid method names for detecting blocking conditions

        Your job is to break each query into subtasks, and if one subtask must wait for the other gripper to finish an action (e.g., placing a block, opening a drawer), then you must inject a line before that subtask:
        blocking_condition: method_name

        Guidelines:
        - Use only the method names from the provided list.
        - Choose the method name that corresponds to the condition described in the other gripper's task.
          For example, if the query says "wait until the black block is placed", then use: is_stacked_and_gripper_moved
        - Do not invent method names.
        - Inject one blocking_condition before every subtask that must wait.
        - Do not add blocking_condition unless explicitly needed.
        - Each subtask must start with a dash, and the list must end with "done"

        --- Example Subtask Patterns ---
        {subtask_examples}

        --- Method list ---
        {WAIT_METHOD_NAMES}

        --- Task Queries ---
        gripper query: {gripper_query}
        gripper1 query: {gripper1_query}

        Now generate the subtask lists:
    """

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.2,
        max_tokens=500
    )

    content = response['choices'][0]['message']['content']
    subtask_dict = {"gripper": [], "gripper1": []}
    current_key = None

    # Step 1: Parse the response into subtask dictionary
    for line in content.strip().splitlines():
        line = line.strip()
        if re.match(r'^gripper1?:', line):
            current_key = line.split(":")[0].strip()
        elif line.startswith("-") and current_key:
            subtask = line.lstrip("- ").strip()
            subtask_dict[current_key].append(subtask)
        elif line.startswith("blocking_condition:") and current_key:
            subtask_dict[current_key].append(line.strip())

    # Step 2: Ensure 'done' is the last subtask for each gripper
    for key in subtask_dict:
        if not subtask_dict[key] or subtask_dict[key][-1].lower() != "done":
            subtask_dict[key].append("done")

    print(f'gripper_queary = {gripper_query}')
    print(f'gripper1_queary = {gripper1_query}')
    print(f'subtask dictionary: {subtask_dict}')

    sub_gripper = subtask_dict["gripper"]
    sub_gripper1 = subtask_dict["gripper1"]

    env.log_explicit(f'sub-task plan for gripper: length: {len(sub_gripper)}\n{sub_gripper}')
    env.log_explicit(f'sub-task plan for gripper 1: length: {len(sub_gripper1)}\n{sub_gripper1}')


    return subtask_dict



##########################################################################################################
############################# Execute Sub Tasks ###########################################################
###########################################################################################################

import traceback
import re

def execute_subtasks(gripper_name, subtasks, gripper_ui, env, object_list, save_directory, task_name):
    """
    Executes a list of subtasks for a given gripper.

    Special handling:
    - If a subtask is 'blocking_condition:<method_name>' or 'blocking_condition:<method_name>:<target_object>'
    - If task_name == 'insert_rings', the method is called without target_object and its guessed_bar return
      replaces <guessed_bar> in remaining subtasks.
    - All other subtasks are executed via gripper_ui.
    """
    import re
    import traceback
    from datetime import datetime

    global description_history
    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"

    if task_name == "insert_rings":
            ring_pattern = re.compile(r"(blue|yellow) ring", re.IGNORECASE)
            first_task = subtasks[0].lower()
            match = ring_pattern.search(first_task)
            if match:
                ring_assignment = match.group(0)
            else:
                print(f"⚠️ No ring name found in first task of {gripper_name}: '{first_task}'")

    while subtasks:
        task = subtasks.pop(0)
        print(f"[{gripper_name}] Subtask: {task}")

        if task == "done":
            print(f"[{gripper_name}] ✅ Completed all subtasks.")
            break

        # Match both forms:
        # - blocking_condition:method
        # - blocking_condition:method:target_object
        match = re.match(r'^blocking_condition:\s*([a-zA-Z0-9_]+)(?::\s*(.+))?$', task)
        if match:
            method_name = match.group(1).strip()
            target_object = match.group(2).strip() if match.group(2) else None

            try:
                print(f"[{gripper_name}] ⏳ Waiting using method: {method_name}")
                method = globals().get(method_name)
                if method is None:
                    raise ValueError(f"Blocking method '{method_name}' is not defined in scope.")

                if task_name == "insert_rings":
                    # Method for insert_rings (no target_object)
                    guessed_bar, _= method(
                        gripper_name,
                        ring_assignment,
                        env=env,
                        object_list=object_list,
                        description_history=description_history,
                        save_dir=save_directory
                    )
                    print(f"[{gripper_name}] 🎯 Guessed bar: {guessed_bar}")
                    # Replace <guessed_bar> in remaining subtasks
                    subtasks = [s.replace("<guessed_bar>", guessed_bar) for s in subtasks]

                

                elif task_name == "shell_game":
                    # Method for insert_rings (no target_object)
                    guessed_cup, _ = method(
                    env=env,
                    guesser_name=gripper_name,
                    hider_name=other_gripper,
                    object_list=object_list
                    )
                    print(f"[{gripper_name}] 🧠 Guessed cup: {guessed_cup}")

                    # Replace <guessed_cup> in all remaining subtasks
                    subtasks = [
                        t.replace("<guessed_cup>", guessed_cup) for t in subtasks
                    ]
                
                elif task_name == "open_drawer_and_put_item":
                    # Method for insert_rings (no target_object)
                    method(
                        gripper_name,
                        env=env,
                        object_list=object_list,
                        description_history=description_history,
                        save_dir=save_directory
                    )
                    # print(f"[{gripper_name}] 🎯 Guessed bar: {guessed_bar}")
                    # # Replace <guessed_bar> in remaining subtasks
                    # subtasks = [s.replace("<guessed_bar>", guessed_bar) for s in subtasks]

                elif task_name == "open_lid_and_put_item":
                    # Method for insert_rings (no target_object)
                    method(
                        gripper_name,
                        env=env,
                        object_list=object_list,
                        description_history=description_history,
                        save_dir=save_directory
                    )
                    # print(f"[{gripper_name}] 🎯 Guessed bar: {guessed_bar}")
                    # # Replace <guessed_bar> in remaining subtasks
                    # subtasks = [s.replace("<guessed_bar>", guessed_bar) for s in subtasks]


                else:
                    # Method with target_object for general blocking conditions
                    if target_object is None:
                        raise ValueError(f"Blocking condition '{method_name}' requires a target object.")
                    method(
                        gripper_name,
                        target_object,
                        env=env,
                        object_list=object_list,
                        description_history=description_history,
                        save_dir=save_directory
                    )

            except Exception as e:
                print(f"[{gripper_name}] ❌ Error during blocking condition '{method_name}': {e}")
                traceback.print_exc()
                break

        else:
            try:
                print(f"[{gripper_name}] ▶️ Executing: {task}")

                # Handle repeat-until only for push_box tasks
                if (
                    task_name == 'push_box_to_target' and len(subtasks) == 1
                ):

                
                    repeat_count = 0
                    MAX_REPEAT = 20  # Set your safety limit here

                    while repeat_count < MAX_REPEAT:
                        print(f"Repeating the last subtask: {task}")
                        gripper_ui(task)
                        env.log_explicit(f"[{gripper_name}] Executed: {task}")
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        reasoning = f"{gripper_name} executed subtask: '{task}'"
                        description_history.append({
                            "timestamp": timestamp,
                            "reasoning": reasoning,
                        })   
                        repeat_count += 1

                        if is_box_on_target(env, object_list=object_list):
                            print("🎯 Box reached target!")
                            break
                    
                        print(f"🔄 Repeat push #{repeat_count}")

                        time.sleep(0.5)
                else:        
                    gripper_ui(task)
                    env.log_explicit(f"[{gripper_name}] Executed: {task}")
                    
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    reasoning = f"{gripper_name} executed subtask: '{task}'"
                    description_history.append({
                        "timestamp": timestamp,
                        "reasoning": reasoning,
                    })   
            except Exception as e:
                print(f"[{gripper_name}] ❌ Error executing '{task}': {e}")
                traceback.print_exc()
                break



def load_api_key(filepath='key.json'):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get('openai_api_key')
    except Exception as e:
        print("Error loading API key:", e)
        sys.exit(1)


def main(run_id, task_name: str):


    if task_name not in task_registry:
        print(f"❌ Unknown task '{task_name}'. Available tasks: {list(task_registry.keys())}")
        return

    # Load task-specific config
    task_cfg = task_registry[task_name]
    query = task_cfg["query"]
    object_list = task_cfg["object_list"]
    prompt_path = task_cfg["prompt_path"]
    subtask_prompt_path = task_cfg["subtask_prompt_path"]
    task_env_class = getattr(tasks, task_cfg["env_class"])


    # Save paths and env init (same as before)
    base_dir = f"/PRISM/data/{task_name}_prism"
    save_directory = os.path.join(base_dir, f"episode_{run_id}/front")
    recording_save_directory = os.path.join(base_dir, f"episode_{run_id}/recording")
    task_log = os.path.join(base_dir, f"episode_{run_id}/logs/")
    tmp_dir = os.path.join(base_dir, f"episode_{run_id}/tmp")


    for d in [save_directory, tmp_dir]:
        if os.path.exists(d): shutil.rmtree(d)
        os.makedirs(d)

    # Load openai key
    openai.api_key = load_api_key()


    config = get_config('rlbench')
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer, save_directory=recording_save_directory)
    env._lock = threading.Lock()
    env.image_counter = 1

    # start loggers
    env.start_terminal_log(task_log)  # or wherever you want the log
    env.set_explicit_log_dir(task_log)

    # initialize language model programs
    lmps, lmps_2, lmp_env = setup_LMP(env, config, debug=False)

    # load task environment
    env.load_task(task_env_class)
    descriptions, obs = env.reset()
    set_lmp_objects(lmps, env.get_object_names())
    set_lmp_objects(lmps_2, env.get_object_names())
    gripper_ui = lmps['composer_ui']
    gripper_ui_2 = lmps_2['composer_ui_2']

    
    # save initial observation
    initial_image = obs.front_rgb_0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(save_directory, f"front_{timestamp}.png")
    print(f'{image_path}')
    imageio.imwrite(image_path, initial_image)



    ###########################################
    # object grounding from initial observation using grounded-sam-2

    text_prompt = "left robot. right robot. " + ". ".join(object_list) + "."
    obs_dict = detect_objects(text_prompt=text_prompt, img_path=image_path, canonical_labels=object_list)

    # Remove 'score' key from each object
    for obj in obs_dict:
        obj.pop('score', None)
        # if obj.get('label') == 'black block':
        #     obj['label'] = 'stacking plane'

    global initial_obs_dict
    initial_obs_dict = obs_dict.copy()

    print(obs_dict)

    ############################################

    # load image file
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"
    
    # collaborative task plan
    plan_dict = get_plan_from_vlm(image_url, obs_dict, query, object_list)

    # increment vlm call
    env.increment_vlm_calls()

    # log collaborative plan
    env.log_explicit("start logging.")
    env.log_explicit(f'initial_observation_grounding:\n{obs_dict}')
    env.log_explicit(f'collaborative plan:\n{plan_dict}')
    print(plan_dict)


    # extract individula gripper plan
    gripper_plan = plan_dict.get('gripper')
    gripper1_plan = plan_dict.get('gripper1')


    # getting subtasks for collaborative plan 
    subtasks = get_subtask_response(gripper_plan, gripper1_plan)
    env.increment_vlm_calls()

    print(subtasks)
    
    # extract subtasks for each of the gripper
    sub_gripper = subtasks["gripper"]
    sub_gripper1 = subtasks["gripper1"]
    
    # log subtask information
    env.log_explicit(f'sub-task plan for gripper: length: {len(sub_gripper)}\n{sub_gripper}')
    env.log_explicit(f'sub-task plan for gripper 1: length: {len(sub_gripper1)}\n{sub_gripper1}')




    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(execute_subtasks, "gripper", sub_gripper.copy(), gripper_ui, env, object_list, tmp_dir),
            executor.submit(execute_subtasks, "gripper1", sub_gripper1.copy(), gripper_ui_2, env, object_list, tmp_dir)
        ]
        concurrent.futures.wait(futures)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("❌ Error: Missing required argument 'run_id'.")
        print("Usage: python shell_game_final.py <run_id>")
        sys.exit(1)

    try:
        run_id = int(sys.argv[1])
    except ValueError:
        print("❌ Error: run_id must be an integer.")
        sys.exit(1)

    task_name = "task_blocks"    
    main(run_id, task_name)
   