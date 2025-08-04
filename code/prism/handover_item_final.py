# === Imports ===
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
import sys
import json
import shutil

import warnings
warnings.simplefilter("error", RuntimeWarning)

# === Utility functions ===

def get_image_base64(image_np):
    buffered = io.BytesIO()
    img = Image.fromarray(image_np)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# === get_handover_plan_from_vlm ===

def get_handover_plan_from_vlm(image_url, object_name, query):
    visual_prompt = (
        "You are analyzing a tabletop scene with two robotic arms: 'gripper' and 'gripper1'. "
        "The goal is to perform a horizontal handover of a single object (e.g., green block). "
        "The image provided is a front-view RGB image.\n\n"
        "Rules:\n"
        "- 'gripper1' will ALWAYS act as the giver.\n"
        "- 'gripper' will ALWAYS act as the receiver.\n"
        "You must reason about:\n"
        "- The giver gripper1 should pick and move the object to a handover pose\n"
        "- The receiver gripper should wait and grasp the object from that pose and move it away\n"
        "- Use the horizontal positions in the front view to confirm the expected roles.\n\n"
        "Output natural language high-level plans. Each robot gets one line, starting with its name."
    )

    plan_prompt = f"""
        You are coordinating two robotic arms (gripper and gripper1) to perform a horizontal handover task.

        Task:
        - The object to be handed over is: {object_name}.
        - 'gripper1' will ALWAYS be the giver.
        - 'gripper' will ALWAYS be the receiver.
        - gripper1 will pick and move the object to a horizontal handover pose.
        - gripper will wait and grasp the object from that pose and move it away.

        The actions must be coordinated using wait signals where necessary:
            - The receiver gripper will wait for the giver to complete the handover pose and stop.
            - The receiver will then grasp the object, and wait for the giver to open its gripper before moving away.

        Role phrasing required:
        - The receiver gripper MUST use this phrasing:
            gripper plan: wait and grasp the {object_name} from gripper1
        - The giver gripper MUST use this phrasing:
            gripper1 plan: pick the {object_name} and handover to gripper

        Instructions:
        1. Follow the required phrasing exactly for the plan.
        2. Output only the two plan lines as described.

        Output Format:
        gripper: [natural language plan]
        gripper1: [natural language plan]

        Example:
        gripper: wait and grasp the green block from gripper1
        gripper1: pick the green block and handover to gripper

        Now generate the high-level robot plans for this task: {query}
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert in collaborative robot task planning. "
                    "You will coordinate a horizontal handover of an object between two robot arms."
                )
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": visual_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
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
    for line in response_text.strip().splitlines():
        line = line.strip()
        if line.startswith("gripper:") or line.startswith("gripper1:"):
            parts = line.split(":", 1)
            if len(parts) == 2:
                robot = parts[0].strip()
                plan = parts[1].strip()
                plan_dict[robot] = plan

    return plan_dict

# === Get Subtask Response ===

def get_subtask_response(gripper_query, gripper1_query, prompt_path='sub_task_prompt.txt'):
    with open(prompt_path, 'r') as f:
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

        Use the examples to understand subtask structure and logic. Then, generate a complete list of subtasks for each gripper.

        Output rules:
        - No explanation or extra text.
        - Each subtask must start with a dash.
        - Each list must end with: done.

        --- Example Subtask Patterns ---
        {subtask_examples}

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

    for line in content.strip().splitlines():
        line = line.strip()
        if re.match(r'^gripper1?:', line):
            current_key = line.split(":")[0].strip()
        elif line.startswith("-") and current_key:
            subtask = line.lstrip("- ").strip()
            subtask_dict[current_key].append(subtask)

    for key in subtask_dict:
        if not subtask_dict[key] or subtask_dict[key][-1].lower() != "done":
            subtask_dict[key].append("done")

    print(f'gripper1_query ={gripper1_query}')
    print(f'gripper_query ={gripper_query}')
    print(f'subtask dictionary: {subtask_dict}')

    return subtask_dict

# === Wait CoT-based functions ===

WAIT_DONE_PROMPT = """
You are observing a horizontal handover task with two grippers.

Task:
- The '{other_gripper}' must move the '{object_name}' to the handover pose:
    - Above the stacking area
    - Held horizontally (90 degree rotation along X)
    - Gripper has stopped moving

Instruction:
- Based on the front view image, decide if this condition is fully met.

Respond strictly in this format:

Reasoning:
<your reasoning>

Status:
{{ "condition_met": true or false }}
"""

WAIT_GRASP_PROMPT = """
You are observing a horizontal handover task.

Task:
- The '{other_gripper}' must grasp the '{object_name}'.

Instruction:
- Based on the front view image, decide if the gripper has securely grasped the '{object_name}'.

Respond strictly in this format:

Reasoning:
<your reasoning>

Status:
{{ "condition_met": true or false }}
"""

WAIT_OPEN_PROMPT = """
You are observing a horizontal handover task.

Task:
- The '{other_gripper}' must open its gripper.

Instruction:
- Based on the front view image, decide if the '{other_gripper}' has opened its gripper.

Respond strictly in this format:

Reasoning:
<your reasoning>

Status:
{{ "condition_met": true or false }}
"""

def front_image_to_base64(env):
    with env._lock:
        obs = env.task.get_observation()
        front_img = obs.front_rgb_0
    return get_image_base64(front_img)

def wait_until_condition(prompt_template, object_name, other_gripper, env, save_dir, required_repetitions=3):
    consecutive_success = 0
    while True:
        front_b64 = front_image_to_base64(env)

        prompt = prompt_template.format(
            object_name=object_name,
            other_gripper=other_gripper
        )

        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Front view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=300,
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"\n[WAIT] 🤖 GPT Reasoning:\n{content}\n")

            status_start = content.find("Status:")
            raw_status = content[status_start + len("Status:"):].strip()
            raw_status_clean = re.sub(r'```(?:python|json)?', '', raw_status).replace('```', '').strip()
            parsed_status = json.loads(raw_status_clean)

            env.increment_vlm_calls()

            if parsed_status.get("condition_met"):
                consecutive_success += 1
                print(f"[WAIT] ✅ Success count: {consecutive_success}/{required_repetitions}")
                if consecutive_success >= required_repetitions:
                    print(f"[WAIT] 🎉 Condition confirmed. Proceeding.")
                    return
            else:
                consecutive_success = 0

        except Exception as e:
            print(f"[WAIT] ❌ GPT error: {e}")
            traceback.print_exc()

        time.sleep(3)

def wait_until_done_with(object_name, other_gripper, env, save_dir):
    wait_until_condition(WAIT_DONE_PROMPT, object_name, other_gripper, env, save_dir, required_repetitions=0)

def wait_until_grasp(object_name, other_gripper, env, save_dir):
    wait_until_condition(WAIT_GRASP_PROMPT, object_name, other_gripper, env, save_dir)

def wait_until_open_gripper(other_gripper, env, save_dir):
    wait_until_condition(WAIT_OPEN_PROMPT, object_name=None, other_gripper=other_gripper, env=env, save_dir=save_dir)

# === Execute Subtasks ===

def execute_subtasks(gripper_name, subtasks, gripper_ui, env, object_name, save_directory):
    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"

    while subtasks:
        task = subtasks.pop(0)
        print(f"[{gripper_name}] Subtask: {task}")

        if task == "done":
            print(f"[{gripper_name}] ✅ Completed all subtasks.")
            break

        match_done = re.match(r'^wait_until_done:\s*(.+)$', task)
        if match_done:
            wait_for_gripper = match_done.group(1).strip()
            print(f"[{gripper_name}] ⏳ Waiting for {wait_for_gripper} to finish handover...")
            wait_until_done_with(object_name, wait_for_gripper, env, save_directory)

        elif task.startswith("wait_until_grasp:"):
            match_grasp = re.match(r'^wait_until_grasp:\s*(.+)$', task)
            wait_for_gripper = match_grasp.group(1).strip()
            print(f"[{gripper_name}] ⏳ Waiting for {wait_for_gripper} to grasp...")
            wait_until_grasp(object_name, wait_for_gripper, env, save_directory)

        elif task.startswith("wait_until_open_gripper:"):
            match_open = re.match(r'^wait_until_open_gripper:\s*(.+)$', task)
            wait_for_gripper = match_open.group(1).strip()
            print(f"[{gripper_name}] ⏳ Waiting for {wait_for_gripper} to open gripper...")
            wait_until_open_gripper(wait_for_gripper, env, save_directory)

        else:
            try:
                print(f"[{gripper_name}] ▶️ Executing: {task}")
                gripper_ui(task)
                env.log_explicit(f"[{gripper_name}] Executed: {task}")
            except Exception as e:
                print(f"[{gripper_name}] ❌ Error executing '{task}': {e}")
                traceback.print_exc()
                break

# === Load API Key ===

def load_api_key(filepath='key.json'):
    try:
        with open(filepath, 'r') as file:
            data = json.load(file)
            return data.get('openai_api_key')
    except Exception as e:
        print("Error loading API key:", e)
        sys.exit(1)


# Example main loop
def main(run_id):
    openai.api_key = load_api_key()

    base_dir = "/PRISM/data/handover_item"
    save_directory = os.path.join(base_dir, f"episode_{run_id}/front")
    recording_save_directory = os.path.join(base_dir, f"episode_{run_id}/recording")
    base_dir_2 = os.path.join(base_dir, f"episode_{run_id}/tmp")

    task_log = os.path.join(base_dir, f"episode_{run_id}/logs/")

    if os.path.exists(save_directory):
        shutil.rmtree(save_directory)

    if os.path.exists(base_dir_2):
        shutil.rmtree(base_dir_2)

    os.makedirs(save_directory)
    os.makedirs(base_dir_2)

    config = get_config('rlbench')
    visualizer = ValueMapVisualizer(config['visualizer'])
    env = VoxPoserRLBench(visualizer=visualizer, save_directory=recording_save_directory)
    env._lock = threading.Lock()
    env.image_counter = 1

    # initiate logger
    
    env.start_terminal_log(task_log)  # or wherever you want the log
    env.set_explicit_log_dir(task_log)

    lmps, lmps_2, lmp_env = setup_LMP(env, config, debug=False)
    env.load_task(tasks.StackBlocksV3)
    descriptions, obs = env.reset()

    initial_image = obs.front_rgb_0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    image_path = os.path.join(save_directory, f"front_{timestamp}.png")
    imageio.imwrite(image_path, initial_image)

    object_name = "yellow block"
    query = "Handover the yellow block"

    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    image_url = f"data:image/png;base64,{base64_image}"

    plan_dict = get_handover_plan_from_vlm(image_url, object_name, query)

    print(plan_dict)

    env.increment_vlm_calls()


    env.log_explicit("start logging.")
    # env.log_explicit(f'initial_observation_grounding:\n{obs_dict}')
    env.log_explicit(f'collaborative plan:\n{plan_dict}')

    gripper_plan = plan_dict.get('gripper')
    gripper1_plan = plan_dict.get('gripper1')

    subtasks = get_subtask_response(gripper_plan, gripper1_plan)

    env.increment_vlm_calls()


    sub_gripper = subtasks["gripper"]
    sub_gripper1 = subtasks["gripper1"]

    env.log_explicit(f'sub-task plan for gripper: length: {len(sub_gripper)}\n{sub_gripper}')
    env.log_explicit(f'sub-task plan for gripper 1: length: {len(sub_gripper1)}\n{sub_gripper1}')


    set_lmp_objects(lmps, env.get_object_names())
    set_lmp_objects(lmps_2, env.get_object_names())

    gripper_ui = lmps['composer_ui']
    gripper_ui_2 = lmps_2['composer_ui_2']

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(execute_subtasks, "gripper", sub_gripper.copy(), gripper_ui, env, object_name, save_directory),
            executor.submit(execute_subtasks, "gripper1", sub_gripper1.copy(), gripper_ui_2, env, object_name, save_directory)
        ]
        concurrent.futures.wait(futures)

if __name__ == "__main__":
    main(10)
   
