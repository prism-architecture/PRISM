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
import ast 



def get_image_base64(image_np):
    buffered = io.BytesIO()
    img = Image.fromarray(image_np)
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")


def save_description_history(save_dir, filename="description_history.json"):
    """Save description_history to JSON file."""
    filepath = os.path.join(save_dir, filename)
    try:
        with open(filepath, "w") as f:
            json.dump(description_history, f, indent=2)
        print(f"[wait] 💾 Saved description history to {filepath}")
    except Exception as e:
        print(f"[wait] ❌ Failed to save description history: {e}")




# is_stacked_and_gripper_moved

def is_stacked_and_gripper_moved(gripper_name, block_name, env, object_list, description_history, save_dir="/tmp"):
    """
    Waits until the other gripper has stacked a block and moved away,
    using front and top RGB views and CoT reasoning with memory.
    Also saves reasoning history to disk.
    """
    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"

    print(f"[{gripper_name}] 🕰️ Waiting for {other_gripper} to stack '{block_name}' and move away...")

    while True:
        with env._lock:
            obs = env.task.get_observation()
            front_img = obs.front_rgb_0
            top_img = obs.overhead_rgb_0

        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        front_path = os.path.join(save_dir, f"wait_front_{timestamp}.png")
        top_path = os.path.join(save_dir, f"wait_top_{timestamp}.png")
        imageio.imwrite(front_path, front_img)

        # Rotate top view and save
        top_pil = Image.fromarray(top_img)
        rotated_top_pil = top_pil.rotate(90, expand=True)
        rotated_top_pil.save(top_path)

        # Convert images to base64
        rotated_top_img_np = np.array(Image.open(top_path).convert('RGB'))
        front_b64 = get_image_base64(front_img)
        top_b64 = get_image_base64(rotated_top_img_np)

        # Construct improved prompt
        prompt = f"""
        You are observing a robot stacking task with two grippers from front and top RGB views.

        Gripper identities:
        - 'gripper1' is the left arm
        - 'gripper' is the right arm

        Task:
        - The robot '{other_gripper}' must stack the '{block_name}' on the stacking plane.
        - After stacking, it should move away (i.e., not occlude the stacking area in the top view).

        Images provided:
        - The first image is the **current front view**.
        - The second image is the **current top view** (rotated top-down view).

        Important strict visual rules:

        - If the '{block_name}' is still inside the fingers of '{other_gripper}' or visually attached to the gripper in any way (even if partly open), it must be considered "not yet placed" and "not stacked".
        - If the '{block_name}' is not resting stably and fully on the stacking plane surface, with no contact with any gripper, it must be considered "not yet stacked".
        - In the **top view**, if any part of '{other_gripper}' is overlapping the stacking plane area (green square), or if the '{block_name}' is not clearly visible as fully placed within the green square, then "safe_to_continue" must be false.
        - Do not assume stacking is complete based on prior history — base your judgment strictly on the current visual evidence.
        - If unsure about any of these, err on the side of caution and report:
            "stacked": false
            "gripper_moved": false
            "safe_to_continue": false

        Sequence of events you must check:

        - The '{other_gripper}' picked the '{block_name}'.
        - The '{other_gripper}' moved toward the stacking plane while holding the block.
        - The '{other_gripper}' placed the block on the stacking plane.
        - The '{other_gripper}' opened its grip and released the block.
        - The '{other_gripper}' moved away while not holding the block.

        Only if this entire sequence of events has clearly occurred (as supported by current and prior evidence), conclude that the block is stacked.
        If any step is not fully confirmed, do not mark the block as stacked.

        1. Carefully describe the **current front view scene**, using only the current front view image.
        2. Then, using both:
            - the current front view description you just wrote, and
            - the prior description history provided below,

        reason about whether the full stacking sequence has been completed.

        3. Only after stacking is confirmed, analyze the **current top view image** to decide whether it is safe to continue.

        4. Maintain step-by-step reasoning using temporal clues across frames.
           - Prior history should help guide reasoning, but it should not override what is clearly visible in the current frame.
           - If the current frame contradicts prior history, trust the current frame.

        --- Description History ---
        {chr(10).join(entry['reasoning'] for entry in description_history[-3:] if 'reasoning' in entry)}

        --- New Observations ---
        Describe and reason about:
        - **Front view (current observation only)**: robot actions and block placements
        - **Top view (current observation only)**: spatial configuration and gripper positions

        Respond in this format:

        Reasoning:
        <step-by-step reasoning using the above>

        Status (as a valid JSON object — no markdown block, no Python syntax):
        {{
            "stacked": true or false,
            "gripper_moved": true or false,
            "safe_to_continue": true or false
        }}
        """

        # Send to GPT
        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks using RGB views."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Front view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                    {"type": "text", "text": "Top view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"\n[wait] 🤖 GPT Reasoning:\n{content}\n")

            # Parse reasoning and status
            reasoning_start = content.find("Reasoning:")
            status_start = content.find("Status:")
            reasoning = content[reasoning_start + len("Reasoning:"):status_start].strip()
            raw_status = content[status_start + len("Status:"):].strip()
            env.increment_vlm_calls()

            # Clean raw_status to avoid markdown artifacts
            raw_status_clean = re.sub(r'```(?:python|json)?', '', raw_status).replace('```', '').strip()

            try:
                parsed_status = json.loads(raw_status_clean)

                # Save to structured description history
                description_history.append({
                    "timestamp": timestamp,
                    "gripper": gripper_name,
                    "block": block_name,
                    "reasoning": reasoning,
                    "raw_status": raw_status_clean,
                    "parsed_status": parsed_status
                })

                # Check parsed status
                if parsed_status.get("safe_to_continue"):
                    print(f"[{gripper_name}] ✅ Wait complete. Proceeding with execution.")
                    
                    # Auto-save description history
                    # save_description_history(save_dir)
                    
                    return

            except Exception as e_parse:
                print(f"[wait] ❌ Error parsing status dictionary: {e_parse}")
                print(f"[wait] Raw status was:\n{raw_status_clean}")
                traceback.print_exc()
                time.sleep(15)
                continue

        except Exception as e:
            print(f"[wait] ❌ GPT call error: {e}")
            traceback.print_exc()

        # Sleep before next iteration
        time.sleep(15)


# is_drawer_open_and_gripper_moved
def is_drawer_open_and_gripper_moved(gripper_name, env, object_list, description_history, save_dir="/tmp"):
    """
    Waits until the other gripper has opened the top drawer enough.
    Once the drawer is opened enough to place the cube, it is safe to continue,
    even if the gripper is still holding the handle or near the drawer.
    """
    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"

    print(f"[{gripper_name}] 🕰️ Waiting for {other_gripper} to open the top drawer enough to place the cube...")

    while True:
        with env._lock:
            obs = env.task.get_observation()
            front_img = obs.front_rgb_0
            top_img = obs.overhead_rgb_0

        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        front_path = os.path.join(save_dir, f"wait_front_{timestamp}.png")
        top_path = os.path.join(save_dir, f"wait_top_{timestamp}.png")
        imageio.imwrite(front_path, front_img)

        # Rotate top view and save
        top_pil = Image.fromarray(top_img)
        rotated_top_pil = top_pil.rotate(90, expand=True)
        rotated_top_pil.save(top_path)

        rotated_top_img_np = np.array(Image.open(top_path).convert('RGB'))
        front_b64 = get_image_base64(front_img)
        top_b64 = get_image_base64(rotated_top_img_np)

        prompt = f"""
        You are observing a robot task with two grippers from front and top RGB views.

        Gripper identities:
        - 'gripper1' is the left arm
        - 'gripper' is the right arm

        Task:
        - The robot '{other_gripper}' must open the top drawer.
        - The other robot '{gripper_name}' is waiting to place the red cube inside the drawer.

        IMPORTANT:
        - You do NOT need to check for the drawer being fully open.
        - It is sufficient if the drawer appears opened **enough** so that a cube can be placed inside comfortably.
        - It is okay if '{other_gripper}' is still holding the handle or is near the drawer — that does not block the other gripper from proceeding.

        Decision rules:

        - If the drawer is clearly closed or barely opened → 'drawer_open' = false, 'safe_to_continue' = false.
        - If the drawer is visibly opened enough for a cube to be placed → 'drawer_open' = true, 'safe_to_continue' = true.
        - 'gripper_moved' is still reported but is not required to be true for 'safe_to_continue' to be true.

        --- Description History ---
        {chr(10).join(entry['reasoning'] for entry in description_history[-3:] if 'reasoning' in entry)}

        --- New Observations ---
        Describe and reason about:
        - **Front view (current observation only)**: drawer state and gripper actions.
        - **Top view (current observation only)**: drawer position, gripper positions.

        Respond strictly in this format:

        Reasoning:
        <step-by-step reasoning using the above>

        Status (as a valid JSON object — no markdown block, no Python syntax):
        {{
            "drawer_open": true or false,
            "gripper_moved": true or false,
            "safe_to_continue": true or false
        }}
        """

        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks using RGB views."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Front view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                    {"type": "text", "text": "Top view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"\n[wait] 🤖 GPT Reasoning:\n{content}\n")

            reasoning_start = content.find("Reasoning:")
            status_start = content.find("Status:")
            reasoning = content[reasoning_start + len("Reasoning:"):status_start].strip()
            raw_status = content[status_start + len("Status:"):].strip()

            raw_status_clean = re.sub(r'```(?:python|json)?', '', raw_status).replace('```', '').strip()
            env.increment_vlm_calls()
            parsed_status = json.loads(raw_status_clean)

            description_history.append({
                "timestamp": timestamp,
                "gripper": gripper_name,
                "reasoning": reasoning,
                "raw_status": raw_status_clean,
                "parsed_status": parsed_status
            })

            # New rule: safe_to_continue is simply equal to drawer_open
            if parsed_status.get("drawer_open"):
                print(f"[{gripper_name}] ✅ Wait complete. Drawer is sufficiently open, safe to place the cube.")
                save_description_history(save_dir)
                return

        except Exception as e:
            print(f"[wait] ❌ GPT call error: {e}")
            traceback.print_exc()

        time.sleep(15)


# is_lid_open_and_gripper_moved

def is_lid_open_and_gripper_moved(gripper_name, env, object_list, description_history, save_dir="/tmp"):
    """
    Waits until the other gripper has opened the lid and moved away,
    using front and top RGB views and CoT reasoning with memory.
    Also saves reasoning history to disk.
    """
    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"

    print(f"[{gripper_name}] 🕰️ Waiting for {other_gripper} to open the lid and move away...")

    while True:
        with env._lock:
            obs = env.task.get_observation()
            front_img = obs.front_rgb_0
            top_img = obs.overhead_rgb_0

        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        front_path = os.path.join(save_dir, f"wait_front_{timestamp}.png")
        top_path = os.path.join(save_dir, f"wait_top_{timestamp}.png")
        imageio.imwrite(front_path, front_img)

        # Rotate top view and save
        top_pil = Image.fromarray(top_img)
        rotated_top_pil = top_pil.rotate(90, expand=True)
        rotated_top_pil.save(top_path)

        # Convert images to base64
        rotated_top_img_np = np.array(Image.open(top_path).convert('RGB'))
        front_b64 = get_image_base64(front_img)
        top_b64 = get_image_base64(rotated_top_img_np)

        # Construct new prompt
        prompt = f"""
        You are observing a robot task with two grippers from front and top RGB views.

        Gripper identities:
        - 'gripper1' is the left arm
        - 'gripper' is the right arm

        Task:
        - The robot '{other_gripper}' must open the lid of the saucepan.
        - After opening the lid, it should move away (i.e., not occlude the saucepan area).

        The robot '{gripper_name}' will wait until it is safe to place the red cube inside the saucepan.

        Your goal is to analyze the current images and reason step-by-step:

        Rules:

        - If the saucepan lid is still visibly covering the saucepan (even partially), then 'lid_open' = false.
        - If the lid is clearly moved away and the opening of the saucepan is fully visible, then 'lid_open' = true.
        - In the **top view**, if any part of '{other_gripper}' is overlapping or hovering above the saucepan area, or if the lid is still covering the saucepan, then 'safe_to_continue' must be false.
        - The lid must be open and the gripper must have moved away to mark 'safe_to_continue' = true.

        - You must not assume the lid is open based on prior history — base your judgment strictly on the current visual evidence.
        - If unsure about any of these, err on the side of caution:
            {{
                "lid_open": false,
                "gripper_moved": false,
                "safe_to_continue": false
            }}

        Sequence of events to check:

        - The '{other_gripper}' grasps the lid.
        - The '{other_gripper}' lifts and moves the lid away from the saucepan.
        - The opening of the saucepan becomes fully visible.
        - The '{other_gripper}' moves away, not occluding the saucepan.

        Use both front and top views carefully:

        - In the front view, look for whether the lid is still covering the saucepan.
        - In the top view, check whether the lid is moved away and whether '{other_gripper}' is out of the way.

        --- Description History ---
        {chr(10).join(entry['reasoning'] for entry in description_history[-3:] if 'reasoning' in entry)}

        --- New Observations ---
        Describe and reason about:
        - **Front view (current observation only)**: lid state and gripper actions
        - **Top view (current observation only)**: lid position, gripper positions, saucepan visibility

        Respond strictly in this format:

        Reasoning:
        <step-by-step reasoning using the above>

        Status (as a valid JSON object — no markdown block, no Python syntax):
        {{
            "lid_open": true or false,
            "gripper_moved": true or false,
            "safe_to_continue": true or false
        }}
        """

        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks using RGB views."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Front view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                    {"type": "text", "text": "Top view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=800,
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"\n[wait] 🤖 GPT Reasoning:\n{content}\n")

            reasoning_start = content.find("Reasoning:")
            status_start = content.find("Status:")
            reasoning = content[reasoning_start + len("Reasoning:"):status_start].strip()
            raw_status = content[status_start + len("Status:"):].strip()

            env.increment_vlm_calls()

            raw_status_clean = re.sub(r'```.*?```', '', raw_status, flags=re.DOTALL).strip()

            try:
                parsed_status = json.loads(raw_status_clean)

                description_history.append({
                    "timestamp": timestamp,
                    "gripper": gripper_name,
                    "reasoning": reasoning,
                    "raw_status": raw_status_clean,
                    "parsed_status": parsed_status
                })

                if parsed_status.get("safe_to_continue"):
                    print(f"[{gripper_name}] ✅ Wait complete. Lid is open and it is safe to place the red cube.")
                    # save_description_history(save_dir)
                    return

            except Exception as e_parse:
                print(f"[wait] ❌ Error parsing status dictionary: {e_parse}")
                print(f"[wait] Raw status was:\n{raw_status_clean}")
                traceback.print_exc()
                time.sleep(15)
                continue

        except Exception as e:
            print(f"[wait] ❌ GPT call error: {e}")
            traceback.print_exc()

        time.sleep(15)

# is_ring_inserted_and_gripper_moved

def is_ring_inserted_and_gripper_moved(gripper_name, ring_name, env, object_list, description_history, save_dir="/tmp"):
    """
    Robust wait-and-guess that uses step-by-step temporal CoT to infer ring insertion.
    Returns guessed bar name and description history once it's safe to continue.
    """

    other_gripper = "gripper1" if gripper_name == "gripper" else "gripper"
    other_ring_name = "yellow ring" if ring_name == "blue ring" else "blue ring"

    print(f"[{gripper_name}] 🕰️ Waiting for {other_gripper} to insert '{other_ring_name}' and move away...")

    move_away_confirmations = 0

    while True:
        with env._lock:
            obs = env.task.get_observation()
            front_img = obs.front_rgb_0
            top_img = obs.overhead_rgb_0

        # Save images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        front_path = os.path.join(save_dir, f"wait_front_{timestamp}.png")
        top_path = os.path.join(save_dir, f"wait_top_{timestamp}.png")
        imageio.imwrite(front_path, front_img)

        # Rotate top view and save
        top_pil = Image.fromarray(top_img)
        rotated_top_pil = top_pil.rotate(90, expand=True)
        rotated_top_pil.save(top_path)
        rotated_top_img_np = np.array(rotated_top_pil.convert('RGB'))

        # Base64 encodings
        front_b64 = get_image_base64(front_img)
        top_b64 = get_image_base64(rotated_top_img_np)

        prompt = f"""
        You are observing a robot ring-insertion task using front and top RGB views.

        Gripper identities:
        - 'gripper1' is the left arm
        - 'gripper' is the right arm

        Task:
        - The robot '{other_gripper}' must insert the '{other_ring_name}' into one of the bars (e.g., red, green, black).
        - The current gripper '{gripper_name}' must wait, infer which bar was used, and only proceed when it's safe.

        Requirements:
        - Your reasoning must be gradual and consistent across frames. Do not jump to conclusions in a single frame.
        - For each frame, observe only one small part of the sequence and reason accordingly.

        Step-by-step confirmation of insertion:
        1. Observe in front view whether '{other_gripper}' picks the '{other_ring_name}'.
        2. Then check whether it moves toward a bar.
        3. Then check whether it releases the ring and it is no longer in the gripper.
        4. Confirm that the ring appears inserted in the bar and not in the hand.
        5. Confirm the gripper has visibly moved far away from the bar (not blocking or hovering above).
        6. Then use **top view**, and across **multiple frames**, confirm:
           - '{other_gripper}' is repeatedly far from the bar.
           - The bar and ring are clearly visible.
           - The gripper appears at the table edge.

        Your job:
        - Describe the current front view carefully.
        - Then reason about the ring insertion status using the current observation and the prior description history.
        - Only after `insertion_complete = true`, analyze the top view to confirm whether `gripper_moved = true`.
        - Only after both are true, return `safe_to_continue = true`.

        If anything is uncertain, err on the side of caution and return:
        {{
            "insertion_complete": false,
            "gripper_moved": false,
            "safe_to_continue": false,
            "suspected_bar": null
        }}

        --- Description History ---
        {chr(10).join(entry['reasoning'] for entry in description_history[-3:] if 'reasoning' in entry)}

        --- New Observations ---
        Describe and reason about:
        - **Front view (current only)**: {other_gripper} behavior, ring state, bar interaction.
        - **Top view (only after insertion confirmed)**: bar visibility, ring position, gripper distance.

        Respond strictly in this format:

        Reasoning:
        <your reasoning here>

        Status (valid JSON, no code block):
        {{
            "insertion_complete": true or false,
            "gripper_moved": true or false,
            "safe_to_continue": true or false,
            "suspected_bar": "red" | "green" | "black" | null
        }}
        """

        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks using RGB views."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "text", "text": "Front view RGB image:"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                    {"type": "text", "text": "Top view RGB image (rotated):"},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.0,
                max_tokens=900,
            )

            content = response['choices'][0]['message']['content'].strip()
            print(f"\n[wait_and_guess] 🤖 GPT Reasoning:\n{content}\n")

            reasoning_start = content.find("Reasoning:")
            status_start = content.find("Status:")
            reasoning = content[reasoning_start + len("Reasoning:"):status_start].strip()
            raw_status = content[status_start + len("Status:"):].strip()
            raw_status_clean = re.sub(r'```(?:python|json)?', '', raw_status).replace('```', '').strip()


            env.increment_vlm_calls()

            parsed_status = json.loads(raw_status_clean)



            # Save reasoning step
            description_history.append({
                "timestamp": timestamp,
                "gripper": gripper_name,
                "ring": ring_name,
                "reasoning": reasoning,
                "raw_status": raw_status_clean,
                "parsed_status": parsed_status
            })

            # Update multi-frame confirmation
            if parsed_status.get("insertion_complete") and parsed_status.get("gripper_moved"):
                move_away_confirmations += 1
            else:
                move_away_confirmations = 0  # reset if broken

            if parsed_status.get("insertion_complete") and move_away_confirmations >= 4:
                if parsed_status.get("safe_to_continue") and parsed_status.get("suspected_bar") in {"red", "green", "black"}:
                    print(f"[{gripper_name}] ✅ Safe to continue. Guessed bar: {parsed_status['suspected_bar']}")
                    return parsed_status["suspected_bar"], description_history

        except Exception as e:
            print(f"[wait_and_guess] ❌ Error during GPT call or parsing: {e}")
            traceback.print_exc()

        time.sleep(15)


# is_box_on_target
def is_box_on_target(env, object_list, save_dir="/tmp"):
    """
    Uses GPT-4o to strictly determine whether the box has fully covered the red target.
    Success only if **no part** of the red target is visible in the top view.
    """
    import base64
    import imageio
    import os
    import numpy as np
    import json
    import ast
    import re
    import traceback
    from PIL import Image
    from datetime import datetime

    with env._lock:
        obs = env.task.get_observation()
        front_img = obs.front_rgb_0
        top_img = obs.overhead_rgb_0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    front_path = os.path.join(save_dir, f"check_front_{timestamp}.png")
    top_path = os.path.join(save_dir, f"check_top_{timestamp}_rotated.png")

    imageio.imwrite(front_path, front_img)
    top_pil = Image.fromarray(top_img).rotate(90, expand=True)
    top_pil.save(top_path)
    rotated_top_img_np = np.array(top_pil.convert('RGB'))

    front_b64 = get_image_base64(front_img)
    top_b64 = get_image_base64(rotated_top_img_np)

    prompt = """
You are a visual inspector for a robotic task involving a box and a red square target.

Determine if the box has **fully reached and covered the red target** using these strict criteria:

1. **Top View (rotated)**:
   - The red square must be **completely covered** by the box.
   - If **any part** of the red area is visible, the box has **not** reached the target.

2. **Front View**:
   - Should show the box aligned or directly in front of the red area.
   - But the top view is the primary decision factor.

Be very strict. Do **not** infer or estimate coverage. Use only visual confirmation.

Return only valid JSON with lowercase booleans:
{
  "reasoning": "explain your conclusion based on both views",
  "reached": true or false
}
"""

    messages = [
        {"role": "system", "content": "You are a visual reasoning expert for tabletop robot inspection tasks."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "text", "text": "Front view RGB image:"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                {"type": "text", "text": "Top view RGB image (rotated):"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
            ]
        }
    ]

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.0,
            max_tokens=500,
        )
        content = response['choices'][0]['message']['content']
        print(f"\n[is_box_on_target] 🧠 GPT Reasoning:\n{content}\n")
        env.increment_vlm_calls()

        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            raw_dict_str = match.group(0)
            try:
                return json.loads(raw_dict_str).get("reached", False)
            except json.JSONDecodeError:
                fixed = raw_dict_str.replace("true", "True").replace("false", "False")
                return ast.literal_eval(fixed).get("reached", False)
    except Exception as e:
        print(f"[is_box_on_target] ❌ Error during GPT check: {e}")
        traceback.print_exc()

    return False


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

def is_handover_pose_done(object_name, other_gripper, env, save_dir):
    wait_until_condition(WAIT_DONE_PROMPT, object_name, other_gripper, env, save_dir, required_repetitions=0)

def is_grasped(object_name, other_gripper, env, save_dir):
    wait_until_condition(WAIT_GRASP_PROMPT, object_name, other_gripper, env, save_dir)

def is_gripper_opened(other_gripper, env, save_dir):
    wait_until_condition(WAIT_OPEN_PROMPT, object_name=None, other_gripper=other_gripper, env=env, save_dir=save_dir)


# wait_and_guess_the_cup
def wait_and_guess_the_cup(env, guesser_name, hider_name, object_list):

    """
    Multi-frame vision reasoning to determine which cup hides the green block
    and whether it's safe to pick it.

    Returns:
        guessed_cup (str), history (list of reasoning + state entries)
    """
    import time
    import base64
    import io
    import ast
    from PIL import Image
    import openai

    def get_image_base64(image_np):
        buffered = io.BytesIO()
        img = Image.fromarray(image_np)
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def sanitize_state_block(text):
        text = text.replace("‘", "'").replace("’", "'").replace("“", "\"").replace("”", "\"")
        if text.startswith("```"):
            text = text.split("```")[-1].strip()
        return text.strip()

    def validate_state_against_reasoning(reasoning, state):
        reason = reasoning.lower()
        if "green block is visible" in reason and state["green_block_occluded"]:
            print("[⚠️] Inconsistency: Block is visible but marked as occluded.")
            return False
        if "green block is hidden" in reason and not state["green_block_occluded"]:
            print("[⚠️] Inconsistency: Block is hidden but state says it's not occluded.")
            return False
        return True

    history = []
    state = {
        "suspected_cup": None,
        "green_block_occluded": False,
        "hider_grasped_cup": False,
        "hider_moved_away": False,
        "safe_to_pick": False
    }
    suspected_cup_locked = False

    print("[wait_and_guess] Starting visual reasoning with temporal history...")
    start_time = time.time()

    while True:
        if time.time() - start_time > 900:
            sys.exit("[wait_and_guess] ⏰ Timeout reached (6 minutes). Episode failed.")
        
        with env._lock:
            obs = env.task.get_observation()
            front_img = obs.front_rgb_0
            top_img = obs.overhead_rgb_0

        front_b64 = get_image_base64(front_img)
        top_b64 = get_image_base64(top_img)

        # Format history
        reasoning_history = "\n\n".join(
            f"Observation {i+1}:\n{entry['reasoning']}\nState: {entry['state']}"
            for i, entry in enumerate(history)
        ) or "No prior reasoning yet."

        prompt = f"""
            You are reasoning over a robot manipulation task using RGB images (top + front).

            Roles:
            - '{hider_name}' hides a green block under a cup (red/blue/black).
            - '{guesser_name}' must guess the cup, but only when it's safe.

            Use these rules:
            - The green block is occluded only after:
                1. The hider robot grasps a cup
                2. Moves it over the green block
                3. Places it, hiding the block from view
            - Once `green_block_occluded` is True, it stays True
            - Once the hider grasps a cup, lock that as the `suspected_cup`
            - If reasoning says "green block is visible", then `green_block_occluded` must be False
            - After oberving steps 1, 2, and 3, if the suspected_cup is clearly visible from top view that is         
            the hider moved away from the top of the suspected_cup the `hider_moved_away` should be true
            - Make sure hider is far away from the suspected_cup. Look into the top view of the image to determine
            that hider is far away
            - Once `hider_moved_away` is True, it stays True
            - Your structured state must always match your reasoning

            Current object list:
            {object_list}

            Previous state:
            {state}

            History:
            {reasoning_history}

            Now reason about this new image observation.

            Return exactly:
            Reasoning:
            <your reasoning>

            State:
            {{
            'suspected_cup': 'red' | 'blue' | 'black' | None,
            'green_block_occluded': True | False,
            'hider_grasped_cup': True | False,
            'hider_moved_away': True | False,
            'safe_to_pick': True | False
            }}
        """

        messages = [
            {"role": "system", "content": "You are a visual reasoning expert for robot manipulation tasks."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},

                    # 🔽 Add this clear labeling block
                    {"type": "text", "text": "The following are two RGB camera images:\n1. Front view (robot eye-level)\n2. Top view (overhead view of the workspace)"},

                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{front_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{top_b64}"}},
                ]
            }
        ]

        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=300,
            )

            content = response['choices'][0]['message']['content'].strip()
            print("\n[wait_and_guess] 🤖 GPT Reasoning + State:\n" + content + "\n")

            reasoning_start = content.find("Reasoning:")
            state_start = content.find("State:")

            reasoning = content[reasoning_start + len("Reasoning:"):state_start].strip()
            raw_state = content[state_start + len("State:"):].strip()
            parsed_state = ast.literal_eval(sanitize_state_block(raw_state))

            env.increment_vlm_calls()

            # 🧠 Sanity Check
            if not validate_state_against_reasoning(reasoning, parsed_state):
                print("[wait_and_guess] ❌ Skipping due to mismatch between reasoning and state.")
                time.sleep(1.5)
                continue

            # === Irreversible logic ===
            if parsed_state["green_block_occluded"]:
                state["green_block_occluded"] = True
            if parsed_state["hider_moved_away"]:
                state["hider_moved_away"] = True

            state["hider_grasped_cup"] = parsed_state["hider_grasped_cup"]

            # === Cup locking logic ===
            if not suspected_cup_locked and parsed_state["hider_grasped_cup"]:
                if parsed_state["suspected_cup"] in {"red", "blue", "black"}:
                    state["suspected_cup"] = parsed_state["suspected_cup"]
                    suspected_cup_locked = True
            elif not suspected_cup_locked and not state["green_block_occluded"]:
                if parsed_state["suspected_cup"] in {"red", "blue", "black"}:
                    state["suspected_cup"] = parsed_state["suspected_cup"]

            state["safe_to_pick"] = state["green_block_occluded"] and state["hider_moved_away"]

            history.append({
                "reasoning": reasoning,
                "state": state.copy()
            })

            if state["safe_to_pick"] and state["suspected_cup"]:
                print(f"[wait_and_guess] ✅ Safe to pick: {state['suspected_cup']}")
                return state["suspected_cup"], history

        except Exception as e:
            print(f"[wait_and_guess] ⚠️ GPT error: {e}")

        time.sleep(1.5)

