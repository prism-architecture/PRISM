prompt_dir = "collab_planning_prompts_prism"

subtask_prompt_path = "sub_task_examples.txt"
collab_plan_prompt_path = "collab_examples.txt"


task_registry = {
    "stack_blocks": {
        "query": "stack the red block on top of the yellow block",
        "object_list": ["red block", "yellow block", "stacking plane"],
        "env_class": "StackBlocks",
        "prompt_yaml": f"{prompt_dir}/stack_blocks.yaml",
    },
    "pyramid_stacking": {
        "query": "stack two blocks",
        "object_list": ["red block", "black block", "stacking plane", "blue block", "yellow block"],
        "env_class": "StackBlocksV3",
        "prompt_yaml": f"{prompt_dir}/pyramid_stacking.yaml",
    },
    "insert_rings": {
        "query": "insert the red and blue rings into the square bars without collision",
        "object_list": ["red bar", "black bar", "green bar", "blue ring", "yellow ring"],
        "env_class": "InsertOntoSquarePeg",
        "prompt_yaml": f"{prompt_dir}/insert_rings.yaml",
    },
    "sort_items": {
        "query": "Sort two peppers into two different containers",
        "object_list": ["red container", "blue container", "green pepper", "yellow pepper"],
        "env_class": "BimanualPickPlate",
        "prompt_yaml": f"{prompt_dir}/sort_items.yaml",
    },
    "open_drawer_and_put_item": {
        "query": "open the drawer and put the cube inside the drawer",
        "object_list": ["top drawer", "top drawer handle", "cube"],
        "env_class": "OpenDrawer",
        "prompt_yaml": f"{prompt_dir}/open_drawer_and_put_item.yaml",
    },    
    "open_lid_and_put_item": {
        "query": "open the saucepan and put the cube inside the saucepan",
        "object_list": ["saucepan lid", "red cube"],
        "env_class": "TakeLidOffSaucepan",
        "prompt_yaml": f"{prompt_dir}/open_lid_and_put_item.yaml",
    },
    "handover_item": {
        "query": "handover the yellow block",
        "object_list": ["yellow block"],
        "env_class": "StackBlocksV3",
        "prompt_yaml": f"{prompt_dir}/handover_item.yaml",
    },
    "push_buttons": {
        "query": "push two buttons",
        "object_list": ["green button", "blue button", "yellow button"],
        "env_class": "BimanualDualPushButtons",
        "prompt_yaml": f"{prompt_dir}/push_buttons.yaml",
    },
    "push_box_to_target": {
        "query": "push the box until it reaches the red target area",
        "object_list": ["box", "red target"],
        "env_class": "BimanualPushBox",
        "prompt_yaml": f"{prompt_dir}/push_box_to_target.yaml",
    },
    "shell_game": {
        "query": "carry out a shell game where one robot hides the block and the other guesses",
        "object_list": ["red cup", "blue cup", "black cup", "green block"],
        "env_class": "StackCups",
        "prompt_yaml": f"{prompt_dir}/shell_game.yaml",
    },
}

