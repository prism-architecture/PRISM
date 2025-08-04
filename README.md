[**Website**](https://github.com/aaai2026-review/preference-aware/)
| [**Code**](https://github.com/aaai2026-review/preference-aware/tree/main/Code/)
| [**Story Dataset**](https://github.com/aaai2026-review/preference-aware/tree/main/story_dataset/)

## A Vision-Language Architecture for Hierarchical Multi-Agent Coordination

<p align="center">
  <img src="./images/prism_example.png" width="60%">
</p>

Multi‑agent systems hold the promise of distributed, parallel execution in domains ranging from manufacturing to assistive care. However, enabling multiple agents to follow natural language instructions in dynamic environments remains a challenging problem due to the lack of mechanisms for grounding language into temporally synchronized, vision-aware action plans. To overcome these limitations and enable robust coordination among multiple agents, we present a hierarchical vision-language framework that grounds natural language instructions into synchronized multi-agent workflows by dynamically injecting synchronization constraints via predicate reasoning over multimodal observations and execution history. Additionally, to enable systematic evaluation of multi-agent coordination, we introduce RLBench-COLLAB, an extended benchmark comprising ten two-robot manipulation tasks spanning sequential coordination, parallel coordination, coupled interaction, and behavior-aware reasoning. Our extensive experiments suggest that the proposed method significantly outperforms state-of-the-art planning frameworks, achieving an average task success rate of 72% and an average subtask success rate of 89%. Our comprehensive evaluation and results demonstrate that the proposed method bridges the gap between adaptability and temporal coordination in multi-agent systems, enabling scalable, real-time collaboration in dynamic, open-ended environments.

### RLBench-COLLAB Task Suite

Collaborative storytelling testbed where two humans and a robot (Sam) take turns building a story. Sam uses an MDP with LLM-based preference estimation to select reward-maximizing actions. The interface displays available words, turn order, and points to support and evaluate adaptive collaboration.

<p align="center">
  <img src="./images/rlbench_task_suit.png" width="80%">
</p>

### PRISM Architecture

<p align="center">
  <img src="./images/prism_framework.png" width="60%">
</p>

### Successful Task Execution Results

<table>
  <tr>
    <td align="center">
      <img src="./images/stack_block.gif" width="160px"><br/>
      Stack Blocks
    </td>
    <td align="center">
      <img src="./images/pyramid_stacking.gif" width="160px"><br/>
      Pyramid Stacking
    </td>
    <td align="center">
      <img src="./images/put_into_drawer.gif" width="160px"><br/>
      Put into Drawer
    </td>
    <td align="center">
      <img src="./images/put_into_saucepan.gif" width="160px"><br/>
      Put into Saucepan
    </td>
    <td align="center">
      <img src="./images/push_buttons.gif" width="160px"><br/>
      Push Buttons
    </td>
  </tr>
  <tr>
    <td align="center">
      <img src="./images/shell_game.gif" width="160px"><br/>
      Shell Game
    </td>
    <td align="center">
      <img src="./images/sort_items.gif" width="160px"><br/>
      Sort Items
    </td>
    <td align="center">
      <img src="./images/handover_item.gif" width="160px"><br/>
      Handover Item
    </td>
    <td align="center">
      <img src="./images/insert_ring.gif" width="160px"><br/>
      Insert Ring
    </td>
    <td align="center">
      <img src="./images/push_box.gif" width="160px"><br/>
      Push Box
    </td>
  </tr>
</table>

### Experiments and Results

To validate M2RL, we evaluated state-of-the-art imitation learning algorithms, namely Behavior Cloning (BC) and Diffusion Policy (DP), on two benchmarks:

**1. Multimodal Interface Benchmark:** We compared the algorithms' performance when trained on demos from a single interface (gamepad) vs all three interfaces. Results showed that both BC and DP (especially DP) achieved lower prediction errors when trained on multimodal interface data. This underscores the importance of collecting demos from diverse interfaces to improve policy robustness.

**2. Multimodal Data Benchmark:** We compared the algorithms' performance when trained on a single RGB+D stream (wrist camera) vs all three streams. DP's performance improved with more camera streams for most tasks, while BC's errors increased. This suggests multimodal data can improve task performance if algorithms can effectively extract relevant cross-modal representations (e.g. using an attention or denoising mechanism like in DP).

M2RL dataset takes a step towards aligning robot imitation learning setups with how humans learn—leveraging multimodal cues and diverse experiences. Our experiments demonstrate the benefits of learning from diverse interfaces and multimodal data to improve manipulation task performance. We hope M2RL enables further research into effective robot learning from human demonstrations.
