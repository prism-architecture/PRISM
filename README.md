
[**Website**](https://github.com/aaai2026-review/preference-aware/)
| [**Code**](https://github.com/aaai2026-review/preference-aware/tree/main/Code/)
| [**Story Dataset**](https://github.com/aaai2026-review/preference-aware/tree/main/story_dataset/)


## A Vision-Language Architecture for Hierarchical Multi-Agent Coordination

![storytelling_testbed](./images/storytelling_testbed.pdf)

Humans excel at rapidly modeling team members' latent factors, such as preferences, even with limited interaction data. However, replicating this capability in human-agent teams is challenging because existing approaches often require either complex computational models of human behavior or large amounts of interaction data, which are often infeasible for real-time deployments. To overcome these limitations, we propose a novel preference-aware decision-making framework that bridges the gap between model-driven and data-driven methods to enable real-time human-agent interaction. In particular, our approach formulates the agent’s decision-making process as a Markov Decision Process (MDP), where the framework estimates human preference distribution by leveraging a Large Language Model (LLM) and incorporates that to solve the MDP, enabling the agent to make adaptive decisions that optimize long-term collaborative outcomes. We evaluate the framework in a large-scale human-agent interaction study (n = 40) using a collaborative storytelling task to assess its impact on task performance and participants' perceptions of collaboration. Our findings indicate that the proposed framework significantly improved task performance, story quality, and robot contribution while enhancing perceived collaboration fluency, agent competency, and interactivity compared to a myopic baseline. Moreover, external evaluators also found the preference-aware strategy to yield more fluent collaboration and higher-quality stories than the baseline. These findings will enable the development of more collaborative and effective agents to improve task performance and user experience across various environments.

### Collaborative storytelling testbed 

Collaborative storytelling testbed where two humans and a robot (Sam) take turns building a story. Sam uses an MDP with LLM-based preference estimation to select reward-maximizing actions. The interface displays available words, turn order, and points to support and evaluate adaptive collaboration.

<figure>
<img src="./images/illustrative_example.png" width="100%" height="100%" align="center">

<figcaption style="display: block; margin: auto; text-align: center; font-size:12;"> Collaborative Storytelling Testbed</figcaption>
</figure>


#### Dataset Structure:
The collaborative story dataset is available on [Story Dataset](https://github.com/aaai2026-review/preference-aware/tree/main/story_dataset/)

```
raw_data
    |- baseline_sessions
        |-session_1.json
        |-session_2.json
         ...
    |- proposed_sessions
        |-session_1.json
        |-session_2.json
         ...

session_wise_processed_stories
    |- session_1
        |-baseline_method.txt
        |-preference_aware_method.txt

    |- session_2
        |-baseline_method.txt
        |-preference_aware_method.txt
    ...
         
```

### Experiments and Results

To validate M2RL, we evaluated state-of-the-art imitation learning algorithms, namely Behavior Cloning (BC) and Diffusion Policy (DP), on two benchmarks:

**1. Multimodal Interface Benchmark:** We compared the algorithms' performance when trained on demos from a single interface (gamepad) vs all three interfaces. Results showed that both BC and DP (especially DP) achieved lower prediction errors when trained on multimodal interface data. This underscores the importance of collecting demos from diverse interfaces to improve policy robustness.

**2. Multimodal Data Benchmark:** We compared the algorithms' performance when trained on a single RGB+D stream (wrist camera) vs all three streams. DP's performance improved with more camera streams for most tasks, while BC's errors increased. This suggests multimodal data can improve task performance if algorithms can effectively extract relevant cross-modal representations (e.g. using an attention or denoising mechanism like in DP).

M2RL dataset takes a step towards aligning robot imitation learning setups with how humans learn - leveraging multimodal cues and diverse experiences. Our experiments demonstrate the benefits of learning from diverse interfaces and multimodal data to improve manipulation task performance. We hope M2RL enables further research into effective robot learning from human demonstrations.

