# Imports and Model Initialization
import copy
from functools import lru_cache
import math
import random
import re
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, SinkCache
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.nn.utils.rnn import pad_sequence

from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, LoraConfig

import contextlib
import accelerate

accelerator = accelerate.Accelerator(gradient_accumulation_steps=64)
accelerator.device

@contextlib.contextmanager
def set_left_padding(tokenizer):
    # Store the original padding side
    original_padding_side = tokenizer.padding_side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side='left'
    # Set padding side to left
    tokenizer.padding_side = "left"
    try:
        yield tokenizer
    finally:
        # Restore original padding side
        tokenizer.padding_side = original_padding_side
        tokenizer.truncation_side = original_truncation_side


# 加载模型和 tokenizer
model_name = "/mnt/hwfile/ai4chem/CKPT/longcot_pt_GEMMA_ZD_10_23_1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
)

# 设置 LoRA 配置
lora_config = LoraConfig(
    r=8,  # 低秩矩阵的秩
    lora_alpha=16,  # LoRA 的缩放系数
    target_modules=["k_proj","q_proj","o_proj", "v_proj","down_proj","gate_proj","up_proj",],  # 目标模块，通常是查询和键的投影层
    lora_dropout=0.1,  # dropout 概率
    bias="none",  # 不在 LoRA 中包含偏置
)

# 使用 peft 将模型转换为 LoRA 微调模型
model = get_peft_model(model, lora_config)

print("Model successfully converted to LoRA format.")

select_prefix = ""
meta_action_types = ["<problem>", "<critic>", "<refine>", "<conclusion>"]
meta_action_types_weight = [0.2, 0.4, 0.4, 0.3]

GT = f"#### 23"

hint = f'<hint> Try generate a reasonable rationale solution that can got final answer {GT}</hint>'
# hint = ''


hint_for_critics = f"<hint> Point out the potential flaws in the current solution. </hint>"
hint_for_refine = f"<hint> Try to refine the current solution for higher quality. </hint>"
hint_for_conclusion = "<hint> Try to summarize the current solution and draw a conclusion. Final answer should bracket in \\box{answer} </hint>"
hint_for_divide_and_conquer = f"<hint> Try divide the problem into smaller easier sub-problems and solve them divide-and-conquer. </hint>"


CHUNCKED_INFERENCE_LENGTH = 64
LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
GENERATE_MAX_NEW_TOKENS = 256
CUT_OFF_LEN = 1024

import torch

# Tree Node Structure
class TreeNode:
    def __init__(self, state, parent=None, index=0):
        self.index = index  # Index of the node in the tree
        self.state = state  # Current state text representation
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of visits
        self.value = 0  # Value estimate of the current node
        self.policy = {}  # Policy probabilities for selecting child nodes
        self.policy_entropy = {}
        self.policy_varentropy = {}
        self.policy_cal_ready_texts = ""
        self.value_cal_ready_texts = ""
        self.true_value_from_tree = None
        self.leaf_type = ""
        self.rectify_visits = 0
        self.original_value = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_child_policy_prob(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy.keys(), prob)}[child]
    
    def get_child_policy_entropy(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy_entropy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy_entropy.keys(), prob)}[child]
    
    def get_child_policy_varentropy(self, child):
        # 提取logit值并转换为数组
        logits = torch.tensor(list(self.policy_varentropy.values()))
        prob, log_prob = robust_softmax(logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy_varentropy.keys(), prob)}[child]


def value_to_rating_token(value):
    if value > 0:
        return "<positive_rating>"
    elif value < 0:
        return "<negative_rating>"
    else:
        return "<unknow_rating>"


def tree_to_string(node):
    cur = f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
    childs_strings = "\n".join([tree_to_string(child) for child in node.children])
    return cur + "\n" + childs_strings


def path_to_string(node):
    path = []
    while node:
        path.append(node)
        node = node.parent
    string = "\n".join(
        [
            f"<start_of_father_id>{node.parent.index if node.parent else -1}<end_of_father_id><start_of_local_id>{node.index}<end_of_local_id><start_of_thought>{node.state}<end_of_thought><start_of_rating>{value_to_rating_token(node.value)}<end_of_rating>"
            for node in path[::-1]
        ]
    )
    return string


def get_max_node_id_in_tree(node):
    if not node.parent:
        while node.parent:
            node = node.parent
    max_id = node.index
    for child in node.children:
        max_id = max(max_id, get_max_node_id_in_tree(child))
    return max_id


def get_root(node):
    while node.parent:
        node = node.parent
    return node


def problem_declaration_template(problem):
    return f"<start_of_father_id>-1<end_of_father_id><start_of_local_id>0<end_of_local_id><start_of_thought><problem>{problem}<end_of_thought>"


def selection_head_template(tree):
    return tree.to_string() + "\n<start_of_father_id>"


def policy_head_template(selected_node, local_id, meta="", hint=""):
    return (
        path_to_string(selected_node)
        + f"{hint}\n<start_of_father_id>{selected_node.index if selected_node else -1}<end_of_father_id><start_of_local_id>{local_id}<end_of_local_id><start_of_thought>{meta}"
    )


def value_head_template(selected_node):
    return (
        path_to_string(selected_node.parent)
        + f"\n<start_of_father_id>{selected_node.parent.index if selected_node.parent else -1}<end_of_father_id><start_of_local_id>{selected_node.index}<end_of_local_id><start_of_thought>{selected_node.state}<end_of_thought><start_of_rating>"
    )


selection_head_stopping_criteria = ["<end_of_father_id>"]

policy_head_stopping_criteria = ["<end_of_thought>"]

value_head_stopping_criteria = ["<end_of_rating>"]


def robust_softmax(logits):
    log_probs = F.log_softmax(logits - logits.max(), dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs


def length_normed_log_probs(
    sequence_ids, logits_tensor, return_entropy=False, return_varentropy=False
):

    # Gather the log probabilities using advanced indexing
    # For each sequence and each position, select the log probability of the generated token
    # https://github.com/xjdr-alt/entropix/blob/e55e9a38cb6fb7ad23e9fe68196c61a257a3da52/entropix/torch_sampler.py#L15

    probs, log_probs = robust_softmax(logits_tensor)

    selected_log_probs = torch.gather(
        log_probs.transpose(0, 1), 2, sequence_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Sum the log probabilities for each sequence
    summed_log_probs = selected_log_probs.sum(dim=1)

    # Normalize by sequence length
    normalized_log_probs = summed_log_probs / sequence_ids.size(1)

    # Compute entropy and variance of entropy

    entropy = -torch.sum(probs * log_probs, dim=-1) / LN_2  # Convert to base-2
    varentropy = torch.sum(
        probs * (log_probs / LN_2 + entropy.unsqueeze(-1)) ** 2, dim=-1
    )

    summed_entropy = entropy.sum(dim=0)
    normalized_entropy = summed_entropy / sequence_ids.size(1)

    summed_varentropy = varentropy.sum(dim=0)
    normalized_varentropy = summed_varentropy / sequence_ids.size(1)

    return normalized_log_probs, normalized_entropy, normalized_varentropy


# # Compute neural UCB Head (p(a|s))
# @torch.no_grad()
# def compute_selection_head(tree, seleted_node_id, num_candidates=3):
#     inputs = tokenizer(state, return_tensors='pt')
#     outputs = model.generate(**inputs, max_new_tokens=8, do_sample=True, num_return_sequences=num_candidates, output_logits=True, return_dict_in_generate=True, stopping_criteria=selection_head_stopping_criteria)

#     # Extract generated sequences and scores
#     output_ids = outputs.sequences

#     if isinstance(outputs.logits, tuple):
#         logits = torch.stack(outputs.logits, dim=0)
#     else:
#         logits = outputs.logits  # If already a tensor

#     normalized_logits = length_normed_logit(output_ids[:,-logits.shape[0]:], logits)

#     generated_texts = tokenizer.batch_decode(output_ids[:,-logits.shape[0]:], skip_special_tokens=False)

#     return generated_texts, normalized_logits.softmax(dim=0)


def clean_generated_text(text):
    return text[: text.find("<end_of_thought>")]


@torch.no_grad()
def meta_compute_policy_head(selected_node, num_candidates=3, meta_ratio=0.80):
    if random.random() < meta_ratio:
        generated_texts, normalized_log_probs, normalized_entropy, varentropy = (
            compute_policy_head(selected_node, num_candidates)
        )
        return generated_texts, normalized_log_probs.tolist(), normalized_entropy.tolist(), varentropy.tolist()

    metas = random.choices(
        meta_action_types, meta_action_types_weight, k=num_candidates
    )
    generated_texts = []
    policy_logits = []
    normalized_entropys = []
    varentropys = []
    for i, meta in enumerate(metas):
        texts, policy_probs, normalized_entropy, varentropy = compute_policy_head(
            selected_node, num_candidates=1, meta=meta
        )
        generated_texts.append(texts[0])
        policy_logits.append(policy_probs.item())
        normalized_entropys.append(normalized_entropy.item())
        varentropys.append(varentropy.item())
    return generated_texts, policy_logits, normalized_entropys, varentropys


# Optimized Compute Policy Head (p(a|s))
@torch.no_grad()
def compute_policy_head(selected_node, num_candidates=3, meta=""):
    if meta == "<conclusion>":
        inputs_string = policy_head_template(
            selected_node, get_max_node_id_in_tree(selected_node) + 1, meta, hint_for_critics
        )
    elif meta == "<problem>":
        inputs_string = policy_head_template(
            selected_node,
            get_max_node_id_in_tree(selected_node) + 1,
            "",
            hint_for_divide_and_conquer,
        )
    elif meta == "<critic>":
        inputs_string = policy_head_template(
            selected_node,
            get_max_node_id_in_tree(selected_node) + 1,
            meta,
            hint_for_critics,
        )
    elif meta == "<refine>":
        inputs_string = policy_head_template(
            selected_node,
            get_max_node_id_in_tree(selected_node) + 1,
            meta,
            hint_for_refine,
        )
    else:
        inputs_string = policy_head_template(
            selected_node, get_max_node_id_in_tree(selected_node) + 1, '',hint 
        )

    with set_left_padding(tokenizer):
        inputs = tokenizer(inputs_string, return_tensors="pt", truncation=True, padding=True, max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = accelerator.unwrap_model(model).generate(
        **inputs,
        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=num_candidates,
        output_logits=True,
        return_dict_in_generate=True,
        stop_strings=policy_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    # Extract generated sequences and scores
    output_ids = outputs.sequences

    if isinstance(outputs.logits, tuple):
        logits = torch.stack(outputs.logits, dim=0)
    else:
        logits = outputs.logits  # If already a tensor

    generated_texts = tokenizer.batch_decode(
        output_ids[:, -logits.shape[0] :], skip_special_tokens=False
    )

    normalized_log_probs, normalized_entropy, varentropy = length_normed_log_probs(
        output_ids[:, -logits.shape[0] :], logits
    )

    generated_texts = [meta + clean_generated_text(text) for text in generated_texts]

    if meta != "<conclusion>":
        for i, generated_text in enumerate(generated_texts):
            if "The answer is:" in generated_text and not generated_texts[i].startswith(
                "<conclusion>"
            ):
                generated_texts[i] = "<conclusion>" + generated_texts[i]

    if meta == "<problem>":
        for i, generated_text in enumerate(generated_texts):
            if not generated_texts[i].startswith("<problem>"):
                generated_texts[i] = "<problem>" + generated_texts[i]
    elif meta == "<critic>":
        for i, generated_text in enumerate(generated_texts):
            if not generated_texts[i].startswith("<critic>"):
                generated_texts[i] = "<critic>" + generated_texts[i]
    elif meta == "<refine>":
        for i, generated_text in enumerate(generated_texts):
            if not generated_texts[i].startswith("<refine>"):
                generated_texts[i] = "<refine>" + generated_texts[i]
    elif meta == "<conclusion>":
        for i, generated_text in enumerate(generated_texts):
            if not generated_texts[i].startswith("<conclusion>"):
                generated_texts[i] = "<conclusion>" + generated_texts[i]

    return generated_texts, normalized_log_probs, normalized_entropy, varentropy


@torch.no_grad()
def compute_value_head(node):
    # 将state传入tokenizer并获取模型输入
    text_for_value = value_head_template(node) + '<positive_rating>'
    with set_left_padding(tokenizer):
        inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding=True, max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits


    last_second_logits = logits[:, -1, :]  # 倒数第一个位置的logit

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    # 提取 "<positive_rating>" 和 "<negative_rating>" 的logit
    positive_logit = last_second_logits[:, positive_token_id]
    negative_logit = last_second_logits[:, negative_token_id]

    # 将两者结合计算 value
    # value = positive_logit - negative_logit  # 或根据需要使用其他计算方式
    prob, log_prob = robust_softmax(torch.cat([positive_logit, negative_logit], dim=0))

    return log_prob[0].item()


def compute_acummlated_path_reward(node):
    if not node.parent:  # root node
        return 0.0
    else:
        return node.value + compute_acummlated_path_reward(node.parent)


from grading import check


def compute_rule_orm_head(selected_node):
    # 将state传入tokenizer并获取模型输入
    result = check(GT, selected_node.state, "")

    return result


def get_max_reward_in_path(node):
    if not node.parent:
        return node.value
    else:
        return max(node.value, get_max_reward_in_path(node.parent))


def reward_normalize(reward, maximum, minimum):
    return (reward - minimum) / (maximum - minimum)


# MCTS Search
class MCTS:
    def __init__(
        self,
        model,
        tokenizer,
        num_simulations=-1,
        num_candidates_per_expansion=2,
        exploration_const=1.0,
        discount_factor=0.9,
        reward_epsilon=1e-6,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations if num_simulations != -1 else 32
        self.exploration_const = exploration_const
        self.end = False
        self.discount_factor = discount_factor
        self.num_candidates = num_candidates_per_expansion
        self.reward_epsilon = reward_epsilon
        self.varentropy_lambda = 0.1

    def search(self, root_node):
        if not root_node.children:
            root_node.value = 0

        for _ in range(self.num_simulations):
            self.simulate(root_node)
            if self.end:
                break

        for leaf in self.identify_leaf(root_node):
            if leaf.leaf_type == "successful":
                self.rectify_values_from_leaf(leaf, 0)
            else:
                self.rectify_values_from_leaf(leaf, np.log(self.reward_epsilon))

        return root_node

        # return self.get_policy_from_visits(root_node)

    def simulate(self, node):
        if node.is_leaf():
            value = self.expand_node(node) * self.discount_factor
        else:
            best_child = self.select_action(node)
            value = self.simulate(best_child) * self.discount_factor
        node.visits += 1
        node.value += (value - node.value) / node.visits
        # if '<critic>' in node.state:
        #     return -node.value
        # else:
        #     return node.value
        return node.value

    def expand_node(self, node):
        texts, policy_probs, entropys, varentropys = meta_compute_policy_head(node, self.num_candidates)

        for i, (text, policy_prob, entropy, varentropy) in enumerate(zip(texts, policy_probs, entropys, varentropys)):
            child_node = TreeNode(
                state=text, parent=node, index=get_max_node_id_in_tree(node) + 1
            )
            # child_node.policy = policy_probs[i]
            node.policy[child_node] = policy_prob
            node.policy_entropy[child_node] = entropy
            node.policy_varentropy[child_node] = varentropy
            node.add_child(child_node)
            child_node.value = self.compute_value(child_node)
            if "<conclusion>" in child_node.state:
                if compute_rule_orm_head(child_node):
                    self.end = True
                    child_node.leaf_type = "successful"
                else:
                    child_node.leaf_type = "failed"
            print(
                f"Id:{child_node.index}, Child: {text}, Policy: {math.exp(policy_prob)}, Value: {math.exp(child_node.value)}"
            )
        return max(child_node.value for child_node in node.children)

    def compute_value(self, node):
        # Use the model to predict the value of the current state
        value = compute_value_head(node)
        node.value = value
        node.original_value = copy.deepcopy(value)
        return value

    def select_action(self, node):
        total_visits = sum(child.visits for child in node.children)
        ucb_scores = [
            (
                child.value
                + self.exploration_const
                * node.get_child_policy_prob(child)
                * node.get_child_policy_entropy(child)
                * np.sqrt(total_visits)
                / (1 + child.visits)
                + self.varentropy_lambda * node.get_child_policy_varentropy(child)
            )
            for child in node.children
        ]
        return node.children[np.argmax(ucb_scores)]

    def identify_leaf(self, node):
        result = set()
        if node.is_leaf():
            if node.leaf_type in ["successful", "failed"]:
                result.add(node)
        else:
            for child in node.children:
                result |= self.identify_leaf(child)
        return result

    def rectify_values_from_leaf(self, node, value):
        node.rectify_visits += 1

        if not node.true_value_from_tree:
            node.true_value_from_tree = value
        else:
            node.true_value_from_tree += (
                value - node.true_value_from_tree
            ) / node.rectify_visits
        if node.parent:
            self.rectify_values_from_leaf(
                node.parent, node.true_value_from_tree * self.discount_factor
            )


@lru_cache(maxsize=1000)
def training_time_policy_predict(node, model):
    # Use the model to predict the value of the current state
    text_for_policy = policy_head_template(node.parent, node.index) + node.state
    with set_left_padding(tokenizer):
        inputs = tokenizer(text_for_policy, return_tensors="pt", truncation=True, padding=True, max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    target = tokenizer(node.state, return_tensors="pt")
    target = {k: v.to(accelerator.device) for k, v in target.items()}
    # print(target["input_ids"].shape,logits.shape)
    normalized_log_probs, normalized_entropy, varentropy = length_normed_log_probs(
        target["input_ids"].transpose(0, 1), logits[:, -target["input_ids"].shape[-1] :]
    )
    # print(normed_logit.shape)
    return normalized_log_probs


@lru_cache(maxsize=1000)
def training_time_value_predict(node):
    text_for_value = value_head_template(node) + value_to_rating_token(node.value)
    with set_left_padding(tokenizer):
        inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding=True, max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    last_second_logits = logits[:, -1, :]  # 倒数第一个位置的logit

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    # 提取 "<positive_rating>" 和 "<negative_rating>" 的logit
    positive_logit = last_second_logits[:, positive_token_id]
    negative_logit = last_second_logits[:, negative_token_id]

    value_logit = torch.cat([positive_logit, negative_logit], dim=0).unsqueeze_(0)

    return value_logit
    # value_logit = positive_logit - negative_logit

    # return torch.tanh(value_logit)


# Model Training using MCTS policy and value
criterion_policy = torch.nn.CrossEntropyLoss()
# criterion_value = torch.nn.MSELoss()
criterion_value = torch.nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)


def traverse_tree(node):
    """
    Generator to traverse the entire tree from the root node
    """
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        # print(f"Current Node: {current_node.state}")
        yield current_node
        nodes.extend(current_node.children)


def ppo_loss_clip(old_probs, new_probs, advantages, epsilon=0.2):
    # print(f"Old Probs: {old_probs}")
    # print(f"New Probs: {new_probs}")
    # print(f"Advantages: {advantages}")

    # 计算策略比率
    ratios = new_probs / (old_probs + 1e-8)
    # print(f"Ratios: {ratios}")

    # 计算clipped和未clipped的损失
    unclipped_loss = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    # print(f"Clipped Ratios: {clipped_ratios}")

    clipped_loss = clipped_ratios * advantages

    # 返回PPO的最终损失
    return -torch.min(unclipped_loss, clipped_loss).mean()


def ppo_loss(
    old_probs,
    new_probs,
    advantages,
    kl_penalty_coeff=0.1,
    desired_kl=0.01,
    epsilon=0.2,
    entropy_coeff=0.01,
):
    # print(f"Old Probs: {old_probs}")
    # print(f"New Probs: {new_probs}")
    # print(f"Advantages: {advantages}")

    # 计算策略比率
    ratios = new_probs / (old_probs.detach() + 1e-8)
    # print(f"Ratios: {ratios}")

    # 计算clipped和未clipped的损失
    unclipped_loss = ratios * advantages
    clipped_ratios = torch.clamp(ratios, 1 - epsilon, 1 + epsilon)
    # print(f"Clipped Ratios: {clipped_ratios}")

    clipped_loss = clipped_ratios * advantages

    # 计算KL散度
    kl_divergence = old_probs.detach() * (
        torch.log(old_probs.detach() + 1e-8) - torch.log(new_probs + 1e-8)
    )
    kl_divergence = kl_divergence.sum(dim=-1).mean()
    # print(f"KL Divergence: {kl_divergence}")

    # 动态调整KL惩罚系数
    if kl_divergence > desired_kl * 1.5:
        kl_penalty_coeff *= 2  # 增大惩罚
        # print(f"Increasing KL Penalty Coefficient to {kl_penalty_coeff}")
    elif kl_divergence < desired_kl / 1.5:
        kl_penalty_coeff *= 0.5  # 减少惩罚
        # print(f"Decreasing KL Penalty Coefficient to {kl_penalty_coeff}")

    ppo_loss = -torch.min(unclipped_loss, clipped_loss).mean()

    entropy = -(new_probs * torch.log(new_probs + 1e-8)).sum(dim=-1).mean()
    total_loss = ppo_loss + kl_penalty_coeff * kl_divergence - entropy_coeff * entropy

    return total_loss


def compute_gae(rewards, values, gamma=0.99, lam=0.95):
    """
    计算广义优势估计（GAE）。
    参数：
    - rewards: 一个 episode 中每个时间步的奖励列表
    - values: 一个 episode 中每个时间步的价值估计列表
    - gamma: 折扣因子
    - lam: GAE 衰减因子
    返回：
    - advantages: 每个时间步的 GAE 估计列表
    """
    advantages = []
    gae = 0
    # 从最后一个时间步开始反向计算 GAE
    for t in reversed(range(len(rewards))):
        # 如果不是最后一步，计算下一个价值
        if t == len(rewards) - 1:
            next_value = 0
        else:
            next_value = values[t + 1]

        # TD 残差
        delta = rewards[t] + gamma * next_value - values[t]

        # GAE 公式
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)  # 将每个步骤的优势插入到列表前端

    return advantages


def train_model_with_ppo_gae(
    root_node,
    optimizer,
    epsilon=0.2,
    gamma=0.99,
    lam=0.95,
    lreg=0.01,
    accumulation_steps=8,
):
    optimizer.zero_grad()
    policy_loss = 0
    value_loss = 0
    step_count = 0
    total_loss = 0
    policy_loss_list = []
    value_loss_list = []
    total_loss_list = []

    # 存储所有 episode 的奖励和价值预测
    rewards = []
    values = []

    # 遍历所有节点获取 rewards 和 value
    for node in traverse_tree(root_node):
        if node == root_node:
            continue
        rewards.append(
            node.true_value_from_tree
            if node.true_value_from_tree != None
            else node.value
        )
        values.append(node.original_value)

        print(node.value,node.true_value_from_tree, node.original_value)

    # 计算 GAE
    advantages = compute_gae(rewards, values, gamma, lam)

    # 遍历节点以应用 PPO 和 GAE
    for node, advantage in tqdm(zip(traverse_tree(root_node), advantages)):
        if node == root_node:
            continue
        if len(node.children) > 0:
            # old_policy_probs = torch.tensor(
            #     [node.policy[child] for child in node.children], dtype=torch.float32
            # ).softmax(dim=0)
            with torch.no_grad():
                old_policy_probs = torch.tensor(
                    [
                        training_time_policy_predict(child, accelerator.unwrap_model(model).get_base_model())[0]
                        for child in node.children
                    ],
                    dtype=torch.float32,
                ).softmax(dim=0)
            new_policy_probs = torch.tensor(
                [
                    training_time_policy_predict(child, model)[0]
                    for child in node.children
                ],
                dtype=torch.float32,
            ).softmax(dim=0)

            policy_loss = ppo_loss(
                old_policy_probs, new_policy_probs, torch.tensor(advantage), epsilon
            )

            predicted_value_logit = training_time_value_predict(node).squeeze(0)
            positive_rating_log_prob = (
                node.true_value_from_tree
                if node.true_value_from_tree != None
                else node.value
            )
            positive_rating_log_prob = torch.tensor(
                [positive_rating_log_prob], dtype=torch.float32
            )
            clamp_positive_rating_prob = torch.exp(torch.clamp(
                positive_rating_log_prob, math.log(1e-6), 0
            ))
            clamp_negative_rating_prob = 1 - clamp_positive_rating_prob
            target_probs = torch.tensor(
                [clamp_positive_rating_prob, clamp_negative_rating_prob],
                dtype=torch.float32,
            )
            print(predicted_value_logit, target_probs)
            value_loss = F.binary_cross_entropy_with_logits(
                predicted_value_logit, target_probs.to(accelerator.device)
            )
            # true_value_one_hot = torch.tensor(
            #     [1 if np.exp(positive_rating_log_prob) >= 0.5 else 0],
            #     dtype=torch.long, device=predicted_value_logit.device
            # )
            # value_loss += criterion_value(predicted_value_logit.softmax(dim=0), true_value_one_hot)

            # 每处理完一个节点后检查是否达到累积步骤
            step_count += 1
            total_loss = policy_loss + value_loss #/ accumulation_steps
            print(policy_loss.item(), value_loss.item(), total_loss.item())
            accelerator.backward(total_loss)

            # 累积足够的梯度后，进行一次参数更新
            optimizer.step()
            optimizer.zero_grad()
            policy_loss_list.append(policy_loss.item())
            value_loss_list.append(value_loss.item())
            total_loss_list.append(total_loss.item())
            policy_loss = 0
            value_loss = 0
            total_loss = 0

    return np.mean(policy_loss_list), np.mean(value_loss_list), np.mean(total_loss_list)


# Self-play and Training Loop
class AlphaGoZeroForMath:
    def __init__(self, model, tokenizer,optimizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mcts = MCTS(model, tokenizer)
        self.optimizer = optimizer

    def self_play(self, initial_state):
        root_node = TreeNode(state=initial_state)
        root_node = self.mcts.search(root_node)
        return root_node

    def train(self, num_iterations, initial_state):
        for i in range(num_iterations):
            self.mcts = MCTS(model, tokenizer)
            root_node = self.self_play(initial_state)
            self.update_model(root_node)

    def update_model(self, root_node):
        loss = train_model_with_ppo_gae(root_node, self.optimizer)
        print(f"Training Loss: {loss}")

from datasets import load_dataset

# gsm8k = load_dataset("gsm8k",subset='train')
# dataloader = torch.utils.data.DataLoader(gsm8k, batch_size=1, shuffle=True)

model, tokenizer, optimizer = accelerator.prepare(model, tokenizer, optimizer)

# Running Training
initial_state = problem_declaration_template(
    "There are several chickens and rabbits in a cage. Counting from the top, there are 35 heads and counting from the bottom, there are 94 feet. How many chickens are there?"
)
agent = AlphaGoZeroForMath(model, tokenizer, optimizer)
agent.train(num_iterations=10, initial_state=initial_state)


# # pip install accelerate
# from transformers import AutoTokenizer, AutoModelForCausalLM
# import torch

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-2b-it",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
# )

# input_text = "Write me a poem about Machine Learning."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids, max_new_tokens=32, do_sample=True,num_return_sequences=3)
# print(outputs)
# print(tokenizer.decode(outputs))
