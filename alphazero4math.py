# Imports and Model Initialization
import copy
from email import policy
from functools import lru_cache
import math
import random
import re
import token
from anyio import value
import datasets
from py import log
from scipy import optimize
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, SinkCache
import torch
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
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
accelerator = accelerate.Accelerator()


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

@contextlib.contextmanager
def set_left_truncate(tokenizer):
    # Store the original padding side
    original_truncation_side = tokenizer.truncation_side
    tokenizer.truncation_side='left'
    try:
        yield tokenizer
    finally:
        tokenizer.truncation_side = original_truncation_side

def value_to_rating_token(value):
    if math.exp(value) >= 0.5 and math.exp(value) <= 1:
        return "<positive_rating>"
    elif math.exp(value) < 0.5 and math.exp(value) >= 0:
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

select_prefix = ""
meta_action_types = ["<problem>", "<critic>", "<refine>", "<conclusion>"]
meta_action_types_weight = [0.2, 0.4, 0.4, 0.3]



LN_2 = 0.69314718056  # ln(2) = 1.0 / LOG2_E
GENERATE_MAX_NEW_TOKENS = 256
CUT_OFF_LEN = 1024
MAX_CHILDREN_NUM = 5

import torch

import math
import numpy as np
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
    
    def get_path_reward(self):
        path_len = 1
        reward = 0
        node = self
        while node.parent:
            path_len += 1
            reward += node.value
            node = node.parent
        return reward / path_len

    def should_expand(self):
        if len(self.children) == 0:
            return True
        if max([child.value for child in self.children]) < self.value and len(self.children) < MAX_CHILDREN_NUM:
            return True
        return False

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
    

# MCTS Search
class MCTS:
    def __init__(
        self,
        envoirment,
        model,
        tokenizer,
        num_simulations=-1,
        num_candidates_per_expansion=2,
        exploration_const=1.414,
        discount_factor=0.9,
        reward_epsilon=1e-6,
        patient=2
    ):
        self.envoirment = envoirment
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations if num_simulations != -1 else 32
        self.exploration_const = exploration_const
        self.patient = patient
        self.discount_factor = discount_factor
        self.num_candidates = num_candidates_per_expansion
        self.reward_epsilon = reward_epsilon
        self.varentropy_lambda = 0.1

    def search(self, root_node):
        if not root_node.children:
            root_node.value = 0

        for _ in tqdm(range(self.num_simulations)):
            self.simulate(root_node)
            max_reward, path_len = find_max_reward_path(root_node)
            print(f'find max reward path: {max_reward} with {path_len} steps.')
            if self.patient <= 0:
                break

        for leaf in self.identify_leaf(root_node):
            if leaf.leaf_type == "successful":
                self.rectify_values_from_leaf(leaf, 0)
            else:
                self.rectify_values_from_leaf(leaf, np.log(self.reward_epsilon))

        return root_node

        # return self.get_policy_from_visits(root_node)

    def simulate(self, node):
        if node.is_leaf() or node.should_expand():
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
        texts, policy_probs, entropys, varentropys = meta_compute_policy_head(self.model, self.tokenizer, node, self.num_candidates, envoirment=self.envoirment)

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
                if self.envoirment.compute_rule_orm_head(child_node):
                    self.patient -= 1
                    child_node.leaf_type = "successful"
                else:
                    child_node.leaf_type = "failed"
            # print(
            #     f"Id:{child_node.index}, Child: {text}, Policy: {node.get_child_policy_prob(child_node)}, Value: {math.exp(child_node.value)}"
            # )
        return self.select_action(node).value

    def compute_value(self, node):
        # Use the model to predict the value of the current state
        value = compute_value_head(self.model, self.tokenizer, node)
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


hint = '<hint> Try generate a reasonable rationale solution that can got final answer {GT}</hint>'
# hint = ''

hint_for_critics = f"<hint> Point out the potential flaws in the current solution. </hint>"
hint_for_refine = f"<hint> Try to refine the current solution for higher quality. </hint>"
hint_for_conclusion = "<hint> Try to summarize the current solution and draw a conclusion. Final answer should bracket in \\box{answer} </hint>"
hint_for_divide_and_conquer = f"<hint> Try divide the problem into smaller easier sub-problems and solve them divide-and-conquer. </hint>"


import torch
import torch.nn.functional as F
from functools import lru_cache
import random

# 模板生成函数
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

def clean_generated_text(text):
    return text[: text.find("<end_of_thought>")]

def find_max_reward_path(node):
    path = 0
    reward = 0
    while node:
        reward += node.value
        path += 1
        if not node.children:
            break
        node = max(node.children, key=lambda x: x.value)
    return math.exp(reward), path

# 数值稳定的 softmax 函数
def robust_softmax(logits):
    logits = torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits
    log_probs = F.log_softmax(logits, dim=-1)
    probs = torch.exp(log_probs)
    return probs, log_probs


# 长度归一化的对数概率、熵和熵的方差计算
def length_normed_log_probs(sequence_ids, logits_tensor, attention_mask=None, return_entropy=False, return_varentropy=False):
    logits_tensor = logits_tensor[..., :-1, :].contiguous()
    sequence_ids = sequence_ids[..., 1:].contiguous()
    attention_mask = attention_mask[..., 1:].contiguous() if attention_mask is not None else None
    log_probs = F.log_softmax(logits_tensor, dim=-1)
    selected_log_probs = log_probs.gather(2, sequence_ids.unsqueeze(-1)).squeeze(-1)

    if attention_mask is not None:
        selected_log_probs = selected_log_probs * attention_mask

    summed_log_probs = selected_log_probs.sum(dim=1)
    length = sequence_ids.size(1) if attention_mask is None else attention_mask.sum(dim=1)
    normalized_log_probs = summed_log_probs / length

    if return_entropy or return_varentropy:
        probs = torch.exp(log_probs)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        if attention_mask is not None:
            entropy = entropy * attention_mask
        summed_entropy = entropy.sum(dim=1)
        normalized_entropy = summed_entropy / length

    if return_varentropy:
        varentropy = torch.sum(probs * (log_probs + entropy.unsqueeze(-1)) ** 2, dim=-1)
        if attention_mask is not None:
            varentropy = varentropy * attention_mask
        summed_varentropy = varentropy.sum(dim=1)
        normalized_varentropy = summed_varentropy / length
        return normalized_log_probs, normalized_entropy, normalized_varentropy

    if return_entropy:
        return normalized_log_probs, normalized_entropy
    else:
        return normalized_log_probs


# 策略生成的主要函数
@torch.no_grad()
def compute_policy_head(model, tokenizer, selected_node, num_candidates=3, meta="", envoirment=None):
    local_id = get_max_node_id_in_tree(selected_node) + 1
    hint_text = {
        "<conclusion>": hint_for_critics,
        "<problem>": hint_for_divide_and_conquer,
        "<critic>": hint_for_critics,
        "<refine>": hint_for_refine,
    }.get(meta, hint.format(GT=envoirment.get_ground_truth(selected_node)))

    inputs_string = policy_head_template(selected_node, local_id, meta, hint_text)
    inputs = tokenizer(
        inputs_string,
        return_tensors="pt",
        truncation=True,
        padding='longest',
        max_length=CUT_OFF_LEN
    )
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}

    outputs = accelerator.unwrap_model(model).generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=GENERATE_MAX_NEW_TOKENS,
        do_sample=True,
        num_return_sequences=num_candidates,
        return_dict_in_generate=True,
        output_scores=True,
        temperature=0.75,
        output_logits=True,
        stop_strings=policy_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    generated_sequences = outputs.sequences[:, inputs['input_ids'].size(1):]
    generated_sequences_mask = generated_sequences != tokenizer.pad_token_id
    generated_texts = tokenizer.batch_decode(generated_sequences, skip_special_tokens=True)

    logits = torch.stack(outputs.logits, dim=1)
    normalized_log_probs, normalized_entropy, varentropy = length_normed_log_probs(
        generated_sequences, logits, attention_mask=generated_sequences_mask, return_entropy=True, return_varentropy=True
    )

    normalized_probs = torch.exp(normalized_log_probs)

    generated_texts = [meta + clean_generated_text(text) for text in generated_texts]
    for i, generated_text in enumerate(generated_texts):
        if not generated_text.startswith(meta):
            generated_texts[i] = meta + generated_text

    return generated_texts, normalized_probs.tolist(), normalized_entropy.tolist(), varentropy.tolist()


# 价值头生成函数
@torch.no_grad()
def compute_value_head(model, tokenizer, node):
    text_for_value = value_head_template(node) + '<positive_rating>'
    inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits

    last_logits = logits[:, -2, :]
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    positive_logit = last_logits[:, positive_token_id]
    negative_logit = last_logits[:, negative_token_id]
    value_logits = torch.stack([positive_logit, negative_logit], dim=1)

    probs, log_probs = robust_softmax(value_logits)
    return log_probs[:, 0].item()


# 元策略生成函数
@torch.no_grad()
def meta_compute_policy_head(model, tokenizer, selected_node, num_candidates=3, meta_ratio=0.5, envoirment=None):
    if random.random() < meta_ratio:
        return compute_policy_head(model, tokenizer, selected_node, num_candidates, envoirment=envoirment)

    metas = random.choices(meta_action_types, meta_action_types_weight, k=num_candidates)
    generated_texts, policy_probs, normalized_entropys, varentropys = [], [], [], []

    for meta in metas:
        texts, policy_probs, normalized_entropy, varentropy = compute_policy_head(model, tokenizer,
            selected_node, num_candidates=1, meta=meta, envoirment=envoirment
        )
        generated_texts.append(texts[0])
        policy_probs.append(policy_probs[0])
        normalized_entropys.append(normalized_entropy[0])
        varentropys.append(varentropy[0])

    return generated_texts, policy_probs, normalized_entropys, varentropys

def padding_nodes(tensor, max_len):
    feature_dim = tensor.size(-1)
    pad_len = max_len - tensor.size(1)
    pad_tensor = torch.zeros(tensor.size(0), pad_len, feature_dim, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad_tensor], dim=1)

def tokenize_policy_predict(nodes,tokenizer):
    with set_left_truncate(tokenizer):
        text_for_policys = [policy_head_template(node.parent, node.index) + node.state for node in nodes]
        targets = [node.state for node in nodes]
        # with set_left_padding(tokenizer):
        inputs = tokenizer(text_for_policys, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
        target = tokenizer(targets, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    ret = {'input_ids':inputs['input_ids'],'attention_mask':inputs['attention_mask'],'target':target['input_ids'],'target_attention_mask':target['attention_mask']}
    return ret

def forward_policy_predict(model,tokenizer,inputs):
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    target_ids = inputs["target"]
    target_mask = inputs["target_attention_mask"]
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits[:,:-1,:][:, -target_ids[:,1:].shape[-1] :] 
    log_probs = F.log_softmax(logits, dim=-1)
    seleted_log_probs = log_probs.gather(2, target_ids[:,1:].unsqueeze(-1)).squeeze(-1) 
    return seleted_log_probs

def tokenize_value_predict(node,tokenizer):
    with set_left_truncate(tokenizer):
        text_for_value = value_head_template(node) + '<positive_rating>'
        inputs = tokenizer(text_for_value, return_tensors="pt", truncation=True, padding='longest', max_length=CUT_OFF_LEN)
    inputs = {'value_' + k: v for k, v in inputs.items()}
    return inputs

import torch

def forward_value_predict(model, tokenizer, inputs):
    inputs = {k: v.to(accelerator.device) for k, v in inputs.items()}
    input_ids = inputs.pop("value_input_ids")
    attention_mask = inputs.pop("value_attention_mask")
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
    logits = outputs.logits
    pos = attention_mask.sum(dim=1) - 1  # [batch_size]

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的 token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    # 构建索引张量
    batch_size = logits.size(0)
    indices = torch.tensor([positive_token_id, negative_token_id], device=accelerator.device)  # [2]

    # 扩展 indices 以匹配输入 logits 的维度
    selected_logit = logits[range(batch_size), pos]  # [batch_size, num_tokens]
    selected_logit = selected_logit[:, indices]      # 提取每行中指定 token 的 logits

    return selected_logit

def get_path_reward_real(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += node.true_value_from_tree if node.true_value_from_tree is not None else node.value
        node = node.parent
    return reward / path_len

def get_path_reward_sim(node):
    path_len = 1
    reward = 0
    while node.parent:
        path_len += 1
        reward += node.original_value
        node = node.parent
    return reward / path_len


def traverse_tree(node):
    """
    Generator to traverse the entire tree from the root node
    """
    visited = set()
    nodes = [node]
    while nodes:
        current_node = nodes.pop()
        if current_node not in visited:
            visited.add(current_node)
            yield current_node
            nodes.extend(current_node.children)
        else:
            continue

def compute_gae_from_node(node, gamma=0.99, lambda_=0.95):
    # 回溯到根节点并记录路径
    path = []
    current_node = node
    while current_node.parent is not None:
        path.append(current_node)
        current_node = current_node.parent
    
    # 从根节点（路径起点）向下遍历到目标节点，逐步计算 GAE
    gae = 0
    factor = 1  # 用于累乘 (gamma * lambda) 的系数

    # 从根节点开始遍历路径到指定节点
    for i in range(len(path) - 1):  # path[-1] 是目标节点，不需要再计算 TD 误差
        current_node = path[i]
        next_node = path[i + 1]
        next_node_reward = get_path_reward_real(next_node)
        next_node_value = get_path_reward_sim(next_node)
        current_node_value = get_path_reward_sim(current_node)

        # 计算 TD 误差
        td_error = next_node_reward + gamma * next_node_value - current_node_value
        # 根据 GAE 累积 TD 误差
        gae += factor * td_error
        # 更新系数，准备下一步的累积
        factor *= gamma * lambda_

    return gae




import os
import pickle
import random
from transformers import Trainer
from datasets import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn.functional as F
import numpy as np

def collator_fn(batch):
    indecies = [example['indices'] for example in batch]
    weights = [example['weights'] for example in batch]
    batch = {k: pad_sequence([torch.tensor(example[k]).squeeze().unsqueeze(-1) for example in batch],True,0).squeeze() for k in batch[0].keys() if k not in ['indices', 'weights']}
    batch['indices'] = torch.tensor(indecies)
    batch['weights'] = torch.tensor(weights)  
    return batch



import pickle
import gzip
import numpy as np
import os

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0
        self.priorities = []
        self.alpha = alpha  # Prioritization factor

    def add(self, data, priority):
        """Add experience to the buffer with its priority."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(data)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = data
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity  # Circular buffer

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences, with probabilities proportional to their priorities."""
        if len(self.buffer) == 0:
            return [], [], []

        priorities = np.array(self.priorities, dtype=np.float32)
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probabilities)
        samples = [self.buffer[idx] for idx in indices]

        # Compute importance-sampling weights
        total = len(self.buffer)
        weights = (total * probabilities[indices]) ** (-beta)
        weights /= weights.max()  # Normalize for stability
        weights = np.array(weights, dtype=np.float32)

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """Update the priorities of sampled experiences."""
        indecies_list = indices.tolist()[0]
        priorities_list = priorities.tolist()
        for idx, priority in zip(indecies_list, priorities_list):
            self.priorities[idx] = priority

    def save(self, filepath):
        """Persist the replay buffer to disk with compression and efficient storage."""
        buffer_array = np.array(self.buffer, dtype=object)  # Convert to NumPy array for storage
        priorities_array = np.array(self.priorities, dtype=np.float32)
        with gzip.open(filepath, 'wb') as f:
            pickle.dump((buffer_array, priorities_array, self.pos), f)

    def load(self, filepath):
        """Load the replay buffer from disk with decompression."""
        if os.path.exists(filepath):
            with gzip.open(filepath, 'rb') as f:
                buffer_array, priorities_array, self.pos = pickle.load(f)
                self.buffer = buffer_array.tolist()
                self.priorities = priorities_array.tolist()


class AlphaGoZeroForMath(Trainer):
    def __init__(self, envoirment, model, tokenizer, mcts, replay_buffer_capacity=100000, **kwargs):
        super().__init__(
            model=model,
            tokenizer=tokenizer,
            **kwargs
        )
        self.envoirment = envoirment
        self.mcts = mcts
        self.tokenizer = tokenizer
        self.replay_buffer = PrioritizedReplayBuffer(capacity=replay_buffer_capacity)
        self.replay_buffer_file = 'replay_buffer.pkl'
        self.replay_buffer.load(self.replay_buffer_file)  # Load existing buffer
        self.create_optimizer_and_scheduler(9999)

    def self_play(self, initial_state):
        self.model.eval()
        """Perform self-play to generate experiences."""
        root_node = TreeNode(state=initial_state)
        root_node = self.mcts.search(root_node)
        return root_node

    def collect_experience(self, root_node):
        """Traverse the MCTS tree to collect experiences and store them in the replay buffer."""

        # Collect training data from the tree
        for node in traverse_tree(root_node):
            if node == root_node:
                continue
            
            reward = node.true_value_from_tree if node.true_value_from_tree is not None else node.value
            advantage = compute_gae_from_node(node)

            policy_input = tokenize_policy_predict([node,], self.tokenizer)
            # Old policy probabilities
            with torch.no_grad():
                old_policy_log_probs = forward_policy_predict(self.model.get_base_model(), self.tokenizer, policy_input)

            advantage_tensor = torch.tensor([advantage], dtype=torch.float32).unsqueeze(0)
            value_input = tokenize_value_predict(node, self.tokenizer)
            value_target = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)

            # Store the experience with initial priority
            experience = {
                'old_policy_log_probs': old_policy_log_probs,
                'advantage': advantage_tensor,
                'value_target': value_target,
                **policy_input,
                **value_input,
            }
            # Use absolute advantage as initial priority
            priority = abs(advantage_tensor.item())
            self.replay_buffer.add(experience, priority)

    def create_dataset_from_buffer(self, batch_size, beta=0.4):
        """Sample a batch from the replay buffer using PER."""
        samples, indices, weights = self.replay_buffer.sample(batch_size, beta)
        if len(samples) == 0:
            return None  # Not enough samples to create a batch

        # Prepare data for the model
        data = {
            'old_policy_log_probs': [],
            'advantage': [],
            'value_target': [],
            'input_ids':[],
            'attention_mask':[],
            'target':[],
            'target_attention_mask':[],
            'value_input_ids':[],
            'value_attention_mask':[],
            'weights': [],
            'indices': [],  # Keep track for priority updates
        }

        for sample in samples:
            data['old_policy_log_probs'].append(sample.get('old_policy_log_probs', 0))
            data['advantage'].append(sample.get('advantage', 0))
            data['value_target'].append(sample.get('value_target', 0))
            data['input_ids'].append(sample.get('input_ids', 0))
            data['attention_mask'].append(sample.get('attention_mask', 0))
            data['target'].append(sample.get('target', 0))
            data['target_attention_mask'].append(sample.get('target_attention_mask', 0))
            data['value_input_ids'].append(sample.get('value_input_ids', 0))
            data['value_attention_mask'].append(sample.get('value_attention_mask', 0))
            data['weights'].append(weights)
            data['indices'].append(indices)

        dataset = Dataset.from_dict(data)
        return dataset

    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute the loss, incorporating importance-sampling weights."""

        # Compute policy loss using PPO
        new_policy_log_probs = forward_policy_predict(self.model, self.tokenizer, inputs)
        old_policy_log_probs = inputs['old_policy_log_probs']
        target_mask = inputs['target_attention_mask']
        advantage = inputs['advantage']
        epsilon = 0.2  # PPO clip parameter

        ratio = (new_policy_log_probs - old_policy_log_probs).exp() * target_mask[:,1:]
        surr1 = ratio * advantage.unsqueeze(-1)
        surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage.unsqueeze(-1)
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Compute value loss
        value_prediction = forward_value_predict(self.model, self.tokenizer, inputs)
        value_target = inputs['value_target']

        clamp_positive_rating_prob = torch.exp(torch.clamp(
            value_target, math.log(1e-6), 0
        ))
        clamp_negative_rating_prob = 1 - clamp_positive_rating_prob
        target_probs = torch.concat(
            [clamp_positive_rating_prob.unsqueeze(-1), clamp_negative_rating_prob.unsqueeze(-1)], dim=1
        )

        value_loss = F.binary_cross_entropy_with_logits(
            value_prediction, target_probs.to(self.accelerator.device)
        )


        # Combine losses
        total_loss = policy_loss + value_loss

        if total_loss == 0:
            return total_loss

        # Apply importance-sampling weights
        weights = torch.tensor(inputs['weights'], dtype=torch.float32).to(total_loss.device)
        total_loss = total_loss * weights
        td_error = total_loss.sum(dim=-1).detach().abs().cpu().numpy()
        total_loss = total_loss.mean()
        print(f'Policy Loss: {policy_loss}, Value Loss: {value_loss}, Total Loss: {total_loss}')
        if return_outputs:
            return total_loss, td_error
        else:
            return total_loss

    def update_priorities(self, indices, td_errors):
        """Update priorities in the replay buffer based on TD-errors."""
        new_priorities = td_errors + 1e-6  # Add small epsilon to avoid zero priorities
        self.replay_buffer.update_priorities(indices, new_priorities)

    def train(self, num_iterations, beta_start=0.4, beta_frames=100000, **kwargs):
        frame_idx = 1
        beta = beta_start
        for iteration in range(num_iterations):
            print(f"Starting iteration {iteration + 1}/{num_iterations}")

            # Self-play to collect new experiences
            initial_state = self.envoirment.sample_initial_state()
            root_node = self.self_play(initial_state)
            self.collect_experience(root_node)



            # Anneal beta over time to 1.0
            beta = min(1.0, beta_start + frame_idx * (1.0 - beta_start) / beta_frames)
            frame_idx += 1

            # Ensure enough data has been collected
            if len(self.replay_buffer.buffer) < self._train_batch_size:
                continue  # Skip training until we have enough data

            # Sample a batch from the replay buffer
            train_dataset = self.create_dataset_from_buffer(self._train_batch_size, beta=beta)

            if train_dataset is None:
                continue  # Not enough data to form a batch

            # Update the Trainer's dataset
            self.train_dataset = train_dataset
            self.data_collator = collator_fn
            
            # Create DataLoader
            train_dataloader = DataLoader(
                self.train_dataset,
                batch_size=self._train_batch_size,
                shuffle=True,  # PER handles sampling
                collate_fn=self.data_collator,
            )

            # Training loop
            for step, inputs in enumerate(train_dataloader):
                self.model.train()
                inputs = self._prepare_inputs(inputs)

                # Compute loss and perform backpropagation
                loss, td_errors = self.compute_loss(self.model, inputs, return_outputs=True)
                loss = loss.mean()
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()

                # Update priorities in the replay buffer
                # For simplicity, we use the absolute value of the loss as the TD-error
                indices = inputs['indices'].cpu().numpy()
                self.update_priorities(indices, td_errors)

            print(f"Iteration {iteration + 1}/{num_iterations} completed.")

        # Save the replay buffer at the end of training
        self.replay_buffer.save(self.replay_buffer_file)

import random
from grading import check

class Environment:
    def __init__(self, problems):
        """
        初始化环境。

        参数：
        - problems: 一个包含数学问题和答案的字典列表，每个字典包含 'problem' 和 'ground_truth' 键。
        """
        self.problems = problems
        self.num_problems = len(problems)
        self.inverse_mapping = {problem_declaration_template(problem['problem']): problem['ground_truth'] for problem in problems}

    def sample_initial_state(self):
        """
        从问题列表中随机采样一个初始状态（数学问题）。

        返回：
        - initial_state: 选中的问题文本。
        - ground_truth: 该问题的正确答案，用于后续的答案验证。
        """
        selected_problem = random.choice(self.problems)
        initial_state = problem_declaration_template(selected_problem['problem'])
        ground_truth = selected_problem['ground_truth']
        return initial_state, ground_truth

    def is_terminal_state(self, state, ground_truth):
        """
        判断当前状态是否为终止状态（正确答案）。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - is_terminal: 布尔值，表示是否为终止状态。
        """
        # 使用 compute_rule_orm_head 函数判断
        result = self.compute_rule_orm_head(state, ground_truth)
        return result
    
    def get_ground_truth(self, node):
        return self.inverse_mapping.get(get_root(node).state)

    # 判断状态是否为正确答案的函数
    def compute_rule_orm_head(self, node):
        """
        使用 grading 模块的 check 函数判断状态是否为正确答案。

        参数：
        - state: 当前状态文本。
        - ground_truth: 当前问题的正确答案。

        返回：
        - result: 布尔值，表示状态是否为正确答案。
        """
        # 将状态和正确答案传入 check 函数进行比较
        try:
            ground_truth = self.inverse_mapping.get(get_root(node).state)
            result = check(ground_truth, node.state, "")
            return result
        except:
            return False

from transformers import AutoModelForCausalLM, AutoTokenizer, AdamW
import torch

# 假设您已经定义了 TreeNode、MCTS 和 AlphaGoZeroForMath 类

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

# # 初始化优化器
# optimizer = AdamW(model.parameters(), lr=1e-4)



# 初始状态和 MCTS 参数
num_simulations = 32
num_candidates_per_expansion = 2
exploration_const = 1.4
discount_factor = 0.9
reward_epsilon = 1e-6

from datasets import load_dataset

ds = load_dataset("openai/gsm8k", "main")['train']

problems = [{"problem": p['question'], "ground_truth": p['answer']} for p in ds]

envoirment = Environment(problems)

# 创建 MCTS 实例
mcts = MCTS(
    envoirment=envoirment,
    model=model,
    tokenizer=tokenizer,
    num_simulations=num_simulations,
    num_candidates_per_expansion=num_candidates_per_expansion,
    exploration_const=exploration_const,
    discount_factor=discount_factor,
    reward_epsilon=reward_epsilon
)

# 创建 AlphaGoZeroForMath 实例
trainer = AlphaGoZeroForMath(
    envoirment=envoirment,
    model=model,
    tokenizer=tokenizer,
    mcts=mcts,
    replay_buffer_capacity=100000,
)

accelerator = trainer.accelerator
model = model.to(accelerator.device)

# 设置训练轮数和批次大小
num_iterations = 64

# 执行训练
trainer.train(num_iterations=num_iterations)
