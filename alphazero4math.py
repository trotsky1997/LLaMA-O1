# Imports and Model Initialization
from email import policy
import random
from anyio import value
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from transformers import StoppingCriteria, StoppingCriteriaList
from typing import List
import torch

# Load Model and Tokenizer
model_name = "/mnt/hwfile/ai4chem/CKPT/longcot_pt_GEMMA_ZD_10_23_1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="auto", torch_dtype=torch.bfloat16
)
select_prefix = ""
meta_action_types = ["<problem>", "<critic>", "<refine>", "<conclusion>"]
meta_action_types_weight = [0.2, 0.4, 0.4, 0.3]

GT = "Natalia sold 48/2 = <<48/2=24>>24 clips in May. Natalia sold 48+24 = <<48+24=72>>72 clips altogether in April and May. #### 72"

# hint = f'<hint> Try generate a reasonable rationale step to approching The true final answer, "{GT}" </hint>'
hint = ''

hint_for_divide_and_conquer = f'<hint> Try divide the problem into smaller parts and solve them separately. </hint>'


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
        self.policy_cal_ready_texts = ""
        self.value_cal_ready_texts = ""
        self.true_value_from_tree = None
        self.leaf_type = ""
        self.rectify_visits = 0

    def add_child(self, child_node):
        self.children.append(child_node)

    def is_leaf(self):
        return len(self.children) == 0

    def get_child_policy_prob(self, child):
        # 提取logit值并转换为数组
        logits = np.array(list(self.policy.values()))

        # 计算Softmax概率
        exp_logits = np.exp(logits - np.max(logits))  # 减去最大logit值以增强数值稳定性
        softmax_probs = exp_logits / np.sum(exp_logits)

        # 构建新的字典，将键与Softmax概率对应
        return {key: prob for key, prob in zip(self.policy.keys(), softmax_probs)}[
            child
        ]


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


def hard_orm_head_template(selected_node):
    return f"For problem {get_root(selected_node).state}, the answer is {selected_node.state}, is it true?"


def soft_orm_head_template(selected_node, gt):
    return f"For problem {get_root(selected_node).state}, the ground truth is {gt}, the predicted answer is {selected_node.state}, is it the predicted answer true?"


selection_head_stopping_criteria = ["<end_of_father_id>"]

policy_head_stopping_criteria = ["<end_of_thought>"]

value_head_stopping_criteria = ["<end_of_rating>"]


def length_normed_logit(sequence_ids, scores_tensor):

    # Gather the log probabilities using advanced indexing
    # For each sequence and each position, select the log probability of the generated token
    selected_logits = torch.gather(
        scores_tensor.transpose(0, 1), 2, sequence_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Sum the log probabilities for each sequence
    summed_log_probs = selected_logits.sum(dim=1)

    # Normalize by sequence length
    normalized_log_probs = summed_log_probs / sequence_ids.size(1)

    return normalized_log_probs


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
def meta_compute_policy_head(selected_node, num_candidates=3):
    if random.random() > 0.25:
        text, probs = compute_policy_head(selected_node, num_candidates)
        return text, probs.tolist()

    metas = random.choices(
        meta_action_types, meta_action_types_weight, k=num_candidates
    )
    generated_texts = []
    policy_probs = []
    for i, meta in enumerate(metas):
        texts, policy_probs_0 = compute_policy_head(
            selected_node, num_candidates=1, meta=meta
        )
        generated_texts.append(texts[0])
        policy_probs.append(policy_probs_0.tolist()[0])
    return generated_texts, torch.tensor(policy_probs).tolist()


# Optimized Compute Policy Head (p(a|s))
@torch.no_grad()
def compute_policy_head(selected_node, num_candidates=3, meta=""):
    if meta != "<conclusion>":
        if meta == '<problem>':
            inputs_string = policy_head_template(
                selected_node, get_max_node_id_in_tree(selected_node) + 1, meta, hint_for_divide_and_conquer
            )
        else:
            inputs_string = policy_head_template(
                selected_node, get_max_node_id_in_tree(selected_node) + 1, meta, hint
            )
    else:
        inputs_string = policy_head_template(
            selected_node, get_max_node_id_in_tree(selected_node) + 1, meta, ""
        )
    inputs = tokenizer(inputs_string, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
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

    normalized_logits = length_normed_logit(output_ids[:, -logits.shape[0] :], logits)

    generated_texts = tokenizer.batch_decode(
        output_ids[:, -logits.shape[0] :], skip_special_tokens=False
    )

    generated_texts = [meta + clean_generated_text(text) for text in generated_texts]

    if meta != "<conclusion>":
        for i,generated_text in enumerate(generated_texts):
            if "The answer is:" in generated_text:
                generated_texts[i] = '<conclusion>' + generated_texts[i]

    return generated_texts, normalized_logits


@torch.no_grad()
def compute_value_head(selected_node):
    if selected_node.value != 0:
        return selected_node.value
    # 将state传入tokenizer并获取模型输入
    inputs_string = value_head_template(selected_node)
    inputs = tokenizer(inputs_string, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 使用模型生成outputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=2,
        # do_sample=True,
        # num_return_sequences=1,
        output_logits=True,
        return_dict_in_generate=True,
        stop_strings=value_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    # 获取生成的序列
    output_ids = outputs.sequences

    # 提取logits
    if isinstance(outputs.logits, tuple):
        logits = torch.stack(outputs.logits, dim=0)
    else:
        logits = outputs.logits  # 如果 logits 已是张量

    # 定位倒数第二个位置的logit
    if logits.size(0) < 1:
        raise ValueError("生成序列太短，无法提取倒数第二个位置的logit。")

    last_second_logits = logits[-1, :, :]  # 倒数第二个位置的logit

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("<positive_rating>")
    negative_token_id = tokenizer.convert_tokens_to_ids("<negative_rating>")

    # 提取 "<positive_rating>" 和 "<negative_rating>" 的logit
    positive_logit = last_second_logits[:, positive_token_id]
    negative_logit = last_second_logits[:, negative_token_id]

    # 将两者结合计算 value
    value = positive_logit - negative_logit  # 或根据需要使用其他计算方式

    return torch.tanh(value).item()


def compute_soft_orm_head(selected_node):
    # 将state传入tokenizer并获取模型输入
    inputs_string = soft_orm_head_template(selected_node, GT)
    inputs = tokenizer(inputs_string, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    # 使用模型生成outputs
    outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        # do_sample=True,
        # num_return_sequences=1,
        output_logits=True,
        return_dict_in_generate=True,
        stop_strings=value_head_stopping_criteria,
        tokenizer=tokenizer,
    )

    # 获取生成的序列
    output_ids = outputs.sequences

    # 提取logits
    if isinstance(outputs.logits, tuple):
        logits = torch.stack(outputs.logits, dim=0)
    else:
        logits = outputs.logits  # 如果 logits 已是张量

    # 定位倒数第二个位置的logit
    if logits.size(0) < 1:
        # print(logits.shape)
        raise ValueError("生成序列太短，无法提取倒数第二个位置的logit。")

    last_second_logits = logits[-1, :, :]  # 倒数第二个位置的logit

    # 获取 "<positive_rating>" 和 "<negative_rating>" 的token ID
    positive_token_id = tokenizer.convert_tokens_to_ids("True")
    negative_token_id = tokenizer.convert_tokens_to_ids("False")

    # 提取 "<positive_rating>" 和 "<negative_rating>" 的logit
    positive_logit = last_second_logits[:, positive_token_id]
    negative_logit = last_second_logits[:, negative_token_id]

    return positive_logit > negative_logit


from grading import check


def compute_rule_orm_head(selected_node):
    # 将state传入tokenizer并获取模型输入
    result = check(GT, selected_node.state, "")

    return result


# MCTS Search
class MCTS:
    def __init__(self, model, tokenizer, num_simulations=-1, exploration_const=1.0):
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations if num_simulations != -1 else 999
        self.exploration_const = exploration_const
        self.end = False
        self.discount_factor = 0.7

    def search(self, root_node):
        if not root_node.children:
            root_node.value = compute_value_head(root_node)

        for _ in range(self.num_simulations):
            self.simulate(root_node)
            if self.end:
                break

        for leaf in self.identify_leaf(root_node):
            if leaf.leaf_type == "successful":
                self.rectify_values_from_leaf(leaf, 1)
            else:
                self.rectify_values_from_leaf(leaf, -1)

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
        return node.value

    def expand_node(self, node):
        texts, policy_probs = meta_compute_policy_head(node)

        for i, text in enumerate(texts):
            child_node = TreeNode(
                state=text, parent=node, index=get_max_node_id_in_tree(node) + 1
            )
            # child_node.policy = policy_probs[i]
            node.policy[child_node] = policy_probs[i]
            node.add_child(child_node)
            child_node.value = self.compute_value(child_node)
            if "<conclusion>" in child_node.state:
                if compute_rule_orm_head(child_node):
                    self.end = True
                    child_node.leaf_type = "successful"
                else:
                    child_node.leaf_type = "failed"
            print(f"Id:{child_node.index}, Child: {text}, Policy: {policy_probs[i]}, Value: {child_node.value}")
        return max(child_node.value for child_node in node.children)

    def compute_value(self, node):
        # Use the model to predict the value of the current state
        value = compute_value_head(node)
        node.value = value
        return value

    def select_action(self, node):
        total_visits = sum(child.visits for child in node.children)
        ucb_scores = [
            (
                child.value
                + self.exploration_const
                * node.get_child_policy_prob(child)
                * np.sqrt(total_visits)
                / (1 + child.visits)
            )
            for child in node.children
        ]
        return node.children[np.argmax(ucb_scores)]

    # def get_policy_from_visits(self, node):
    #     total_visits = sum(child.visits for child in node.children)
    #     policy = {
    #         i: child.visits / total_visits for i, child in enumerate(node.children)
    #     }
    #     return policy

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
        if node.leaf_type == "successful" and node.is_leaf():
            node.true_value_from_tree = value
            # node.value = value
            if node.parent:
                self.rectify_values_from_leaf(
                    node.parent, node.true_value_from_tree * self.discount_factor
                )
        elif node.leaf_type == "failed" and node.is_leaf():
            node.true_value_from_tree = value
            # node.value = value
            if node.parent:
                self.rectify_values_from_leaf(
                    node.parent, node.true_value_from_tree * self.discount_factor
                )
        elif not node.is_leaf():
            if not node.true_value_from_tree:
                # node.value = value
                node.true_value_from_tree = value
            else:
                node.true_value_from_tree += (
                    value - node.true_value_from_tree
                ) / node.rectify_visits
                # node.true_value_from_tree = 1 if node.value >= 0 else -1
            if node.parent:
                self.rectify_values_from_leaf(
                    node.parent, node.true_value_from_tree * self.discount_factor
                )


def training_time_predict(node):
    # Use the model to predict the value of the current state
    text_for_policy = policy_head_template(node.parent, node.index) + node.state
    inputs = tokenizer(text_for_policy, return_tensors="pt")
    target = tokenizer(node.state, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    target = {k: v.to(model.device) for k, v in target.items()}
    outputs = model(**inputs, return_dict=True)
    logits = outputs.logits
    # print(target["input_ids"].shape,logits.shape)
    normed_logit = length_normed_logit(
        target["input_ids"].transpose(0, 1), logits[:,-target["input_ids"].shape[-1] :]
    )

    text_for_value = value_head_template(node) + value_to_rating_token(node.value)
    inputs = tokenizer(text_for_value, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
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

    return normed_logit, value_logit


# Model Training using MCTS policy and value
criterion_policy = torch.nn.CrossEntropyLoss()
# criterion_value = torch.nn.MSELoss()
criterion_value = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=5e-5)


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

def ppo_loss(old_probs, new_probs, advantages, kl_penalty_coeff=0.1, desired_kl=0.01, epsilon=0.2):
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

    # 计算KL散度
    kl_divergence = old_probs * (torch.log(old_probs + 1e-8) - torch.log(new_probs + 1e-8))
    kl_divergence = kl_divergence.sum(dim=-1).mean()
    # print(f"KL Divergence: {kl_divergence}")

    # 动态调整KL惩罚系数
    if kl_divergence > desired_kl * 1.5:
        kl_penalty_coeff *= 2  # 增大惩罚
        # print(f"Increasing KL Penalty Coefficient to {kl_penalty_coeff}")
    elif kl_divergence < desired_kl / 1.5:
        kl_penalty_coeff *= 0.5  # 减少惩罚
        # print(f"Decreasing KL Penalty Coefficient to {kl_penalty_coeff}")

    # 总损失 = Clipped PPO 损失 - KL 惩罚项
    ppo_loss = -torch.min(unclipped_loss, clipped_loss).mean()
    total_loss = ppo_loss + kl_penalty_coeff * kl_divergence

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


def train_model_with_ppo_gae(root_node, optimizer, epsilon=0.2, gamma=0.99, lam=0.95):
    optimizer.zero_grad()
    policy_loss = 0
    value_loss = 0
    # regularization_loss = 0

    # 存储所有 episode 的奖励和价值预测
    rewards = []
    values = []

    # 遍历所有节点获取 rewards 和 value
    for node in traverse_tree(root_node):
        # _, predicted_value = model.predict(node.state)
        rewards.append(
            node.true_value_from_tree
            if node.true_value_from_tree != None
            else node.value
        )  # 假设 true_value 存储真实的奖励
        values.append(node.value)

    # 计算 GAE
    advantages = compute_gae(rewards, values, gamma, lam)
    # print(f"Advantages: {advantages}")

    # 遍历节点以应用 PPO 和 GAE
    for node, advantage in zip(traverse_tree(root_node), advantages):
        # 获取当前策略和价值估计
        predicted_policy, predicted_value = training_time_predict(node)

        # # 目标策略 (MCTS结果)
        # actual_policy = {child: child.visits / node.visits for child in node.children if node.visits > 0}
        # target_policy_probs = torch.tensor([actual_policy.get(child, 0) for child in node.children], dtype=torch.float32)

        if len(node.children) > 0 and node.rectify_visits > 0:
            # 旧策略：从节点缓存中获取，计算旧策略的概率
            old_policy_probs = torch.tensor(
                [node.policy[child] for child in node.children], dtype=torch.float32
            ).softmax(dim=0)

            # PPO 策略损失 (使用 GAE)
            new_policy_probs = torch.tensor(
                [predicted_policy[child.index] for child in node.children],
                dtype=torch.float32,
            ).softmax(dim=0)

            ppo_loss_0 = ppo_loss(
                old_policy_probs, new_policy_probs, torch.tensor(advantage), epsilon
            )
            # print(f"PPO Loss: {ppo_loss_0}")
            policy_loss += ppo_loss_0

        # 价值损失
        true_value = (
            node.true_value_from_tree
            if node.true_value_from_tree != None
            else node.value
        )
        true_value_one_hot = torch.tensor(
            [1 if true_value >= 0 else 0],
            dtype=torch.long,device=predicted_value.device
        )
        value_loss += criterion_value(predicted_value.softmax(dim=0), true_value_one_hot)

        # # L2 正则化损失
        # for param in model.parameters():
        #     regularization_loss += torch.norm(param) ** 2

    # 总损失
    total_loss = policy_loss + value_loss
    print(f"Total Loss: {total_loss}, Policy Loss: {policy_loss}, Value Loss: {value_loss}")
    total_loss.backward()
    optimizer.step()

    return total_loss.item()


# Self-play and Training Loop
class AlphaGoZeroForMath:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.mcts = MCTS(model, tokenizer)

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
        loss = train_model_with_ppo_gae(root_node, optimizer)
        print(f"Training Loss: {loss}")


# Running Training
initial_state = problem_declaration_template(
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?"
)
agent = AlphaGoZeroForMath(model, tokenizer)
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
