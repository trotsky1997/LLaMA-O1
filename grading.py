from functools import lru_cache
import re

from numpy import extract
import openai
from openai_grading.grader import grade_answer as openai_is_equiv
from openai_grading.math_normalize import normalize_answer

def last_boxed_only_string(string):
    idx = string.rfind('\\boxed')
    if idx < 0:
        idx = string.rfind('\\fbox')
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == '{':
            num_left_braces_open += 1
        if string[i] == '}':
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]

    return retval


def remove_boxed(s):
    left = '\\boxed{'
    try:
        assert s[:len(left)] == left
        assert s[-1] == '}'
        return s[len(left):-1]
    except Exception:
        return None


def extract_boxed_answer(pred_str, strip_double_curly_brace=False):
    boxed_str = last_boxed_only_string(pred_str)
    if boxed_str is None:
        return None
    answer = remove_boxed(boxed_str)
    if answer is None:
        return None
    if strip_double_curly_brace:
        match = re.match('^\{(.*)\}$', answer)  # noqa: W605
        if match:
            answer = match.group(1)
    return answer


class Extractor:

    def extract_matching_bracket(cls, target_str: str):
        if not target_str:
            return target_str
        current_nest_level = 1
        for i, ch in enumerate(target_str):
            if ch == '{':
                current_nest_level += 1
            elif ch == '}':
                current_nest_level -= 1
            if current_nest_level == 0:
                break
        return target_str[:i]

    def clean(cls, target_str: str):
        opt = target_str.strip().replace('{{', '{').replace('}}', '}')
        if not opt:
            return opt
        if opt[-1] == '.' or opt[-1] == '。':
            return opt[:-1]
        return opt

    def extract_answer(cls, pred: str, extract_last_num=False):
        if pred.find('The final answer is ') >= 0:
            x = pred[pred.find('The final answer is ') +
                     len('The final answer is '):]
            x = x[1:x.find('$.')]
            # print(x)
            return cls.clean(x)
        if pred.find('\n\nQuestion:') >= 0:
            pred = pred.split('\n\nQuestion:')[0]
            if pred.find('The answer is'):
                pred = pred[pred.find('The answer is') + len('The answer is'):]
                return cls.clean(pred)
        if pred.find('# Answer') >= 0:
            return cls.clean(pred[pred.find('# Answer') + len('# Answer'):])
        if pred.find('The answer is:') >= 0:
            return cls.clean(pred[pred.find('The answer is:') +
                                  len('The answer is:'):])
        if pred.find('####') >= 0:
            return cls.clean(pred[pred.find('####') + 4:])
        left = '\\boxed{'
        if pred.find(left) >= 0:
            pred = pred[pred.find(left) + len(left):]
            return cls.clean(cls.extract_matching_bracket(pred))

        if extract_last_num:
            nums = []
            opt = ''

            def contain_digit(opt):
                for ch in opt:
                    if ch.isdigit():
                        return True
                return False

            for ch in pred:
                if ch.isdigit() or ch in ' ,.':
                    opt = opt + ch
                else:
                    if contain_digit(opt):
                        nums.append(opt)
                    opt = ''
            if contain_digit(opt):
                return cls.clean(opt)
            if nums:
                return cls.clean(nums[-1])
        return None


def fix_fracs(string):
    substrs = string.split('\\frac')
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += '\\frac'
            if substr[0] == '{':
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != '{':
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}{' + b + '}' + post_substr
                    else:
                        new_str += '{' + a + '}{' + b + '}'
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += '{' + a + '}' + b + post_substr
                    else:
                        new_str += '{' + a + '}' + b
    string = new_str
    return string


def fix_a_slash_b(string):
    if len(string.split('/')) != 2:
        return string
    a = string.split('/')[0]
    b = string.split('/')[1]
    try:
        a = int(a)
        b = int(b)
        assert string == '{}/{}'.format(a, b)
        new_string = '\\frac{' + str(a) + '}{' + str(b) + '}'
        return new_string
    except AssertionError:
        return string


def remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set)
    if '\\text{ ' in string:
        splits = string.split('\\text{ ')
        assert len(splits) == 2
        return splits[0]
    else:
        return string


def fix_sqrt(string):
    if '\\sqrt' not in string:
        return string
    splits = string.split('\\sqrt')
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != '{':
            a = split[0]
            new_substr = '\\sqrt{' + a + '}' + split[1:]
        else:
            new_substr = '\\sqrt' + split
        new_string += new_substr
    return new_string


def strip_string(string):
    # linebreaks
    string = string.replace('\n', '')

    # remove inverse spaces
    string = string.replace('\\!', '')

    # replace \\ with \
    string = string.replace('\\\\', '\\')

    # replace tfrac and dfrac with frac
    string = string.replace('tfrac', 'frac')
    string = string.replace('dfrac', 'frac')

    # remove \left and \right
    string = string.replace('\\left', '')
    string = string.replace('\\right', '')

    # Remove circ (degrees)
    string = string.replace('^{\\circ}', '')
    string = string.replace('^\\circ', '')

    # remove dollar signs
    string = string.replace('\\$', '')

    # remove units (on the right)
    string = remove_right_units(string)

    # remove percentage
    string = string.replace('\\%', '')
    string = string.replace('\%', '')  # noqa: W605

    string = string.replace(' .', ' 0.')
    string = string.replace('{.', '{0.')
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == '.':
        string = '0' + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split('=')) == 2:
        if len(string.split('=')[0]) <= 2:
            string = string.split('=')[1]

    # fix sqrt3 --> sqrt{3}
    string = fix_sqrt(string)

    # remove spaces
    string = string.replace(' ', '')

    string = fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == '0.5':
        string = '\\frac{1}{2}'

    string = fix_a_slash_b(string)
    string = string.replace('x \\in', '').strip()  # noqa: W605

    # a_b == a, a_{b} == a_b for bit conversion
    if string.find('_') >= 0:
        p = string.split('_')
        p[1] = p[1].replace('{', '').replace('}', '')
        string = '_'.join(p)

    # 10800 == 10,800; we only deal with single number
    if string.strip().find(' ') == -1 and string.find('(') == -1:
        string = string.replace(',', '')

    return string


def is_equiv(str1, str2, verbose=False):
    if str1 is None and str2 is None:
        # print("WARNING: Both None")
        return False
    if str1 is None or str2 is None:
        return False

    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2



pattern = re.compile(r'\-?\d+\.\d+|\-?\d+')
extractor_0 = Extractor()
@lru_cache(1024)
def org_extract_label(text: str,DATA_NAME,type='') -> str:
    # print(text)
    if isinstance(text, float) or isinstance(text, int):
        return str(text)

    # if type != 'digit':

    if type == 'digit':
        if 'The answer is' in text:
            text = text.split('The answer is')[-1].replace(',','')
            if '####' in text:
                text = text.split('####')[0]
        elif '####' in text:
            text = text.split('####')[-1].replace(',','')
        numbers = pattern.findall(text)
        if numbers:
            if '####' in text or 'The answer is' in text:
                return numbers[0]
            else :
                return numbers[-1]
    
    if 'The answer is' in text:
        text = text.split('The answer is')[-1]
        if '####' in text:
            text = text.split('####')[0]
    elif '####' in text:
        text = text.split('####')[-1]
    if '\\box' in text:
        return extract_boxed_answer(text)
    
    extracted = extractor_0.extract_answer(text, extract_last_num=True)
    if extracted:
        return extracted
    if '$' in text or '\\(' in text or '\\[' in text:
        if '$$' in text:
            extracted = text.split('$$')[-2]
        elif '\\[' in text:
            extracted = text.split('\\[')[-1].split('\\]')[0]
        elif '\\(' in text:
            extracted = text.split('\\(')[-1].split('\\)')[0]
        elif '$' in text and text.count('$') % 2 == 0 :
            extracted = text.split('$')[-2]
        if extracted:
            return extracted
    
    return text
    

def post_clean(ans_label):
    if ans_label.startswith('$$') and ans_label.endswith('$$'):
        ans_label = ans_label[2:-2]
    elif ans_label.startswith('$') and ans_label.endswith('$'):
        ans_label = ans_label[1:-1]
    elif ans_label.startswith('\\[') and ans_label.endswith('\\]'):
        ans_label = ans_label[2:-2]
    elif ans_label.startswith('\\(') and ans_label.endswith('\\)'):
        ans_label = ans_label[2:-2]
    elif ans_label.startswith('='):
        ans_label = ans_label[1:]
    elif ans_label.endswith('.0'):
        ans_label = ans_label[:-2]
    return ans_label

def extract_label(text: str,DATA_NAME,type='') -> str:
    ans_label = org_extract_label(text,DATA_NAME,type)
    if ans_label is not None:
        ans_label = post_clean(ans_label)
    else:
        print('Failed to extract label from:')
    return ans_label

def check_label(gt_label,ans_label):
    simple_criteria = is_equiv(gt_label,ans_label)
    openai_criteria = openai_is_equiv(ans_label,gt_label)
    # print(gt_label,ans_label)
    literal_criteria_0 = gt_label is None or ans_label is None

    literal_criteria_1 = ans_label == gt_label
    try: 
        value_criteria = abs(float(eval(ans_label)) - float(eval(gt_label))) < 1e-5
    except:
        value_criteria = False
    

    # print(simple_criteria,openai_criteria,not literal_criteria_0,literal_criteria_1)

    return simple_criteria or literal_criteria_1 or value_criteria or openai_criteria and not literal_criteria_0

@lru_cache(1024)
def check(gt,ans,DATA_NAME):
    gt_label = extract_label(gt,DATA_NAME)
    if gt_label.isdigit():
        type = 'digit'
    elif gt_label.isupper() and gt_label.isalpha():
        type = 'option'
    elif gt_label.lower() in ['yes','no','true','false']:
        gt_label = gt_label.lower()
        type = 'yesorno'
    else :
        type = 'formula'
    ans_label = extract_label(ans,DATA_NAME,type)
    if ans_label:
        if type == 'option':
            ans_label = ans_label.strip()[0]
        elif type == 'yesorno':
            ans_label = ans_label.lower()
        elif type == 'formula':
            ans_label = ans_label.replace('$','')
    # print(gt_label,ans_label)
    # print(type)
    return check_label(gt_label,ans_label)