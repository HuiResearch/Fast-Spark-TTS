# -*- coding: utf-8 -*-
# Time      :2025/3/29 11:32
# Author    :Hui Huang
import asyncio
from typing import Callable, Optional
import regex
import re
from ..logger import get_logger

logger = get_logger()


def limit_concurrency(semaphore: asyncio.Semaphore):
    def decorator(func):
        async def wrapped(*args, **kwargs):
            async with semaphore:  # 在这里限制并发请求数
                return await func(*args, **kwargs)

        return wrapped

    return decorator


def contains_chinese(s: str) -> bool:
    """
    判断字符串中是否包含中文字符
    """
    return bool(re.search(r'[\u4e00-\u9fff]', s))


# 以下代码从cosyvoice项目copy的
def is_only_punctuation(text):
    # Regular expression: Match strings that consist only of punctuation marks or are empty.
    punctuation_pattern = r'^[\p{P}\p{S}]*$'
    return bool(regex.fullmatch(punctuation_pattern, text))


def replace_corner_mark(text):
    text = text.replace('²', '平方')
    text = text.replace('³', '立方')
    return text


# remove meaningless symbol
def remove_bracket(text):
    text = text.replace('（', '').replace('）', '')
    text = text.replace('【', '').replace('】', '')
    text = text.replace('`', '').replace('`', '')
    text = text.replace("——", " ")
    return text


def text_normalize(text: str) -> str:
    if contains_chinese(text):
        text = text.replace("\n", "")
        text = replace_corner_mark(text)
        text = text.replace(".", "。")
        text = text.replace(" - ", "，")
        text = remove_bracket(text)
        text = re.sub(r'[，,、]+$', '。', text)
        if text[-1] not in ['。', '？', '！', '；', '：', '、', '.', '?', '!', ';']:
            text += "。"
    return text


def split_text(
        text: str,
        window_size: int,
        tokenize_fn: Callable[[str], list[str]],
        split_fn: Optional[Callable[[str], list[str]]] = None,
        length_threshold: int = 50,
) -> list[str]:
    """
    将长文本拆分成多个片段。首先使用中英文句号、问号、感叹号等切分文本，
    然后根据传入的窗口大小将切分后的句子合并成不超过窗口大小的片段。
    如果单个句子长度超过窗口大小，则会对该句子进行切割。

    :param text: 输入的长文本
    :param window_size: 每个片段的最大长度
    :param tokenize_fn: 分词函数
    :param split_fn: 片段切分方法，如果传入就使用自定义切分函数
    :param length_threshold: 长度阈值，超过这个值将进行切分
    :return: 切分后的文本片段列表
    """
    text = text.strip()
    text = text_normalize(text)
    is_chinese = contains_chinese(text)
    if len(tokenize_fn(text)) <= length_threshold:
        return [text]

    if split_fn is None:
        sentences = re.split(r'(?<=[。？！；;.!?：:])', text)
        # 去除拆分过程中产生的空字符串，并去除两侧空白
    else:
        sentences = split_fn(text)

    sentences = [s.strip() for s in sentences if s.strip()]
    segments = []
    current_segment = ""
    current_length = 0
    for sentence in sentences:
        sent_len = len(tokenize_fn(sentence))
        if sent_len > window_size:
            segments.append(current_segment)
            segments.append(sentence)  # 不进一步细分
            current_segment = ""
            current_length = 0
        else:
            if current_length + sent_len > window_size:
                segments.append(current_segment)
                current_segment = sentence
                current_length = sent_len
            else:
                current_length += sent_len
                if is_chinese:
                    current_segment += sentence
                else:
                    current_segment += " " + sentence
    if current_segment:
        segments.append(current_segment)
    return [seg for seg in segments if not is_only_punctuation(seg)]


def parse_multi_speaker_text(text, speakers):
    """
    解析文本，将文本分割为多个角色及其对应的台词。
    支持可选的 pitch 和 speed 属性。

    参数:
    text (str): 待解析的文本，文本中各段台词前以 <role:角色名,pitch:值,speed:值> 标识。
    speakers (list): 允许的角色名称列表，只有在列表中的角色会被解析。

    返回:
    list: 每个元素为一个字典，包含 'name'（角色名称）, 'text'（台词文本）,
         'pitch'（音调）和 'speed'（速度）。
    """

    # 允许的属性值
    valid_attributes = ["very_low", "low", "moderate", "high", "very_high"]

    # 使用正则表达式匹配标记和标记后的文本
    pattern = r'<role:([^,>]+)(?:,pitch:([^,>]+))?(?:,speed:([^,>]+))?>'
    parts = re.split(pattern, text)

    result = []
    # parts 的排列方式为：[前置文本, 角色1, pitch1, speed1, 台词1, 角色2, pitch2, speed2, 台词2, ...]

    i = 1
    while i < len(parts):
        if i + 3 >= len(parts):
            break

        role = parts[i].strip()
        pitch = parts[i + 1].strip() if parts[i + 1] else None
        speed = parts[i + 2].strip() if parts[i + 2] else None
        dialogue = parts[i + 3].strip()

        # 验证角色是否在允许列表中
        if role not in speakers:
            logger.warning(f"{role}并不在已有的角色列表（{', '.join(speakers)}）中，将跳过该角色的文本。")
            i += 4
            continue

        # 验证pitch和speed的值是否合法
        if pitch and pitch not in valid_attributes:
            logger.warning(f"属性pitch的值'{pitch}'不合法，将设为None。合法值为: {', '.join(valid_attributes)}")
            pitch = None

        if speed and speed not in valid_attributes:
            logger.warning(f"属性speed的值'{speed}'不合法，将设为None。合法值为: {', '.join(valid_attributes)}")
            speed = None

        if dialogue:
            result.append({
                "name": role,
                "text": dialogue,
                "pitch": pitch,
                "speed": speed
            })

        i += 4

    return result
