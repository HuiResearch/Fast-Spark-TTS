# -*- coding: utf-8 -*-
# Time      :2025/3/15 13:39
# Author    :Hui Huang

import requests
import base64

# 设置服务器地址
BASE_URL = "http://127.0.0.1:8000"


def generate_voice():
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"
    payload = {
        "text": text,
        "name": "male",
        "pitch": "moderate",
        "speed": "moderate",
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 2048
    }
    response = requests.post(f"{BASE_URL}/speak", json=payload)
    if response.status_code == 200:
        with open("generate_voice.mp3", "wb") as f:
            f.write(response.content)
        print("生成的音频已保存为 generate_voice.mp3")
    else:
        print("请求失败：", response.status_code, response.text)


def clone_with_base64():
    # 使用 base64 编码的参考音频
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

    reference_audio_path = "data/roles/赞助商/reference_audio.wav"  # 请替换为你本地的参考音频文件路径
    try:
        with open(reference_audio_path, "rb") as f:
            audio_bytes = f.read()
        # 将二进制音频数据转换为 base64 字符串
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("读取本地文件失败：", e)
        return

    payload = {
        "text": text,
        "reference_text": None,
        "reference_audio": audio_base64,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 2048
    }
    response = requests.post(f"{BASE_URL}/clone_voice", data=payload)
    if response.status_code == 200:
        with open("clone_voice.mp3", "wb") as f:
            f.write(response.content)
        print("克隆的音频已保存为 clone_voice.mp3")
    else:
        print("请求失败：", response.status_code, response.text)


def clone_with_file():
    # 使用 base64 编码的参考音频
    text = "身临其境，换新体验。塑造开源语音合成新范式，让智能语音更自然。"

    reference_audio_path = "data/roles/赞助商/reference_audio.wav"  # 请替换为你本地的参考音频文件路径

    payload = {
        "text": text,
        "reference_text": None,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 2048
    }
    response = requests.post(
        f"{BASE_URL}/clone_voice",
        data=payload,
        files={
            "reference_audio_file": open(reference_audio_path, "rb")
            # "latent_file": open(latent_file, "rb") # 如果是mega tts模型，这里需要增加一个npy文件
        })
    if response.status_code == 200:
        with open("clone_voice.mp3", "wb") as f:
            f.write(response.content)
        print("克隆的音频已保存为 clone_voice.mp3")
    else:
        print("请求失败：", response.status_code, response.text)


def clone_voice_stream():
    import pyaudio

    # 使用长文本检测是否及时返回
    long_text = ("今日是二零二五年三月十九日，国内外热点事件聚焦于国际局势、经济政策及社会民生领域。"
                 "国际局势中，某国领导人围绕地区冲突停火问题展开对话，双方同意停止攻击对方能源设施并推动谈判，但对全面停火提议的落实仍存分歧。"
                 "某地区持续军事行动导致数百人伤亡，引发民众抗议，质疑冲突背后的政治动机。另有一方宣称对连续袭击军事目标负责，称此为对前期打击的回应。"
                 "欧洲某国通过争议性财政草案，计划放宽债务限制以支持国防与环保项目，引发经济政策讨论。 "
                 "国内动态方面，新修订的市场竞争管理条例将于四月二十日施行，重点规范市场秩序。"
                 "多部门联合推出机动车排放治理新规，加强对高污染车辆的监管。"
                 "社会层面，某地涉及非法集资的大案持续引发关注，受害人数以万计，涉案金额高达数百亿元，暴露出特定领域投资风险。"
                 "经济与科技领域，某科技企业公布年度营收突破三千六百五十九亿元，并上调智能汽车交付目标至三十五万台。"
                 "另一巨头宣布全面推动人工智能转型，要求各部门绩效与人工智能应用深度绑定，计划年内推出多项相关产品。"
                 "充电基础设施建设加速，公共充电桩总量已接近四百万个，同比增长超六成。 "
                 "民生政策方面，多地推出新举措：某地限制顺风车单日接单次数以规范运营，另一地启动职工数字技能培训计划，目标三年内覆盖十万女性从业者。"
                 "整体来看，今日热点呈现国际博弈复杂化、国内经济科技加速转型、民生政策精准化调整的特点。")

    reference_audio_path = "data/roles/赞助商/reference_audio.wav"  # 请替换为你本地的参考音频文件路径
    try:
        with open(reference_audio_path, "rb") as f:
            audio_bytes = f.read()
        # 将二进制音频数据转换为 base64 字符串
        audio_base64 = base64.b64encode(audio_bytes).decode("utf-8")
    except Exception as e:
        print("读取本地文件失败：", e)
        return

    payload = {
        "text": long_text,
        "reference_text": None,
        "reference_audio": audio_base64,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 50,
        "max_tokens": 2048,
        "stream": True,
        "response_format": 'wav'
    }
    response = requests.post(f"{BASE_URL}/clone_voice", data=payload, stream=True)

    # 初始化 PyAudio
    p = pyaudio.PyAudio()

    # 根据接口返回的音频格式配置参数（此处假设返回的是16位单声道、采样率为16000的PCM数据）
    stream = p.open(
        format=pyaudio.paInt16,  # 16位音频
        channels=1,  # 单声道
        rate=16000,  # 采样率
        output=True
    )

    # 循环读取响应数据，并写入音频流进行播放
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            stream.write(chunk)

    # 播放结束后，关闭流
    stream.stop_stream()
    stream.close()
    p.terminate()


def openai_speech():
    """
    openai 接口请求
    Returns:

    """
    from openai import OpenAI

    client = OpenAI(
        base_url=f"{BASE_URL}/v1",
        api_key="not-needed"  # 如果设置了api key，请传入
    )
    with client.audio.speech.with_streaming_response.create(
            model="orpheus",
            voice="tara",
            input="Hey there guys. It's, <giggle> Tara here, and let me introduce you to Zac.. who seems to asleep. Zac, it's time to wakey-wakey!"
    ) as response:
        response.stream_to_file("output.mp3")


if __name__ == "__main__":
    clone_voice_stream()
