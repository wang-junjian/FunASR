import gradio as gr
from funasr import AutoModel
import numpy as np
import os
import librosa
import torch


# --- 0. 设备检测逻辑 ---
def get_available_devices():
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    
    # 默认选择顺序：cuda > mps > cpu
    default_device = "cpu"
    if "cuda" in devices:
        default_device = "cuda"
    elif "mps" in devices:
        default_device = "mps"
    
    return devices, default_device

AVAILABLE_DEVICES, DEFAULT_DEVICE = get_available_devices()

# --- 1. 模型列表配置 ---
MODELS = {
    "Paraformer-zh (中文长语音)": "iic/speech_paraformer-large-vad-punc_asr_nat-zh-cn-16k-common-vocab8404-pytorch",
    "SenseVoiceSmall (多语言/情感/事件)": "iic/SenseVoiceSmall",
    "Nano (中/英/日)": "FunAudioLLM/Fun-ASR-Nano-2512",
    "MLT-Nano (多语言)": "FunAudioLLM/Fun-ASR-MLT-Nano-2512"
}

loaded_models = {}

def resample_audio(y, sr, target_sr=16000):
    if y.dtype == np.int16:
        y = y.astype(np.float32) / 32768.0
    elif y.dtype != np.float32:
        y = y.astype(np.float32)
    
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y

# --- 2. 实时识别核心逻辑 ---
class StreamASRService:
    def __init__(self):
        self.device = DEFAULT_DEVICE # 初始使用默认设备
        self.sample_rate = 16000
        self.chunk_size = [0, 10, 5] 
        self.chunk_stride = self.chunk_size[1] * 960 
        self.asr_model = None

    def load_model(self, device):
        """当用户切换设备时重新加载流式模型"""
        if self.asr_model is None or self.device != device:
            print(f"正在加载/切换流式模型至设备: {device}...")
            self.device = device
            self.asr_model = AutoModel(
                model="paraformer-zh-streaming", 
                device=self.device, 
                chunk_size_ms=600, 
                disable_pbar=True
            )

    def process_stream(self, audio_data, state, device):
        if audio_data is None:
            return "", state

        # 确保模型在当前选定设备上
        self.load_model(device)

        sr, y = audio_data
        if sr != self.sample_rate:
            y = resample_audio(y, sr, self.sample_rate)
        
        if y.ndim > 1:
            y = y.mean(axis=1)
        y = y.astype(np.float32)

        if state is None:
            state = {
                "asr_cache": {},
                "full_text": "",
                "buffer": np.array([], dtype=np.float32)
            }

        state["buffer"] = np.concatenate([state["buffer"], y])

        while len(state["buffer"]) >= self.chunk_stride:
            chunk = state["buffer"][:self.chunk_stride]
            state["buffer"] = state["buffer"][self.chunk_stride:]

            res = self.asr_model.generate(
                input=chunk,
                cache=state["asr_cache"],
                is_final=False,
                chunk_size=self.chunk_size,
                encoder_chunk_look_back=4,
                decoder_chunk_look_back=1,
            )

            if res and res[0].get('text'):
                state["full_text"] += res[0]['text']

        return state["full_text"], state

stream_service = StreamASRService()

# --- 3. 离线识别函数 ---
def transcribe_offline(audio_path, model_name, hotwords_str, device, use_itn, use_punc, use_speaker):
    if audio_path is None: return "请先上传音频。"
    
    hotwords_list = [line.strip() for line in hotwords_str.split('\n') if line.strip()]
    model_dir = MODELS[model_name]
    
    # 如果设备改变或模型未加载，重新加载
    cache_key = f"{model_dir}_{device}_{use_itn}_{use_punc}_{use_speaker}"
    if cache_key not in loaded_models:
        print(f"正在加载离线模型 {model_name} 至 {device}...")
        model_params = {"model": model_dir, "device": device, "trust_remote_code": True, "disable_pbar": True}
        if "Paraformer" in model_name:
            model_params["vad_model"] = "iic/speech_fsmn_vad_zh-cn-16k-common-pytorch"
            if use_punc:
                model_params["punc_model"] = "iic/punc_ct-transformer_cn-en-common-vocab471067-large"
            if use_speaker:
                model_params["spk_model"] = "iic/speech_campplus_sv_zh-cn_16k-common"
        loaded_models[cache_key] = AutoModel(**model_params)

    model = loaded_models[cache_key]
    
    gen_kwargs = {"input": audio_path, "language": "auto", "use_itn": use_itn, "itn": use_itn, "hotwords": hotwords_list}
    res = model.generate(**gen_kwargs)
    
    if res:
        result = res[0]
        if use_speaker and "sentence_info" in result:
            formatted_text = ""
            last_spk = None
            for item in result["sentence_info"]:
                speaker_id = item.get("spk", "未知")
                text = item['text']
                if speaker_id != last_spk:
                    if formatted_text:
                        formatted_text += "\n\n"
                    formatted_text += f"😀【{speaker_id}】{text}"
                    last_spk = speaker_id
                else:
                    formatted_text += text
            return formatted_text.strip()
        return result.get("text", "无识别结果")
    
    return "识别失败"

# --- 4. 构建 Gradio 界面 ---
custom_css = """
#text_output textarea, #stream_output textarea {
    height: 65vh !important;
}
footer { display: none !important; }
"""

with gr.Blocks(title="FunASR 综合语音识别工具", css=custom_css) as demo:
    gr.Markdown("# 🎙️ FunASR 实时/离线 语音识别")

    # 全局设置区域
    with gr.Accordion("⚙️ 全局设置", open=False):
        with gr.Row():
            device_selector = gr.Dropdown(
                choices=AVAILABLE_DEVICES, 
                value=DEFAULT_DEVICE, 
                label="计算设备 (已自动识别最佳选项)"
            )
        with gr.Row():
            use_itn = gr.Checkbox(label="开启 ITN (数字转写)", value=True)
            use_punc = gr.Checkbox(label="开启标点预测", value=True)
            use_speaker = gr.Checkbox(label="开启说话人识别", value=False)

    with gr.Tabs():
        with gr.TabItem("离线语音识别"):
            with gr.Row():
                with gr.Column():
                    audio_input = gr.Audio(sources=["upload", "microphone"], type="filepath", label="上传音频")
                    model_selector = gr.Dropdown(choices=list(MODELS.keys()), value=list(MODELS.keys())[0], label="选择模型")
                    hotwords_input = gr.Textbox(label="热词 (每行一个)", placeholder="阿里巴巴\n人工智能", lines=3)
                    submit_btn = gr.Button("开始识别", variant="primary")
                with gr.Column():
                    text_output = gr.Textbox(label="识别结果", show_copy_button=True, elem_id="text_output")

        with gr.TabItem("实时语音识别"):
            gr.Markdown("实时模式使用 `paraformer-zh-streaming`。")
            with gr.Row():
                with gr.Column():
                    # 设置 streaming=True，time_limit 决定回调频率（秒）
                    stream_input = gr.Audio(sources=["microphone"], streaming=True, label="麦克风输入")
                    clear_btn = gr.Button("清空记录")
                with gr.Column():
                    stream_output = gr.Textbox(label="实时识别内容", show_copy_button=True, elem_id="stream_output")
            
            stream_state = gr.State()

    # 事件绑定 - 离线
    submit_btn.click(
        fn=transcribe_offline, 
        inputs=[audio_input, model_selector, hotwords_input, device_selector, use_itn, use_punc, use_speaker], 
        outputs=text_output
    )

    # 事件绑定 - 实时
    stream_input.stream(
        fn=stream_service.process_stream,
        inputs=[stream_input, stream_state, device_selector],
        outputs=[stream_output, stream_state],
        show_progress="hidden"
    )

    # 清空按钮功能
    def reset_state():
        return "", None
    clear_btn.click(fn=reset_state, outputs=[stream_output, stream_state])

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        debug=True,
    )
