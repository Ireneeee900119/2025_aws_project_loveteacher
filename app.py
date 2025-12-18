import os
import requests
import re
import threading
import time
import random
import boto3
import opencc
from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from PIL import Image, ImageDraw, ImageFont 
from io import BytesIO

# 引入設定檔與記憶模組
import config
from memory_manager import MemoryManager

app = Flask(__name__)

# 初始化 AWS S3
s3_client = boto3.client('s3', region_name=config.AWS_REGION)

# 初始化 Line Bot
line_bot_api = LineBotApi(config.LINE_ACCESS_TOKEN)
handler = WebhookHandler(config.LINE_SECRET)
cc = opencc.OpenCC('s2twp')

# 初始化 RAG
print("正在載入向量資料庫...")
try:
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL_NAME,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
    vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
    print("✅ 本地 Embedding 模型與向量庫載入完成")
except Exception as e:
    print(f"❌ 資料庫載入失敗: {e}")

# ==============================================================================
# ComfyUI & 圖片處理
# ==============================================================================

def get_comfy_workflow(prompt_text):
    """生成 ComfyUI Workflow JSON"""
    therapeutic_prompt = f"{prompt_text}, therapeutic, soft lighting, pastel colors, cozy atmosphere, warm colors, perfect reflection, snowy mountains, golden hour, highly detailed, 8k"
    
    return {
        "3": {
            "class_type": "KSampler",
            "inputs": {
                "cfg": 8, "denoise": 1, "latent_image": ["5", 0], "model": ["4", 0], 
                "negative": ["7", 0], "positive": ["6", 0], "sampler_name": "euler", 
                "scheduler": "normal", "seed": random.randint(1, 9999999999), "steps": 20
            }
        },
        "4": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": "rev.safetensors"}},
        "5": {"class_type": "EmptyLatentImage", "inputs": {"batch_size": 1, "height": 512, "width": 768}},
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": therapeutic_prompt}}, 
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "text, watermark, low quality, worst quality, noise, blurry, deformed, mutated, tiling, poorly drawn, extra limbs"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "line_api", "images": ["8", 0]}}
    }

def check_history(prompt_id):
    start_time = time.time()
    while time.time() - start_time < 60:
        try:
            response = requests.get(f"{config.COMFY_BASE_URL}/history/{prompt_id}")
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        except Exception:
            time.sleep(2)
    raise TimeoutError("ComfyUI 圖片生成超時")

def get_image_bytes(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{config.COMFY_BASE_URL}/view", params=data)
    response.raise_for_status()
    return response.content

def add_text_overlay_bytes(image_bytes):
    quote = random.choice(config.INSPIRATIONAL_QUOTES)
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img)

        font_size = 40 
        try:
            font = ImageFont.load_default(size=font_size)
        except TypeError:
            font = ImageFont.load_default()

        # 自動換行與置中邏輯
        max_width = img.width - 80 
        lines = []
        words = quote.split(' ')
        current_line = ""

        for word in words:
            test_line = current_line + word + " "
            bbox = draw.textbbox((0, 0), test_line, font=font)
            text_width = bbox[2] - bbox[0]
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)

        line_height = font_size + 10
        total_text_height = len(lines) * line_height
        start_y = img.height - total_text_height - 50
        current_y = start_y

        for line in lines:
            bbox = draw.textbbox((0, 0), line, font=font)
            line_width = bbox[2] - bbox[0]
            x = (img.width - line_width) / 2
            
            draw.text((x, current_y), line, font=font, fill=(255,255,255), 
                      stroke_width=2, stroke_fill=(0,0,0))
            current_y += line_height

        output_buffer = BytesIO()
        img.convert("RGB").save(output_buffer, format="PNG")
        output_buffer.seek(0)
        return output_buffer.getvalue()

    except Exception as e:
        print(f"文字疊加失敗: {e}")
        return image_bytes

def upload_bytes_to_s3(image_bytes, s3_object_key):
    """上傳至 S3 並回傳 Presigned URL"""
    try:
        s3_client.put_object(Bucket=config.S3_BUCKET_NAME, Key=s3_object_key, Body=image_bytes, ContentType='image/png')
        url = s3_client.generate_presigned_url(
            ClientMethod='get_object',
            Params={
                'Bucket': config.S3_BUCKET_NAME, 
                'Key': s3_object_key
            },
            ExpiresIn=3600 
        )
        print(f"S3 上傳成功 (Presigned): {url}")
        return url
    except Exception as e:
        print(f"S3 上傳失敗: {e}")
        return None

def generate_and_upload_image(user_prompt):
    try:
        workflow = get_comfy_workflow(user_prompt)
        response = requests.post(config.COMFY_URL_ENDPOINT, json={"prompt": workflow})
        response.raise_for_status()
        prompt_id = response.json()['prompt_id']
        
        history = check_history(prompt_id)
        img_info = history['outputs']['9']['images'][0]
        original_bytes = get_image_bytes(img_info['filename'], img_info['subfolder'], img_info['type'])
        
        final_bytes = add_text_overlay_bytes(original_bytes)
        s3_key = f"{config.S3_IMG_FOLDER}/{int(time.time())}_{random.randint(100,999)}.png"
        return upload_bytes_to_s3(final_bytes, s3_key)
        
    except Exception as e:
        print(f"圖片生成流程錯誤: {e}")
        return None

# ==============================================================================
# 核心邏輯：LLM 與 RAG
# ==============================================================================

def to_traditional(text):
    if not text: return ""
    return cc.convert(text)

def clean_incomplete_sentence(text):
    if not text: return ""
    match = re.search(r'[。！？!?\n](?!.*[。！？!?\n])', text)
    if match: return text[:match.end()].strip()
    return text.strip() + ("..." if len(text) > 10 else "")

def call_llm_generic(prompt: str, max_tokens=400, temperature=0.7) -> tuple[str, str]:
    # 1. 優先 vLLM
    try:
        payload = {
            "model": config.VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens
        }
        response = requests.post(config.VLLM_CHAT_URL, json=payload, timeout=25)
        response.raise_for_status()
        content = response.json()['choices'][0]['message']['content']
        if content: return content.strip(), "vLLM"
    except Exception:
        pass 

    # 2. 備援 Ollama
    try:
        payload = {
            "model": config.OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "max_tokens": max_tokens}
        }
        response = requests.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=90)
        response.raise_for_status()
        result = response.json().get('response', '')
        if result: return result.strip(), "Ollama"
    except Exception as e:
        print(f"❌ LLM Total Failure: {e}")
    
    return "", "Fail"

def check_safety_risk(user_msg: str) -> bool:
    prompt = f"[System] 判斷訊息是否有自殺/自殘/暴力意圖。僅輸出 DANGER 或 SAFE。\n訊息：{user_msg}"
    result, _ = call_llm_generic(prompt, max_tokens=10, temperature=0.1)
    return "DANGER" in result.upper()

def get_crisis_response(user_msg: str):
    prompt = f"[System] 緊急狀況：使用者有危險傾向。你是諮商師 Soother。請提供「1925安心專線」。(繁體中文, 100字內)\n使用者：{user_msg}"
    return call_llm_generic(prompt, max_tokens=150, temperature=0.3)

def check_mood_for_image(user_msg: str) -> bool:
    prompt = f"[System] 判斷使用者情緒。若非常負面、悲傷或需要安慰，輸出 NEGATIVE。否則輸出 OTHER。\n訊息：{user_msg}"
    result, _ = call_llm_generic(prompt, max_tokens=10, temperature=0.1)
    return "NEGATIVE" in result.upper()

def generate_rag_prompt(user_msg, memory_context):
    docs = vector_store.similarity_search(user_msg, k=1)
    rag_info = "\n".join([d.page_content for d in docs]) if docs else "無"
    
    return f"""
[System Instruction]
你是諮商師 Soother。

**長期記憶與現狀**
{memory_context}

**參考知識**
{rag_info}

**規則**
1. 溫柔共情，不說教。
2. 結合記憶回應。
3. 若使用者詢問具體建議，參考知識回答。
4. 繁體中文，150字內。

使用者：{user_msg}
Soother：
"""

# --- Line Loading 動畫 ---
def start_loading_animation(chat_id):
    stop_event = threading.Event()
    def worker():
        url = "https://api.line.me/v2/bot/chat/loading/start"
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {config.LINE_ACCESS_TOKEN}"}
        while not stop_event.is_set():
            requests.post(url, headers=headers, json={"chatId": chat_id, "loadingSeconds": 20}, timeout=1)
            stop_event.wait(15)
    t = threading.Thread(target=worker, daemon=True)
    t.start()
    return stop_event

# ==============================================================================
# Webhook 與 主流程
# ==============================================================================

@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']
    body = request.get_data(as_text=True)
    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    user_msg = event.message.text.strip()
    user_id = event.source.user_id
    if not user_msg: return
    
    stop_anim = start_loading_animation(user_id)
    user_mem = MemoryManager(user_id)

    try:
        ai_reply = ""
        image_url = None
        
        if check_safety_risk(user_msg):
            print(f"🚨 [危機介入] User: {user_msg}")
            ai_reply, _ = get_crisis_response(user_msg)
        else:
            if check_mood_for_image(user_msg):
                print(f"🎨 [生圖觸發] 偵測到負面情緒...")
                image_url = generate_and_upload_image(user_msg)
            
            full_context = user_mem.get_full_context()
            reply_prompt = generate_rag_prompt(user_msg, full_context)
            ai_reply, src = call_llm_generic(reply_prompt)
            
            if ai_reply:
                user_mem.add_dialogue("User", user_msg)
                user_mem.add_dialogue("Soother", ai_reply)
                print(f"🤖 回應 ({src}): {ai_reply}")

        messages = []
        if ai_reply:
            final_text = clean_incomplete_sentence(to_traditional(ai_reply))
            messages.append(TextSendMessage(text=final_text))
        
        if image_url:
            messages.append(ImageSendMessage(original_content_url=image_url, preview_image_url=image_url))
            print(f"🖼️ 附圖: {image_url}")

        if messages:
            line_bot_api.reply_message(event.reply_token, messages)

    except Exception as e:
        print(f"❌ 處理錯誤: {e}")
        line_bot_api.reply_message(event.reply_token, TextSendMessage(text="抱歉，我現在有點混亂，請稍後再跟我說話。"))
    finally:
        user_mem.close()
        stop_anim.set()

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)