from flask import Flask, request, abort
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage, ImageSendMessage

import requests
import json
import time
import random
import os
import boto3
from botocore.exceptions import NoCredentialsError
from PIL import Image, ImageDraw, ImageFont 
from io import BytesIO

# ==============================================================================
# 1. 配置與初始化
# ==============================================================================

# --- LLM 服務設定 ---
OLLAMA_SERVER_URL = "" 
OLLAMA_GENERATE_URL = ""
OLLAMA_MODEL = "GPT-OSS:120B" 

VLLM_CHAT_URL = ""
VLLM_MODEL = "/model" 

# --- AWS S3 配置 (加入您的配置) ---
S3_BUCKET_NAME = '' 
AWS_REGION = '' 
s3 = boto3.client('s3', region_name=AWS_REGION)

# --- ComfyUI 配置 ---
COMFY_URL_ENDPOINT = ""
BASE_URL = ""

# --- Line Bot 憑證 (加入您的配置) ---
CHANNEL_SECRET = ''
CHANNEL_ACCESS_TOKEN = ''

# --- 心情小語清單 (隨機選取) ---
INSPIRATIONAL_QUOTES = [
    "The journey of a thousand miles begins with a single step.",
    "Believe you can and you're halfway there.",
    "The best way to predict the future is to create it.",
    "Success is not final, failure is not fatal: it is the courage to continue that counts.",
    "The purpose of our lives is to be happy.",
    "Strive for progress, not perfection.",
    "Don't watch the clock; do what it does. Keep going.",
    "It does not matter how slowly you go as long as you do not stop."
]

# --- Flask & Line Bot SDK 初始化 ---
app = Flask(__name__)
line_bot_api = LineBotApi(CHANNEL_ACCESS_TOKEN) 
handler = WebhookHandler(CHANNEL_SECRET)


# ==============================================================================
# 2. LLM 核心功能：備援呼叫與情感分析
# ==============================================================================

def call_llm_generic(prompt: str, max_tokens=100, temperature=0.0) -> tuple[str, str]:
    """
    通用 LLM 呼叫函數：優先 VLLM (快)，失敗則備援 Ollama (穩)。
    """
    # 1. 嘗試 VLLM (優先)
    try:
        payload = {
            "model": VLLM_MODEL,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens
        }
        response = requests.post(VLLM_CHAT_URL, json=payload, timeout=5) # 縮短 timeout
        response.raise_for_status()
        data = response.json()
        content = data['choices'][0].get('message', {}).get('content', '').strip()
        if content: return content, "vLLM"
    except Exception:
        pass

    # 2. 備援 Ollama
    try:
        payload = {
            "model": OLLAMA_MODEL, "prompt": prompt, "stream": False,
            "options": {"temperature": temperature, "top_p": 0.9, "max_tokens": max_tokens}
        }
        response = requests.post(OLLAMA_GENERATE_URL, json=payload, timeout=20)
        response.raise_for_status()
        result = response.json().get('response', '').strip()
        if result: return result, "Ollama"
    except Exception as e:
        print(f"LLM 備援 Ollama Failed: {e}")
    
    return "NEUTRAL", "Fail" # 失敗時預設為中性，避免誤觸生圖

def get_user_mood(user_msg: str) -> str:
    """
    使用 LLM 判斷中文情感，返回 POSITIVE/NEGATIVE/NEUTRAL。
    """
    prompt = f"""[System] 判斷以下用戶訊息的情緒傾向。只回覆一個詞：正面、中性、負面。
用戶訊息："{user_msg}"
情緒："""
    # 使用極低的 temperature 0.0 確保分類穩定
    mood, _ = call_llm_generic(prompt, max_tokens=10, temperature=0.0) 
    
    if "負面" in mood or "差勁" in mood: return "NEGATIVE"
    if "正面" in mood: return "POSITIVE"
    return "NEUTRAL"

def get_therapeutic_prompt(original_prompt):
    """根據負面情緒調整 ComfyUI 的 Prompt，加入治癒修飾詞。"""
    enhancements = ", therapeutic, soft lighting, pastel colors, cozy atmosphere, focus on peace and calm, warm colors"
    return original_prompt + enhancements

# ==============================================================================
# 3. ComfyUI 核心運行與 S3 處理
# ==============================================================================

def get_comfy_workflow(prompt_text):
    """根據輸入的 Prompt 文本動態生成 ComfyUI 工作流程 JSON"""
    base_style_prompt = "perfect reflection, snowy mountains in the background, golden hour, highly detailed, photorealistic, cinematic lighting, 8k, extremely high quality"
    final_text = f"{prompt_text}, {base_style_prompt}"

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
        "6": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": final_text}}, 
        "7": {"class_type": "CLIPTextEncode", "inputs": {"clip": ["4", 1], "text": "text, watermark, low quality, worst quality, noise, blurry, deformed, mutated, tiling, poorly drawn, extra limbs"}},
        "8": {"class_type": "VAEDecode", "inputs": {"samples": ["3", 0], "vae": ["4", 2]}},
        "9": {"class_type": "SaveImage", "inputs": {"filename_prefix": "line_api", "images": ["8", 0]}}
    }

def check_history(prompt_id):
    """輪詢 ComfyUI 歷史紀錄，直到圖片生成完成 (60 秒超時)"""
    start_time = time.time()
    TIMEOUT_SECONDS = 60 
    while time.time() - start_time < TIMEOUT_SECONDS:
        try:
            response = requests.get(f"{BASE_URL}/history/{prompt_id}")
            history = response.json()
            if prompt_id in history:
                return history[prompt_id]
            time.sleep(1)
        except Exception:
            time.sleep(2)
    raise TimeoutError("ComfyUI 圖片生成超時 (60 秒)")

def get_image_bytes(filename, subfolder, folder_type):
    """從 ComfyUI 伺服器下載圖片內容 (Bytes)"""
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    response = requests.get(f"{BASE_URL}/view", params=data)
    response.raise_for_status()
    return response.content

def add_text_overlay_bytes(image_bytes):
    """
    在記憶體 (Bytes) 中對圖片疊加文字 (使用隨機英文小語)。
    
    """
    quote = random.choice(INSPIRATIONAL_QUOTES)
    
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
        draw = ImageDraw.Draw(img)
    
        initial_font_size = 10 
        
        try:
             font = ImageFont.truetype("arial.ttf", initial_font_size)
        except IOError:
             font = ImageFont.load_default()

        max_width = img.width - 40
        lines = []
        current_line = ""
        words = quote.split(' ')
        for word in words:
            test_line = (current_line + ' ' + word).strip()
            
            text_bbox = draw.textbbox((0, 0), test_line, font=font) 
            text_width = text_bbox[2] - text_bbox[0]
            
            if text_width <= max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word
        lines.append(current_line)

        final_font_size = initial_font_size
        line_spacing = 5 
        total_text_height = len(lines) * final_font_size + (len(lines) - 1) * line_spacing

        # 計算文字繪製的起始 Y 座標
        margin = 20 
        start_y = img.height - total_text_height - margin
        
        y = start_y
        for line in lines:
            x = margin 
            draw.text((x, y), line, font=font, fill=(255, 255, 255), stroke_width=1, stroke_fill=(0,0,0,150)) 
            y += final_font_size + line_spacing
        
        output_buffer = BytesIO()
        img = img.convert("RGB") 
        img.save(output_buffer, format="PNG") 
        output_buffer.seek(0)
        return output_buffer.getvalue()

    except Exception as e:
        print(f"文字疊加失敗: {e}")
        return image_bytes

def upload_bytes_to_s3(image_bytes, s3_object_key):
    """將 Bytes 資料直接上傳到 S3"""
    try:
        s3.put_object(Bucket=S3_BUCKET_NAME, Key=s3_object_key, Body=image_bytes, ContentType='image/png')
        s3_url = f"https://{S3_BUCKET_NAME}.s3.amazonaws.com/{s3_object_key}"
        print(f"S3 上傳成功: {s3_url}")
        return s3_url
    except Exception as e:
        print(f"S3 上傳失敗 (請檢查 IAM 權限): {e}")
        return None

def generate_and_upload_image(user_prompt, sentiment):
    """執行生成 -> 疊加文字 -> 上傳 S3 的完整流程"""
    try:
        # A. 調整 Prompt
        final_prompt = get_therapeutic_prompt(user_prompt)

        # B. 準備工作流
        workflow = get_comfy_workflow(final_prompt)
        s3_object_key = f"line_gen_output/{int(time.time())}_{random.randint(100, 999)}.png"
        
        # 提交給 ComfyUI
        response = requests.post(COMFY_URL_ENDPOINT, json={"prompt": workflow})
        response.raise_for_status() 
        prompt_id = response.json()['prompt_id']
        
        # 等待生成完成
        history_data = check_history(prompt_id)
        
        # 下載圖片 (Bytes)
        images_info = history_data['outputs'].get('9', {}).get('images', []) 
        if not images_info:
            return "找不到生成的圖片資訊", None
            
        img_info = images_info[0]
        original_image_bytes = get_image_bytes(img_info['filename'], img_info['subfolder'], img_info['type'])

        # 後製疊加文字
        final_image_bytes = add_text_overlay_bytes(original_image_bytes)

        # 上傳到 S3
        s3_url = upload_bytes_to_s3(final_image_bytes, s3_object_key)
        
        if s3_url:
            return f"圖片生成成功，S3 連結: {s3_url}", s3_url
        else:
            return "S3 上傳失敗", None
            
    except TimeoutError:
        return "圖片生成超時 (60 秒)，請稍後再試。", None
    except requests.exceptions.RequestException as e:
        return f"ComfyUI 或網路連線錯誤: {e}", None
    except Exception as e:
        return f"服務端發生未知錯誤: {e}", None


# ==============================================================================
# 4. Line Webhook 處理 (Flask 路由)
# ==============================================================================

@app.route("/callback", methods=['POST'])
def callback():
    """接收 Line Webhook 傳來的 POST 請求"""
    signature = request.headers.get('X-Line-Signature')
    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)
    
    if not signature:
         return 'OK' 

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        print("Invalid signature. Check channel access token/secret.")
        abort(400)
    
    return 'OK'

@handler.add(MessageEvent, message=TextMessage)
def handle_message(event):
    """處理接收到的文字訊息"""
    text = event.message.text
    reply_token = event.reply_token
    user_id = event.source.user_id

    # 判斷是否為生成指令
    if text.lower().startswith('') or text.lower().startswith(''):
        prompt = text.split(':', 1)[-1].strip()
        
        # 1. 立即回覆提示 (使用 Reply Token)
        """
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text=f"正在為您分析情緒，Prompt: {prompt}...")
        )
        """
        # 2. 執行情感分析
        sentiment_result = get_user_mood(prompt)

        # 3. 條件判斷：僅在負面情緒時才生成圖片
        if sentiment_result == 'NEGATIVE':
            
            # A. 告知開始生成 (使用 Push Message)
            """
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=f"情緒偵測為 **負面** ({sentiment_result})，開始生成治癒圖片...")
            )
            """
            # B. 執行 ComfyUI, Pillow, S3 上傳
            print(f"[{user_id}] 開始生成圖片，Prompt: {prompt}")
            message, image_url = generate_and_upload_image(prompt, sentiment_result)
            print(f"[{user_id}] 任務結果: {message}")

            # C. 發送最終結果
            if image_url:
                line_bot_api.push_message(
                    user_id,
                    ImageSendMessage(
                        original_content_url=image_url,
                        preview_image_url=image_url
                    )
                )
            else:
                line_bot_api.push_message(
                    user_id,
                    TextSendMessage(text=message)
                )
                
        else:
            # 4. 非負面情緒：直接文字回覆 (使用 Push Message)
            reply_message = (
                f"偵測到您的情緒為 **{sentiment_result}**。\n"
                f"本服務目前僅在偵測到負面情緒時提供生成治癒圖片的服務。\n"
                f"如果需要安慰，請嘗試輸入更明確的負面指令喔！"
            )
            line_bot_api.push_message(
                user_id,
                TextSendMessage(text=reply_message)
            )
            
    else:
        # 非生成指令的回覆 (使用 Reply Token)
        line_bot_api.reply_message(
            reply_token,
            TextSendMessage(text="請輸入 '生成: [您的描述]' 來開始圖片生成 (例如：生成: 我心情很差，安慰我)")
        )

# ==============================================================================
# 5. 運行 Flask 服務
# ==============================================================================

if __name__ == "__main__":
    # 在 5000 埠運行，供 API Gateway 集成
    print("--- 啟動 Line Bot Flask 伺服器，監聽 5000 埠 ---")
    app.run(host='0.0.0.0', port=5000)