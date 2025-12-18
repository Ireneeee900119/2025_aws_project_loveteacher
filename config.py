# config.py
import os

# ==============================================================================
# [LINE Bot 設定]
# ==============================================================================
LINE_ACCESS_TOKEN = 'YOUR_LINE_ACCESS_TOKEN_HERE'
LINE_SECRET = 'YOUR_LINE_SECRET_HERE'

# ==============================================================================
# [AWS S3 設定]
# ==============================================================================
S3_BUCKET_NAME = 'yourS3BucketName'  # 請確認這也是 auto_sync 和 memory_manager 要用的 Bucket
AWS_REGION = 'us-east-1'

# S3 內的路徑設定
S3_DB_KEY = "database/chat_history.db"  # 資料庫在 S3 的路徑
S3_IMG_FOLDER = "gen_images"            # 圖片上傳的資料夾前綴

# ==============================================================================
# [LLM 模型伺服器設定]
# ==============================================================================
# Ollama 設定
OLLAMA_SERVER_URL = " "
OLLAMA_GENERATE_URL = " "
OLLAMA_MODEL = "GPT-OSS:120B"
SUMMARY_MODEL_OLLAMA = "GPT-OSS:120B"   # 記憶總結用的模型

# vLLM 設定 (備援)
VLLM_CHAT_URL = " "
VLLM_MODEL = "/model"
SUMMARY_MODEL_VLLM = "/model"           # 記憶總結用的模型

# ==============================================================================
# [ComfyUI 生圖設定]
# ==============================================================================
COMFY_BASE_URL = " "
COMFY_URL_ENDPOINT = f"{COMFY_BASE_URL}/prompt"

# ==============================================================================
# [RAG 與 資料庫設定]
# ==============================================================================
EMBED_MODEL_NAME = "BAAI/bge-large-zh-v1.5"
CHROMA_DB_PATH = "./chroma_db"          # 向量資料庫路徑
SQLITE_DB_FILE = "chat_history.db"      # 對話紀錄 SQL 檔案名稱
MANIFEST_FILE = "local_files.json"      # PDF 同步紀錄檔

# 自動同步檢查間隔 (秒)
SYNC_CHECK_INTERVAL = 30

# ==============================================================================
# [其他資源]
# ==============================================================================
# 心情小語清單
INSPIRATIONAL_QUOTES = [
    "The journey of a thousand miles begins with a single step.",
    "Believe you can and you're halfway there.",
    "The best way to predict the future is to create it.",
    "Success is not final, failure is not fatal.",
    "The purpose of our lives is to be happy.",
    "Strive for progress, not perfection.",
    "Keep going. Everything you need will come to you.",
    "It does not matter how slowly you go as long as you do not stop."
]