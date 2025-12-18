import sqlite3
import json
import time
import os
import requests
import boto3
import opencc
from threading import Thread
from botocore.exceptions import ClientError

# 引入設定檔
import config

class MemoryManager:
    def __init__(self, user_id):
        self.user_id = user_id
        self.s3 = boto3.client('s3', region_name=config.AWS_REGION)
        self.cc = opencc.OpenCC('s2twp')
        
        # 使用 config 中的路徑設定
        self.db_file = config.SQLITE_DB_FILE
        
        # 1. 初始化檢查
        if not os.path.exists(self.db_file):
            self._download_db_from_s3()
            
        # 2. 建立資料庫連線
        self.conn = sqlite3.connect(self.db_file, check_same_thread=False)
        self.cursor = self.conn.cursor()
        self._init_table()
        
        # 3. 載入使用者資料
        self.cursor.execute("SELECT summary, recent_logs FROM users WHERE user_id=?", (user_id,))
        row = self.cursor.fetchone()
        
        if row:
            self.summary = row[0] if row[0] else "（尚無深入了解）"
            self.recent_logs = json.loads(row[1]) if row[1] else []
        else:
            self.summary = "（新使用者）"
            self.recent_logs = []
            self.cursor.execute("INSERT INTO users (user_id, summary, recent_logs, last_updated) VALUES (?, ?, ?, ?)", 
                                (user_id, self.summary, json.dumps([]), time.time()))
            self.conn.commit()

    def _init_table(self):
        self.cursor.execute('''CREATE TABLE IF NOT EXISTS users
                             (user_id TEXT PRIMARY KEY, 
                              summary TEXT, 
                              recent_logs TEXT,
                              last_updated REAL)''')
        self.conn.commit()

    def _download_db_from_s3(self):
        print("☁️ [S3 Sync] 正在從 S3 下載資料庫...")
        try:
            self.s3.download_file(config.S3_BUCKET_NAME, config.S3_DB_KEY, self.db_file)
            print("✅ [S3 Sync] 下載完成，記憶已還原。")
        except ClientError as e:
            error_code = e.response['Error']['Code']
            if error_code == "404":
                print("ℹ️ [S3 Sync] S3 上還沒有資料庫，將建立新的。")
            elif error_code == "403":
                print("⛔ [S3 Sync] 下載失敗：權限不足 (403 Forbidden)。")
            else:
                print(f"❌ [S3 Sync] 下載失敗: {e}")

    def _upload_db_to_s3(self):
        try:
            self.s3.upload_file(self.db_file, config.S3_BUCKET_NAME, config.S3_DB_KEY)
        except Exception as e:
            print(f"❌ [S3 Sync] 上傳失敗: {e}")

    def _call_llm_for_summary(self, prompt):
        # 1. 嘗試 Ollama
        try:
            payload = {
                "model": config.SUMMARY_MODEL_OLLAMA,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3, "max_tokens": 300}
            }
            res = requests.post(config.OLLAMA_GENERATE_URL, json=payload, timeout=60)
            res.raise_for_status()
            result = res.json().get("response", "").strip()
            if result:
                return result, "Ollama"
        except Exception as e:
            print(f"⚠️ [Memory] Ollama Summary Failed: {e}, 切換至 vLLM...")

        # 2. 備援嘗試 vLLM
        try:
            payload = {
                "model": config.SUMMARY_MODEL_VLLM,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3, 
                "top_p": 0.9,
                "max_tokens": 300
            }
            res = requests.post(config.VLLM_CHAT_URL, json=payload, timeout=30)
            res.raise_for_status()
            data = res.json()
            if 'choices' in data and data['choices']:
                content = data['choices'][0].get('message', {}).get('content')
                if content and content.strip():
                    return content.strip(), "vLLM"
        except Exception as e:
            print(f"❌ [Memory] vLLM Summary Failed: {e}")

        return None, "Fail"

    def add_dialogue(self, role, content):
        self.recent_logs.append(f"{role}: {content}")
        self._save_to_db()
        if len(self.recent_logs) > 4:
            Thread(target=self._consolidate_memory).start()

    def get_full_context(self):
        logs_str = "\n".join(self.recent_logs)
        return f"【長期記憶】\n{self.summary}\n\n【短期記憶】\n{logs_str}"

    def _save_to_db(self):
        self.cursor.execute("UPDATE users SET summary=?, recent_logs=?, last_updated=? WHERE user_id=?", 
                            (self.summary, json.dumps(self.recent_logs), time.time(), self.user_id))
        self.conn.commit()
        Thread(target=self._upload_db_to_s3).start()

    def _consolidate_memory(self):
        print(f"🧠 [Memory] 記憶壓縮中 ({self.user_id[-5:]})...")
        old_logs = self.recent_logs[:2]
        self.recent_logs = self.recent_logs[2:]
        old_logs_str = "\n".join(old_logs)
        
        prompt = f"""[System] 更新使用者畫像。保留重要資訊，加入新資訊。
原記憶：{self.summary}
新對話：{old_logs_str}
新記憶(繁體中文)："""
        
        new_sum, source = self._call_llm_for_summary(prompt)
        
        if new_sum:
            self.summary = self.cc.convert(new_sum)
            try:
                temp_conn = sqlite3.connect(self.db_file)
                temp_cursor = temp_conn.cursor()
                temp_cursor.execute("SELECT recent_logs FROM users WHERE user_id=?", (self.user_id,))
                row = temp_cursor.fetchone()
                if row:
                    current_db_logs = json.loads(row[0])
                    trimmed_logs = current_db_logs[-4:] 
                    temp_cursor.execute("UPDATE users SET summary=?, recent_logs=?, last_updated=? WHERE user_id=?", 
                                        (self.summary, json.dumps(trimmed_logs), time.time(), self.user_id))
                    temp_conn.commit()
                temp_conn.close() 
                self._upload_db_to_s3()
                print(f"✅ [Memory] 更新完成 (By {source}): {self.summary[:20]}...")
            except Exception as e:
                print(f"❌ [Background Save] 寫入失敗: {e}")
        else:
            print("❌ [Memory] 失敗，還原對話。")
            self.recent_logs = old_logs + self.recent_logs 
            try:
                temp_conn = sqlite3.connect(self.db_file)
                temp_cursor = temp_conn.cursor()
                temp_cursor.execute("UPDATE users SET summary=?, recent_logs=?, last_updated=? WHERE user_id=?", 
                                    (self.summary, json.dumps(self.recent_logs), time.time(), self.user_id))
                temp_conn.commit()
                temp_conn.close()
            except:
                pass
    def close(self):
        self.conn.close()