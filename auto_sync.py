import boto3
import os
import json
import time
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# 引入設定檔
import config

def load_manifest():
    if os.path.exists(config.MANIFEST_FILE):
        with open(config.MANIFEST_FILE, 'r') as f:
            return json.load(f)
    return []

def save_manifest(files):
    with open(config.MANIFEST_FILE, 'w') as f:
        json.dump(files, f)

def main():
    print(f"🚀 S3 自動同步精靈已啟動 (PID: {os.getpid()})")
    print(f"🎯 目標 Bucket: {config.S3_BUCKET_NAME}")
    
    print("🧠 正在載入本地 Embedding 模型...")
    try:
        model_kwargs = {'device': 'cpu'}
        encode_kwargs = {'normalize_embeddings': True}
        
        embeddings = HuggingFaceEmbeddings(
            model_name=config.EMBED_MODEL_NAME,
            model_kwargs=model_kwargs,
            encode_kwargs=encode_kwargs
        )
        print("✅ Embedding 模型載入完成")
    except Exception as e:
        print(f"❌ 模型載入失敗: {e}")
        return

    vector_store = Chroma(persist_directory=config.CHROMA_DB_PATH, embedding_function=embeddings)
    s3 = boto3.client('s3', region_name=config.AWS_REGION)

    while True:
        try:
            # 1. 取得 S3 檔案清單
            try:
                response = s3.list_objects_v2(Bucket=config.S3_BUCKET_NAME)
                s3_files = [obj['Key'] for obj in response.get('Contents', []) if obj['Key'].endswith('.pdf')]
            except Exception as e:
                print(f"❌ S3 連線錯誤: {e}")
                time.sleep(config.SYNC_CHECK_INTERVAL)
                continue

            local_files = load_manifest()
            
            # 2. 比對差異
            s3_set = set(s3_files)
            local_set = set(local_files)
            
            to_add = list(s3_set - local_set)
            to_remove = list(local_set - s3_set)

            if to_add or to_remove:
                print(f"\n🔍 偵測到變動! 新增: {len(to_add)}, 刪除: {len(to_remove)}")

            # 3. 處理新增
            for file_key in to_add:
                print(f"📥 下載並處理: {file_key}")
                local_path = f"/tmp/{os.path.basename(file_key)}"
                
                try:
                    s3.download_file(config.S3_BUCKET_NAME, file_key, local_path)
                    
                    loader = PyPDFLoader(local_path)
                    docs = loader.load()
                    
                    for doc in docs:
                        doc.metadata["source"] = file_key
                    
                    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
                    chunks = splitter.split_documents(docs)
                    
                    if chunks:
                        vector_store.add_documents(chunks)
                        local_files.append(file_key)
                        save_manifest(local_files)
                        print(f"   ✅ 已寫入 {len(chunks)} 個向量片段。")
                    
                    os.remove(local_path)
                except Exception as e:
                    print(f"   ❌ 處理失敗: {e}")
                    if os.path.exists(local_path):
                        os.remove(local_path)

            # 4. 處理刪除
            for file_key in to_remove:
                print(f"🗑️ 移除檔案: {file_key}")
                try:
                    vector_store._collection.delete(where={"source": file_key})
                    local_files.remove(file_key)
                    save_manifest(local_files)
                    print("   ✅ 資料庫已清理。")
                except Exception as e:
                    print(f"   ❌ 移除失敗: {e}")

        except Exception as e:
            print(f"⚠️ 發生未預期錯誤: {e}")

        time.sleep(config.SYNC_CHECK_INTERVAL)

if __name__ == "__main__":
    main()