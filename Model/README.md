# AI CUP 2024 玉山人工智慧公開挑戰賽－RAG與LLM在金融問答的應用

## 說明
- 本資料夾包含一支程式：rag_chunk.py，為模型推論之主程式。
- 參照文件須放置於reference資料夾下
- 問題json檔須放置在dataset/preliminary下
- 比賽時使用之程式參數與模型超參數已直接寫死在程式碼裡，無須修改便可直接執行：python rag_chunk.py
- 程式執行後產生之答案檔為answer_chunk_rerank_BAAI_bge-reranker-large_ws.json