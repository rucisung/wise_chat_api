from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from chatbot.engine import ChatbotEngine
import os
from dotenv import load_dotenv
import tempfile
import logging

from app_utils import load_local_rag_documents


# 로그 디렉토리 생성
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'chatbot.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# .env 파일 로드
load_dotenv()

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Google API 키 확인
google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    logger.error("GOOGLE_API_KEY is not set in .env file")
    raise ValueError("GOOGLE_API_KEY is not set in .env file")

# 챗봇 엔진 초기화
bot = ChatbotEngine(api_key=google_api_key)
logger.info("FastAPI app initialized")

#RAG temp
local_db_directory_path = "/Users/ruci/Documents/DB/wise_DB"
load_local_rag_documents(bot, local_db_directory_path)


class Query(BaseModel):
    message: str

class Role(BaseModel):
    role: str

@app.post("/chat_dynamic_role")
async def chat_dynamic_role(query: Query):
    """
    매 메시지마다 역할을 자동으로 분석하고 설정한 후 응답합니다.
    RAG 사용 가능 시 자동으로 RAG를 사용합니다 ('auto' mode).
    """
    try:
        # 현재 메시지 내용에 따라 역할 자동 설정
        role = bot.auto_set_role(query.message)
        # 설정된 역할 기반으로 응답 생성 - mode='auto' (기본값) 사용
        response_data = bot.get_response(query.message, mode="auto") # 명시적으로 'auto' 지정
        logger.info(f"Dynamic role chat: Role set to '{role}' for query: '{query.message}'")

        # response_data는 {"answer": "...", "sources": [...], "mode_used": "..."} 형태
        return {
            "role": role,
            "response": response_data.get("answer", "Error retrieving answer."),
            "sources": response_data.get("sources", []),
            "mode_used": response_data.get("mode_used", "unknown"),
            "message": "Role automatically determined for this response. RAG used if documents available."
        }
    except Exception as e:
        logger.error(f"Error in /chat_dynamic_role: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_persistent_role")
async def chat_persistent_role(query: Query):
    """
    현재 설정된 역할을 사용하여 응답합니다.
    RAG 사용 가능 시 자동으로 RAG를 사용합니다 ('auto' mode).
    """
    try:
        # 현재 bot 인스턴스에 설정된 역할을 사용하여 응답 생성 - mode='auto' (기본값) 사용
        response_data = bot.get_response(query.message, mode="auto") # 명시적으로 'auto' 지정
        current_role_info = bot.get_role_info()
        logger.info(f"Persistent role chat: Using role '{current_role_info['role']}' for query: '{query.message}'")

        # response_data는 {"answer": "...", "sources": [...], "mode_used": "..."} 형태
        return {
            "role": current_role_info['role'],
            "response": response_data.get("answer", "Error retrieving answer."),
            "sources": response_data.get("sources", []),
            "mode_used": response_data.get("mode_used", "unknown"),
            "message": f"Responded using the persistent role: {current_role_info['role']}. RAG used if documents available."
        }
    except Exception as e:
        logger.error(f"Error in /chat_persistent_role: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/chat")
# async def chat(query: Query):
#     try:
#         response = bot.get_response(query.message)
#         logger.info(f"Chat response for query: {query.message}")
#         return {"response": response}
#     except Exception as e:
#         logger.error(f"Error in /chat: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.post("/chat_rag")
async def chat_rag(query: Query):
    """
    명시적으로 RAG를 사용하여 응답합니다.
    문서가 로드되지 않은 경우 에러를 반환합니다.
    """
    try:
        # RAG 모드로 응답 생성
        response_data = bot.get_response(query.message, mode="rag")
        logger.info(f"Explicit RAG chat for query: '{query.message}'")

        # response_data는 {"answer": "...", "sources": [...], "mode_used": "..."} 형태
        return {
            "response": response_data.get("answer", "Error retrieving answer."),
            "sources": response_data.get("sources", []),
            "mode_used": response_data.get("mode_used", "unknown"),
            "message": "Response generated using RAG." if response_data.get("mode_used") == "rag" else "RAG was requested but not used."
        }
    except HTTPException:
         # ChatbotEngine.get_response에서 RAG 불가 시 HTTPException을 발생시키지 않으므로 여기서 잡을 필요 없음
         raise
    except Exception as e:
        logger.error(f"Error in /chat_rag: {str(e)}")
        # ChatbotEngine에서 에러가 딕셔너리 형태로 반환될 경우를 대비하여 확인
        if isinstance(e, dict) and "answer" in e:
             raise HTTPException(status_code=500, detail=e["answer"])
        else:
             raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat_base")
async def chat_base(query: Query):
    """
    명시적으로 기본 대화 체인을 사용하여 응답합니다.
    RAG 문서가 로드되어 있어도 사용하지 않습니다.
    """
    try:
        # 기본 모드로 응답 생성
        response_data = bot.get_response(query.message, mode="base")
        logger.info(f"Explicit base chat for query: '{query.message}'")

        # response_data는 {"answer": "...", "sources": [...], "mode_used": "..."} 형태
        return {
            "response": response_data.get("answer", "Error retrieving answer."),
            "sources": response_data.get("sources", []), # Base mode에서는 항상 빈 리스트
            "mode_used": response_data.get("mode_used", "unknown"),
            "message": "Response generated using base chat."
        }
    except Exception as e:
        logger.error(f"Error in /chat_base: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    try:
        temp_files = []
        for file in files:
            if file.filename.endswith(('.pdf', '.html')):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                    content = await file.read()
                    tmp.write(content)
                    temp_files.append(tmp.name)
                    logger.info(f"Temporary file created: {tmp.name}")
            else:
                logger.warning(f"Invalid file type: {file.filename}")
                raise HTTPException(status_code=400, detail="Only PDF or HTML files are allowed")
        
        bot.process_documents(temp_files)
        
        for tmp_file in temp_files:
            os.unlink(tmp_file)
            logger.info(f"Temporary file deleted: {tmp_file}")
            
        return {"message": "Files processed successfully"}
    except Exception as e:
        logger.error(f"Error in /upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/set_role")
async def set_role(role: Role):
    try:
        success = bot.set_role(role.role)
        if success:
            logger.info(f"Role set via /set_role: {role.role}")
            return {"message": f"Role set to {role.role}"}
        else:
            logger.warning(f"Invalid role in /set_role: {role.role}")
            raise HTTPException(status_code=400, detail=f"Invalid role: {role.role}")
    except Exception as e:
        logger.error(f"Error in /set_role: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# @app.post("/auto_role")
# async def auto_role(query: Query):
#     try:
#         role = bot.auto_set_role(query.message)
#         response = bot.get_response(query.message)
#         logger.info(f"Auto role set to {role} for query: {query.message}")
#         return {"role": role, "response": response}
#     except Exception as e:
#         logger.error(f"Error in /auto_role: {str(e)}")
#         raise HTTPException(status_code=500, detail=str(e))

@app.get("/get_role")
async def get_role():
    try:
        role_info = bot.get_role_info()
        logger.info(f"Role info requested: {role_info}")
        return role_info
    except Exception as e:
        logger.error(f"Error in /get_role: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))