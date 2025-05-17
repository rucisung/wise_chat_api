import os
import logging
# ChatbotEngine 클래스를 직접 임포트할 필요는 없지만,
# 인자로 받을 engine 객체의 타입을 명시하기 위해 임포트할 수 있습니다.
# from chatbot.engine import ChatbotEngine

logger = logging.getLogger(__name__) # main.py에서 설정한 로거를 사용합니다.

def load_local_rag_documents(engine, db_directory: str):
    """
    주어진 디렉토리에서 PDF 및 HTML 파일을 스캔하여 ChatbotEngine에 로드합니다.

    Args:
        engine: 문서를 처리할 ChatbotEngine 인스턴스
        db_directory: 문서를 스캔할 로컬 디렉토리 경로
    """
    local_file_paths = []

    # 디렉토리가 존재하는지 확인
    if not os.path.isdir(db_directory):
        logger.warning(f"Local DB directory not found: {db_directory}. Skipping local document processing.")
        return

    logger.info(f"Scanning local directory: {db_directory}")
    for filename in os.listdir(db_directory):
        # 파일의 전체 경로 생성
        full_path = os.path.join(db_directory, filename)
        # 파일인지 확인하고 .pdf 또는 .html 확장자를 가졌는지 확인
        if os.path.isfile(full_path) and (filename.endswith(".pdf") or filename.endswith(".html")):
            local_file_paths.append(full_path)

    logger.info(f"Found {len(local_file_paths)} eligible local files in {db_directory}")

    # 로컬 파일 처리
    if local_file_paths:
        try:
            logger.info("Starting processing of local documents...")
            engine.process_documents(local_file_paths)
            logger.info("Processing of local documents complete. RAG is enabled for these files.")
        except Exception as e:
            logger.error(f"Error processing local documents from {db_directory}: {str(e)}")
    else:
        logger.info(f"No eligible local PDF/HTML files found to process from {db_directory}.")