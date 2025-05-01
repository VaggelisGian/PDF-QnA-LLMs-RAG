from PyPDF2 import PdfReader
import os
import time
import uuid
import re
import asyncio
from tqdm import tqdm
from typing import Optional
from src.backend.api.progress import create_job, update_job_progress, complete_job_sync

class PDFLoader:
    def __init__(self, pdf_directory: str, max_pages: Optional[int] = None):
        self.pdf_directory = pdf_directory
        self.max_pages = max_pages

    async def extract_text_from_pdf(self, file_path: str, job_id: Optional[str] = None) -> str:
        text = ""
        start_time = time.time()

        try:
            with open(file_path, 'rb') as file:
                reader = PdfReader(file)
                total_pages = len(reader.pages)
                pages_to_process = min(total_pages, self.max_pages) if self.max_pages else total_pages

                if job_id:
                    file_name = os.path.basename(file_path)
                    create_job(job_id, file_name, pages_to_process)
                    print(f"Tracking extraction of {pages_to_process} pages for {file_name}")

                for i in range(pages_to_process):
                    page = reader.pages[i]
                    text += page.extract_text() + "\n"

                    if job_id:
                        await update_job_progress(job_id, i + 1)
                        await asyncio.sleep(0)

                if job_id:
                    complete_job_sync(job_id, "PDF extraction complete")

                print(f"Extracted {pages_to_process} pages in {time.time()-start_time:.2f}s")

        except Exception as e:
            error_msg = f"PDF extraction failed: {type(e).__name__} - {str(e)}"
            print(error_msg)
            if job_id:
                complete_job_sync(job_id, error_msg, "failed")
            return ""

        return self._clean_raw_text(text)

    def _clean_raw_text(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'(\w+)-\s*\n\s*(\w+)', r'\1\2', text)
        return text.strip()
