from PyPDF2 import PdfReader

from unstructured.partition.pdf import partition_pdf


def upload_pdf(file_path: str) -> dict:
    """
    Uploads a PDF file and extracts its text content.

    Args:
        file_path (str): The path to the PDF file.

    Returns:
        dict: A dictionary containing the file name and extracted text.
    """

    # reader = PdfReader(file_path)
    # extracted_text = []

    # for page in reader.pages:
    #     page_text = page.extract_text()
    #     if page_text and page_text.strip():
    #         extracted_text.append(page_text.strip())

    # return {"file_name": file_path, "extracted_text": "\n".join(extracted_text)}

    elements = partition_pdf(file_path)

    extracted_text = []
    for el in elements:
        if el.text and len(el.text.strip()) > 30:
            extracted_text.append(el.text.strip())

    return {
        "file_name": file_path,
        "extracted_text": "\n\n".join(extracted_text),
    }


def ingest_jd(jd_text: str) -> dict:
    """
    Ingests job description text.

    Args:
        jd_text (str): The job description text.

    Returns:
        dict: A dictionary containing the job description text.
    """
    return {"job_id": "jd_001", "raw_text": jd_text, "structured": None}
