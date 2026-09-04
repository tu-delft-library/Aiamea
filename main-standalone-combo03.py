from pathlib import Path
import pandas as pd
import json
import xml.etree.ElementTree as ET
import uuid
import threading
import multiprocessing as mp
import queue
import tkinter as tk
import sys
import time
import os
from tkinter import ttk, filedialog, messagebox
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# MODEL CONFIGURATION
# ----------------------------

OLLAMA_BASE_URL = "http://127.0.0.1:11434/v1"
OPENAI_MODEL = "gpt-4o-mini"
OLLAMA_MODEL_8B = "qwen3:8b"
OLLAMA_MODEL_14B = "qwen3:14b"

OLLAMA_TIMEOUT = 90
OLLAMA_RETRIES = 2
OLLAMA_MAX_TOKENS = 1500
OPENAI_TIMEOUT = 90
OPENAI_RETRIES = 2
OPENAI_MAX_TOKENS = 1500

MODEL_OPTIONS = [
    f"OpenAI — {OPENAI_MODEL}",
    f"Local Ollama — {OLLAMA_MODEL_8B}",
    f"Local Ollama — {OLLAMA_MODEL_14B}",
]

MODEL_CONFIG = {
    MODEL_OPTIONS[0]: {
        "backend": "openai",
        "model": OPENAI_MODEL,
        "cache_key": "openai_gpt-4o-mini",
    },
    MODEL_OPTIONS[1]: {
        "backend": "ollama",
        "model": OLLAMA_MODEL_8B,
        "cache_key": "ollama_qwen3-8b",
    },
    MODEL_OPTIONS[2]: {
        "backend": "ollama",
        "model": OLLAMA_MODEL_14B,
        "cache_key": "ollama_qwen3-14b",
    },
}


# ----------------------------
# CONSTANTS
# ----------------------------

TESSERACT_CONFIG = "--oem 3 --psm 12"

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
POPPLER_PATH = BASE_DIR / "poppler-25.11.0" / "Library" / "bin"

env_path = BASE_DIR / ".env"

COAR_TYPE_MAP = {
    "journal":    "http://purl.org/coar/resource_type/c_6501",
    "conference": "http://purl.org/coar/resource_type/c_5794",
    "book":       "http://purl.org/coar/resource_type/c_1790",
    "chapter":    "http://purl.org/coar/resource_type/c_1791",
    "report":     "http://purl.org/coar/resource_type/c_1897",
}

RIS_TYPE_MAP = {
    "journal":    "JOUR",
    "conference": "GEN",
    "book":       "BOOK",
    "chapter":    "CHAP",
    "report":     "RPRT",
}

BASE_PROMPT = """
Extract structured metadata from this research paper text.

Return ONLY valid JSON with the following fields:

- title
- subtitle (optional)
- authors (array of objects with:
    - name (full name)
    - affiliations (string; if multiple affiliations, separate them with semicolons))
- year
- abstract
- keywords (array)
- doi
- publication_type: journal or conference?
- publisher
- journal (optional)
- conference_name (optional)
- conference_acronym (optional)
- conference_place (optional)
- conference_country (optional)
- conference_dates (optional; start_date, end_date)

Rules for author–affiliation matching:

- Detect author–affiliation markers (numbers, *, †, superscripts)
- Use markers first if present
- If no markers exist, match based on layout proximity
- If layout is unclear, match based on most probable relation
- Each author should have the most likely affiliation if available
- Do NOT create a separate affiliations field at the top level

General rules:

- Return only valid JSON
- No explanations
- No markdown
- No comments
- Missing fields should be null

OCR text:
{OCR_TEXT}
"""

# ----------------------------
# HELPERS
# ----------------------------

def indent(elem, level=0):
    i = "\n" + level * "    "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "    "
        for child in elem:
            indent(child, level + 1)
        if not elem.tail or not elem.tail.strip():
            elem.tail = i
    else:
        if level and (not elem.tail or not elem.tail.strip()):
            elem.tail = i


def safe_text(val):
    if val is None:
        return ""
    if isinstance(val, dict):
        return json.dumps(val)
    return str(val)


def as_list(val):
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if v is not None]
    return [val]


def get_affiliations(author):
    raw = author.get("affiliations") or author.get("affiliation") or ""

    if isinstance(raw, list):
        result = []

        for item in raw:
            for part in safe_text(item).split(";"):
                part = part.strip()

                if part:
                    result.append(part)

        return result

    else:
        return [
            a.strip()
            for a in safe_text(raw).split(";")
            if a.strip()
        ]


def resource_path(relative_path):
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS) / relative_path

    return Path(__file__).resolve().parent / relative_path


def apply_overrides(metadata, event_override=None, pub_override=None):
    """
    Merge manually entered UI values into a single local-model-extracted
    metadata dict before XML generation.

    - event_override: overwrites conference_name/acronym/place/country/dates
      when the corresponding manual field is non-blank. Blank manual fields
      fall back to whatever the local model extracted.
    - pub_override: injects publication_date, volume, edition, ISBNs,
      publisher and host publication details.
    """

    if event_override:
        if event_override.get("name"):
            metadata["conference_name"] = event_override["name"]

        if event_override.get("acronym"):
            metadata["conference_acronym"] = event_override["acronym"]

        if event_override.get("place"):
            metadata["conference_place"] = event_override["place"]

        if event_override.get("country"):
            metadata["conference_country"] = event_override["country"]

        dates = metadata.get("conference_dates") or {}

        if not isinstance(dates, dict):
            dates = {}

        if event_override.get("start_date"):
            dates["start_date"] = event_override["start_date"]

        if event_override.get("end_date"):
            dates["end_date"] = event_override["end_date"]

        if dates:
            metadata["conference_dates"] = dates

    if pub_override:
        if pub_override.get("publication_date"):
            metadata["year"] = pub_override["publication_date"]

        if pub_override.get("volume"):
            metadata["volume"] = pub_override["volume"]

        if pub_override.get("edition"):
            metadata["edition"] = pub_override["edition"]

        if pub_override.get("isbn_print"):
            metadata["isbn_print"] = pub_override["isbn_print"]

        if pub_override.get("isbn_print_2"):
            metadata["isbn_print_2"] = pub_override["isbn_print_2"]

        if pub_override.get("isbn_online"):
            metadata["isbn_online"] = pub_override["isbn_online"]

        if pub_override.get("publisher"):
            metadata["publisher"] = pub_override["publisher"]

        if pub_override.get("host_title"):
            metadata["host_title"] = pub_override["host_title"]

        if pub_override.get("host_subtitle"):
            metadata["host_subtitle"] = pub_override["host_subtitle"]

    return metadata


# ----------------------------
# OCR FUNCTIONS
# ----------------------------

def ocr_with_tesseract(pdf_path, dpi, first_page, last_page):
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    import numpy as np

    TESSERACT_PATH = BASE_DIR / "tesseract" / "tesseract.exe"
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_PATH)

    pages = convert_from_path(
        str(pdf_path),
        dpi,
        poppler_path=POPPLER_PATH,
        first_page=first_page,
        last_page=last_page
    )

    master_lines = []

    for i, page in enumerate(pages, start=first_page):
        gray = cv2.cvtColor(
            np.array(page),
            cv2.COLOR_RGB2GRAY
        )

        _, thresh = cv2.threshold(
            gray,
            127,
            255,
            cv2.THRESH_BINARY
        )

        data = pytesseract.image_to_data(
            thresh,
            config=TESSERACT_CONFIG,
            output_type="dict"
        )

        df = pd.DataFrame(data)
        df = df[df["conf"].astype(float) > 0]

        page_dict = {}

        for _, row in df.iterrows():
            key = f"{i}_{row['par_num']}_{row['line_num']}"
            text = str(row["text"]).strip()

            if not text:
                continue

            page_dict.setdefault(key, "")
            page_dict[key] += (
                " " if page_dict[key] else ""
            ) + text

        for key in sorted(page_dict.keys()):
            master_lines.append(page_dict[key])

    return " ".join(master_lines)


def ocr_with_pymupdf(pdf_path, first_page, last_page):
    import fitz

    doc = fitz.open(str(pdf_path))

    all_text = []

    start = first_page - 1
    end = min(last_page, len(doc)) if last_page is not None else len(doc)

    for page_num in range(start, end):
        page = doc[page_num]
        text = page.get_text("text")
        all_text.append(text)

    doc.close()

    return " ".join(all_text)


def ocr_pdf(pdf_path, ocr_engine, dpi, first_page, last_page):
    if ocr_engine == "pymupdf":
        return ocr_with_pymupdf(
            pdf_path,
            first_page,
            last_page
        )

    elif ocr_engine == "tesseract":
        return ocr_with_tesseract(
            pdf_path,
            dpi,
            first_page,
            last_page
        )

    else:
        raise ValueError(
            f"Unknown OCR engine: '{ocr_engine}'"
        )


# ----------------------------
# LOCAL MODEL EXTRACTION
# ----------------------------

def extract_json_from_response(raw_output):
    """
    Parse JSON from the local model response.

    Handles:
    - plain JSON
    - markdown code fences
    - Qwen <think>...</think> followed by JSON
    - surrounding explanatory text
    """
    raw_output = (raw_output or "").strip()

    if "</think>" in raw_output:
        raw_output = raw_output.split("</think>", 1)[1].strip()

    if raw_output.startswith("```"):
        lines = raw_output.splitlines()

        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]

        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]

        raw_output = "\n".join(lines).strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(
            "No JSON object found in model response."
        )

    return json.loads(raw_output[start:end + 1])


def _ollama_worker(prompt_json, model_name, result_queue):
    """Run one Ollama request in a separate process.

    The parent process can terminate this worker if Ollama gets stuck.
    """
    try:
        import json as _json
        from urllib import request as _request

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": prompt_json
                }
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "num_predict": OLLAMA_MAX_TOKENS
            }
        }

        data = _json.dumps(payload).encode("utf-8")

        req = _request.Request(
            "http://127.0.0.1:11434/api/chat",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with _request.urlopen(req, timeout=OLLAMA_TIMEOUT) as response:
            body = response.read().decode("utf-8")

        result_queue.put({
            "ok": True,
            "content": _json.loads(body)["message"]["content"]
        })

    except Exception as e:
        result_queue.put({
            "ok": False,
            "error": repr(e)
        })


def _extract_json_from_response(raw_output):
    """Extract JSON from plain JSON, fenced JSON, or Qwen think output."""
    raw_output = (raw_output or "").strip()

    if "</think>" in raw_output:
        raw_output = raw_output.split("</think>", 1)[1].strip()

    if raw_output.startswith("```"):
        lines = raw_output.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw_output = "\n".join(lines).strip()

    try:
        return json.loads(raw_output)
    except json.JSONDecodeError:
        pass

    start = raw_output.find("{")
    end = raw_output.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in model response.")

    return json.loads(raw_output[start:end + 1])


def openai_extract_json(
    ocr_text,
    prompt_mode="base",
    prompt_file=None,
    snippet_length=8000,
    log_callback=None
):
    """Extract metadata through the OpenAI API using gpt-4o-mini."""
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set. Add it to the environment "
            "or to the .env file next to Aiamea."
        )

    snippet = ocr_text[:snippet_length]

    if prompt_mode == "base":
        prompt_json = BASE_PROMPT.replace(
            "{OCR_TEXT}",
            snippet
        )
    else:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt_json = prompt_template.replace(
            "{OCR_TEXT}",
            snippet
        )

    last_error = None

    for attempt in range(1, OPENAI_RETRIES + 1):
        if log_callback:
            log_callback(
                f"OpenAI attempt {attempt}/{OPENAI_RETRIES}..."
            )

        try:
            client = OpenAI(api_key=api_key)

            response = client.chat.completions.create(
                model=OPENAI_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt_json
                    }
                ],
                temperature=0,
                max_tokens=OPENAI_MAX_TOKENS,
                timeout=OPENAI_TIMEOUT
            )

            raw_output = (
                response.choices[0].message.content or ""
            ).strip()

            return extract_json_from_response(raw_output)

        except Exception as e:
            last_error = repr(e)

            if log_callback:
                log_callback(
                    f"OpenAI attempt {attempt} failed: {last_error}"
                )

            if attempt < OPENAI_RETRIES:
                time.sleep(1)

    raise RuntimeError(
        f"OpenAI extraction failed after {OPENAI_RETRIES} attempts: "
        f"{last_error}"
    )


def ollama_extract_json(
    ocr_text,
    model_name,
    prompt_mode="base",
    prompt_file=None,
    snippet_length=8000,
    log_callback=None
):
    snippet = ocr_text[:snippet_length]

    if prompt_mode == "base":
        prompt_json = BASE_PROMPT.replace(
            "{OCR_TEXT}",
            snippet
        )
    else:
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt_template = f.read()
        prompt_json = prompt_template.replace(
            "{OCR_TEXT}",
            snippet
        )

    last_error = None

    for attempt in range(1, OLLAMA_RETRIES + 1):
        if log_callback:
            log_callback(
                f"Qwen attempt {attempt}/{OLLAMA_RETRIES}..."
            )

        result_queue = mp.Queue()
        worker = mp.Process(
            target=_ollama_worker,
            args=(prompt_json, model_name, result_queue),
            daemon=True
        )

        try:
            worker.start()
            worker.join(OLLAMA_TIMEOUT)

            if worker.is_alive():
                worker.terminate()
                worker.join(5)
                last_error = (
                    f"Qwen hard timeout after {OLLAMA_TIMEOUT} seconds"
                )

                if log_callback:
                    log_callback(last_error)

                continue

            if result_queue.empty():
                last_error = (
                    f"Qwen worker exited without a result "
                    f"(exit code {worker.exitcode})"
                )
                if log_callback:
                    log_callback(last_error)
                continue

            result = result_queue.get()

            if not result.get("ok"):
                last_error = result.get("error", "Unknown Ollama error")
                if log_callback:
                    log_callback(
                        f"Qwen attempt {attempt} failed: {last_error}"
                    )
                continue

            return _extract_json_from_response(
                result.get("content", "")
            )

        except Exception as e:
            last_error = repr(e)
            if log_callback:
                log_callback(
                    f"Qwen attempt {attempt} failed: {last_error}"
                )
        finally:
            try:
                if worker.is_alive():
                    worker.terminate()
                    worker.join(5)
            except Exception:
                pass
            try:
                result_queue.close()
                result_queue.join_thread()
            except Exception:
                pass

        if attempt < OLLAMA_RETRIES and log_callback:
            log_callback("Retrying Qwen after 1 second...")
            time.sleep(1)

    raise RuntimeError(
        f"Ollama extraction failed after {OLLAMA_RETRIES} attempts: "
        f"{last_error}"
    )


def extract_metadata_json(
    ocr_text,
    model_choice,
    prompt_mode="base",
    prompt_file=None,
    snippet_length=8000,
    log_callback=None
):
    """Dispatch metadata extraction to the selected model backend."""
    config = MODEL_CONFIG.get(model_choice)

    if not config:
        raise ValueError(
            f"Unknown metadata model: '{model_choice}'"
        )

    if config["backend"] == "openai":
        return openai_extract_json(
            ocr_text,
            prompt_mode=prompt_mode,
            prompt_file=prompt_file,
            snippet_length=snippet_length,
            log_callback=log_callback
        )

    return ollama_extract_json(
        ocr_text,
        model_name=config["model"],
        prompt_mode=prompt_mode,
        prompt_file=prompt_file,
        snippet_length=snippet_length,
        log_callback=log_callback
    )


# ----------------------------
# XML BUILDER
# ----------------------------

def json_to_oai_pmh(
    metadata_list,
    event_override=None,
    pub_override=None
):
    ns_oai = "http://www.openarchives.org/OAI/2.0/"
    ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns_cerif = "https://www.openaire.eu/cerif-profile/1.2/"
    ns_pubt = "https://www.openaire.eu/cerif-profile/vocab/COAR_Publication_Types"
    ns_ar = "http://purl.org/coar/access_right"

    ET.register_namespace("", ns_oai)
    ET.register_namespace("cerif", ns_cerif)
    ET.register_namespace("pubt", ns_pubt)
    ET.register_namespace("ar", ns_ar)

    root = ET.Element(
        f"{{{ns_oai}}}OAI-PMH",
        attrib={
            f"{{{ns_xsi}}}schemaLocation":
                f"{ns_oai} {ns_oai}/OAI-PMH.xsd "
                f"{ns_cerif} https://www.openaire.eu/schema/cris/current/openaire-cerif-profile.xsd"
        }
    )

    list_records = ET.SubElement(
        root,
        f"{{{ns_oai}}}ListRecords"
    )

    for metadata in metadata_list:
        metadata = apply_overrides(
            metadata,
            event_override=event_override,
            pub_override=pub_override
        )

        record = ET.SubElement(
            list_records,
            f"{{{ns_oai}}}record"
        )

        metadata_el = ET.SubElement(
            record,
            "metadata"
        )

        pub_el = ET.SubElement(
            metadata_el,
            f"{{{ns_cerif}}}Publication",
            id=f"Publications/{uuid.uuid4()}"
        )

        # --------------------------------------------------
        # PUBLICATION TYPE
        # --------------------------------------------------

        for pub_type in as_list(
            metadata.get("publication_type")
        ):
            pub_type_uri = COAR_TYPE_MAP.get(
                safe_text(pub_type).lower(),
                COAR_TYPE_MAP["conference"]
            )

            ET.SubElement(
                pub_el,
                f"{{{ns_pubt}}}Type"
            ).text = pub_type_uri

        # --------------------------------------------------
        # LANGUAGE
        # --------------------------------------------------

        ET.SubElement(
            pub_el,
            f"{{{ns_cerif}}}Language"
        ).text = "en"

        # --------------------------------------------------
        # TITLE
        # --------------------------------------------------

        for title in as_list(
            metadata.get("title")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Title",
                attrib={
                    "{http://www.w3.org/XML/1998/namespace}lang": "en"
                }
            ).text = safe_text(title)

        # --------------------------------------------------
        # SUBTITLE
        # --------------------------------------------------

        for subtitle in as_list(
            metadata.get("subtitle")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Subtitle",
                attrib={
                    "{http://www.w3.org/XML/1998/namespace}lang": "en"
                }
            ).text = safe_text(subtitle)

        # --------------------------------------------------
        # HOST PUBLICATION / PART OF
        # --------------------------------------------------

        host_titles = as_list(
            metadata.get("host_title")
        )

        if host_titles:
            part_of_el = ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}PartOf"
            )

            host_pub_el = ET.SubElement(
                part_of_el,
                f"{{{ns_cerif}}}Publication"
            )

            ET.SubElement(
                host_pub_el,
                f"{{{ns_pubt}}}Type"
            ).text = COAR_TYPE_MAP["book"]

            for host_title in host_titles:
                ET.SubElement(
                    host_pub_el,
                    f"{{{ns_cerif}}}Title",
                    attrib={
                        "{http://www.w3.org/XML/1998/namespace}lang": "en"
                    }
                ).text = safe_text(host_title)

            for host_subtitle in as_list(
                metadata.get("host_subtitle")
            ):
                ET.SubElement(
                    host_pub_el,
                    f"{{{ns_cerif}}}Subtitle",
                    attrib={
                        "{http://www.w3.org/XML/1998/namespace}lang": "en"
                    }
                ).text = safe_text(host_subtitle)

        # --------------------------------------------------
        # ABSTRACT
        # --------------------------------------------------

        for abstract in as_list(
            metadata.get("abstract")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Abstract",
                attrib={
                    "{http://www.w3.org/XML/1998/namespace}lang": "en"
                }
            ).text = safe_text(abstract)

        # --------------------------------------------------
        # KEYWORDS
        # --------------------------------------------------

        for kw in as_list(
            metadata.get("keywords")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Keyword",
                attrib={
                    "{http://www.w3.org/XML/1998/namespace}lang": "en"
                }
            ).text = safe_text(kw)

        # --------------------------------------------------
        # DOI
        # --------------------------------------------------

        for doi in as_list(
            metadata.get("doi")
        ):
            doi_raw = safe_text(doi).replace(
                "https://doi.org/",
                ""
            )

            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}DOI"
            ).text = doi_raw

        # --------------------------------------------------
        # AUTHORS
        # --------------------------------------------------

        authors_el = ET.SubElement(
            pub_el,
            f"{{{ns_cerif}}}Authors"
        )

        for author in as_list(
            metadata.get("authors")
        ):
            author_name = safe_text(
                author.get("name")
            )

            affiliations = get_affiliations(
                author
            )

            author_el = ET.SubElement(
                authors_el,
                f"{{{ns_cerif}}}Author"
            )

            ET.SubElement(
                author_el,
                f"{{{ns_cerif}}}DisplayName"
            ).text = author_name

            person_el = ET.SubElement(
                author_el,
                f"{{{ns_cerif}}}Person",
                id=str(uuid.uuid4())
            )

            name_el = ET.SubElement(
                person_el,
                f"{{{ns_cerif}}}PersonName"
            )

            if " " in author_name:
                first, last = author_name.split(
                    " ",
                    1
                )
            else:
                first, last = author_name, ""

            ET.SubElement(
                name_el,
                f"{{{ns_cerif}}}FirstNames"
            ).text = safe_text(first)

            ET.SubElement(
                name_el,
                f"{{{ns_cerif}}}FamilyNames"
            ).text = safe_text(last)

            for aff in affiliations:
                aff_el = ET.SubElement(
                    author_el,
                    f"{{{ns_cerif}}}Affiliation"
                )

                org_el = ET.SubElement(
                    aff_el,
                    f"{{{ns_cerif}}}OrgUnit"
                )

                ET.SubElement(
                    org_el,
                    f"{{{ns_cerif}}}Name",
                    attrib={
                        "{http://www.w3.org/XML/1998/namespace}lang": "en"
                    }
                ).text = aff

        # --------------------------------------------------
        # CONFERENCE / EVENT
        # --------------------------------------------------

        for conf_name in as_list(
            metadata.get("conference_name")
        ):
            presented_at = ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}PresentedAt"
            )

            event = ET.SubElement(
                presented_at,
                f"{{{ns_cerif}}}Event"
            )

            for acronym in as_list(
                metadata.get("conference_acronym")
            ):
                ET.SubElement(
                    event,
                    f"{{{ns_cerif}}}Acronym"
                ).text = safe_text(acronym)

            ET.SubElement(
                event,
                f"{{{ns_cerif}}}Name",
                attrib={
                    "{http://www.w3.org/XML/1998/namespace}lang": "en"
                }
            ).text = safe_text(conf_name)

            for place in as_list(
                metadata.get("conference_place")
            ):
                ET.SubElement(
                    event,
                    f"{{{ns_cerif}}}Place"
                ).text = safe_text(place)

            for country in as_list(
                metadata.get("conference_country")
            ):
                ET.SubElement(
                    event,
                    f"{{{ns_cerif}}}Country"
                ).text = safe_text(country).lower()

            dates = metadata.get(
                "conference_dates"
            ) or {}

            for start in as_list(
                dates.get("start_date")
            ):
                ET.SubElement(
                    event,
                    f"{{{ns_cerif}}}StartDate"
                ).text = safe_text(start)

            for end in as_list(
                dates.get("end_date")
            ):
                ET.SubElement(
                    event,
                    f"{{{ns_cerif}}}EndDate"
                ).text = safe_text(end)

        # --------------------------------------------------
        # JOURNAL
        # --------------------------------------------------

        for journal in as_list(
            metadata.get("journal")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}PublishedIn"
            ).text = safe_text(journal)

        # --------------------------------------------------
        # PUBLICATION DATE
        # --------------------------------------------------

        for year in as_list(
            metadata.get("year")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}PublicationDate"
            ).text = safe_text(year)

        # --------------------------------------------------
        # VOLUME
        # --------------------------------------------------

        for volume in as_list(
            metadata.get("volume")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Volume"
            ).text = safe_text(volume)

        # --------------------------------------------------
        # EDITION
        # --------------------------------------------------

        for edition in as_list(
            metadata.get("edition")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Edition"
            ).text = safe_text(edition)

        # --------------------------------------------------
        # ISBN PRINT
        # --------------------------------------------------

        for isbn_print in as_list(
            metadata.get("isbn_print")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}ISBN",
                attrib={
                    "medium": "http://issn.org/vocabularies/Medium#Print"
                }
            ).text = safe_text(isbn_print)

        # --------------------------------------------------
        # ISBN PRINT 2
        # --------------------------------------------------

        for isbn_print_2 in as_list(
            metadata.get("isbn_print_2")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}ISBN",
                attrib={
                    "medium": "http://issn.org/vocabularies/Medium#Print"
                }
            ).text = safe_text(isbn_print_2)

        # --------------------------------------------------
        # ISBN ONLINE
        # --------------------------------------------------

        for isbn_online in as_list(
            metadata.get("isbn_online")
        ):
            ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}ISBN",
                attrib={
                    "medium": "http://issn.org/vocabularies/Medium#Online"
                }
            ).text = safe_text(isbn_online)

        # --------------------------------------------------
        # PUBLISHERS
        # --------------------------------------------------

        publishers_list = as_list(
            metadata.get("publisher")
        )

        if publishers_list:
            publishers_el = ET.SubElement(
                pub_el,
                f"{{{ns_cerif}}}Publishers"
            )

            for publisher in publishers_list:
                publisher_el = ET.SubElement(
                    publishers_el,
                    f"{{{ns_cerif}}}Publisher"
                )

                org_el = ET.SubElement(
                    publisher_el,
                    f"{{{ns_cerif}}}OrgUnit"
                )

                ET.SubElement(
                    org_el,
                    f"{{{ns_cerif}}}Name",
                    attrib={
                        "{http://www.w3.org/XML/1998/namespace}lang": "en"
                    }
                ).text = safe_text(publisher)

        # --------------------------------------------------
        # STATUS
        # --------------------------------------------------

        ET.SubElement(
            pub_el,
            f"{{{ns_cerif}}}Status",
            attrib={
                "scheme": "/dk/atira/pure/researchoutput/status"
            }
        ).text = "published"

    indent(root)

    return ET.tostring(
        root,
        encoding="utf-8"
    ).decode("utf-8")


# ----------------------------
# RIS BUILDER
# ----------------------------

def json_to_ris(metadata_list):
    lines = []

    for metadata in metadata_list:
        pub_type_key = safe_text(
            metadata.get("publication_type")
        ).lower()

        ris_type = RIS_TYPE_MAP.get(
            pub_type_key,
            "GEN"
        )

        lines.append(
            f"TY  - {ris_type}"
        )

        for title in as_list(
            metadata.get("title")
        ):
            lines.append(
                f"T1  - {safe_text(title)}"
            )

        for author in as_list(
            metadata.get("authors")
        ):
            lines.append(
                f"AU  - {safe_text(author.get('name'))}"
            )

        for year in as_list(
            metadata.get("year")
        ):
            lines.append(
                f"PY  - {safe_text(year)}"
            )

            lines.append(
                f"Y1  - {safe_text(year)}"
            )

        for abstract in as_list(
            metadata.get("abstract")
        ):
            lines.append(
                f"N2  - {safe_text(abstract)}"
            )

            lines.append(
                f"AB  - {safe_text(abstract)}"
            )

        for kw in as_list(
            metadata.get("keywords")
        ):
            lines.append(
                f"KW  - {safe_text(kw)}"
            )

        for doi in as_list(
            metadata.get("doi")
        ):
            doi_raw = safe_text(doi).replace(
                "https://doi.org/",
                ""
            )

            if doi_raw:
                lines.append(
                    f"U2  - {doi_raw}"
                )

                lines.append(
                    f"DO  - {doi_raw}"
                )

        lines.append(
            "M3  - Conference contribution"
        )

        conf_names = as_list(
            metadata.get("conference_name")
        )

        if conf_names:
            for conf_name in conf_names:
                lines.append(
                    f"BT  - {safe_text(conf_name)}"
                )

                lines.append(
                    f"T2  - {safe_text(conf_name)}"
                )

            dates = metadata.get(
                "conference_dates"
            ) or {}

            start = (
                safe_text(
                    as_list(
                        dates.get("start_date")
                    )[0]
                )
                if as_list(
                    dates.get("start_date")
                )
                else ""
            )

            end = (
                safe_text(
                    as_list(
                        dates.get("end_date")
                    )[0]
                )
                if as_list(
                    dates.get("end_date")
                )
                else ""
            )

            if start or end:
                lines.append(
                    f"Y2  - {start} through {end}"
                )

        else:
            for journal in as_list(
                metadata.get("journal")
            ):
                lines.append(
                    f"JO  - {safe_text(journal)}"
                )

                lines.append(
                    f"T2  - {safe_text(journal)}"
                )

        for publisher in as_list(
            metadata.get("publisher")
        ):
            lines.append(
                f"PB  - {safe_text(publisher)}"
            )

        lines.append("ER  - ")
        lines.append("")

    return "\n".join(lines)


# ----------------------------
# BATCH RUNNER
# ----------------------------

def run_batch(settings, log_q, progress_q):
    pdf_dir = Path(settings.pdf_dir)

    ocr_engine = settings.ocr_engine
    output_ris = settings.output_ris
    output_xml = settings.output_xml
    dpi = settings.dpi

    model_choice = getattr(
        settings,
        "model_choice",
        MODEL_OPTIONS[1]
    )

    model_config = MODEL_CONFIG.get(model_choice)

    if not model_config:
        log_q.put(
            f"ERROR: Unknown metadata model: {model_choice}"
        )
        return

    backend = model_config["backend"]
    model_name = model_config["model"]
    model_cache_key = model_config["cache_key"]

    # Manual event/conference override
    event_override = {
        "acronym": getattr(
            settings,
            "event_acronym",
            ""
        ).strip(),

        "name": getattr(
            settings,
            "event_name",
            ""
        ).strip(),

        "place": getattr(
            settings,
            "event_place",
            ""
        ).strip(),

        "country": getattr(
            settings,
            "event_country",
            ""
        ).strip(),

        "start_date": getattr(
            settings,
            "event_start_date",
            ""
        ).strip(),

        "end_date": getattr(
            settings,
            "event_end_date",
            ""
        ).strip(),
    }

    # Manual publication detail override
    pub_override = {
        "publication_date": getattr(
            settings,
            "publication_date",
            ""
        ).strip(),

        "volume": getattr(
            settings,
            "volume",
            ""
        ).strip(),

        "edition": getattr(
            settings,
            "edition",
            ""
        ).strip(),

        "isbn_print": getattr(
            settings,
            "isbn_print",
            ""
        ).strip(),

        "isbn_print_2": getattr(
            settings,
            "isbn_print_2",
            ""
        ).strip(),

        "isbn_online": getattr(
            settings,
            "isbn_online",
            ""
        ).strip(),

        "publisher": getattr(
            settings,
            "publisher",
            ""
        ).strip(),

        "host_title": getattr(
            settings,
            "host_title",
            ""
        ).strip(),

        "host_subtitle": getattr(
            settings,
            "host_subtitle",
            ""
        ).strip(),
    }

    files = sorted(
        pdf_dir.glob("*.pdf")
    )

    total = len(files)

    # ----------------------------
    # CACHE CLEANUP
    # ----------------------------

    if settings.clear_cache:
        log_q.put(
            "Clearing cache..."
        )

        cache_root = pdf_dir / "cache"

        if cache_root.is_dir():
            import shutil

            shutil.rmtree(
                cache_root,
                ignore_errors=True
            )

        log_q.put(
            "Cache cleared."
        )

    if not total:
        log_q.put(
            "No PDF files found."
        )

        return

    metadata_list = []
    failed_files = []

    log_q.put(
        f"Starting batch: {total} PDF(s)"
    )

    if backend == "ollama":
        log_q.put(
            f"Metadata model: Local Ollama — {model_name} | "
            f"Hard timeout: {OLLAMA_TIMEOUT}s | "
            f"Max tokens: {OLLAMA_MAX_TOKENS} | Thinking: off"
        )
    else:
        log_q.put(
            f"Metadata model: OpenAI — {model_name} | "
            f"Timeout: {OPENAI_TIMEOUT}s | "
            f"Max tokens: {OPENAI_MAX_TOKENS}"
        )

    for i, pdf_file in enumerate(
        files,
        start=1
    ):
        base_name = pdf_file.stem
        output_dir = pdf_file.parent

        log_q.put(
            f"[{i}/{total}] Processing: {pdf_file.name}"
        )

        # Cache folder
        cache_dir = (
            output_dir
            / "cache"
            / ocr_engine
        )

        cache_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        # Cache files
        ocr_cache_path = (
            cache_dir
            / f"{base_name}_total_{ocr_engine}.txt"
        )

        # Keep OCR cache independent of the selected model, but keep
        # metadata caches model-specific so OpenAI and Ollama results
        # are never mixed.
        json_path = (
            cache_dir
            / f"{base_name}_{ocr_engine}_{model_cache_key}.json"
        )

        # ----------------------------
        # OCR
        # ----------------------------

        if ocr_cache_path.exists():
            log_q.put(
                f"[{i}/{total}] OCR cache found: {ocr_cache_path.name}"
            )

            try:
                ocr_text = ocr_cache_path.read_text(
                    encoding="utf-8"
                )
            except Exception as e:
                log_q.put(
                    f"[{i}/{total}] ERROR reading OCR cache: {e}"
                )
                failed_files.append(pdf_file.name)
                progress_q.put((i, total))
                continue

        else:
            log_q.put(
                f"[{i}/{total}] OCR started ({ocr_engine})"
            )

            try:
                ocr_text = ocr_pdf(
                    pdf_file,
                    ocr_engine,
                    dpi,
                    settings.first_page,
                    settings.last_page
                )

                ocr_cache_path.write_text(
                    ocr_text,
                    encoding="utf-8"
                )

                log_q.put(
                    f"[{i}/{total}] OCR finished: {len(ocr_text):,} characters"
                )

            except Exception as e:
                log_q.put(
                    f"[{i}/{total}] ERROR (OCR) {pdf_file.name}: {e}"
                )

                failed_files.append(pdf_file.name)
                progress_q.put((i, total))
                continue

        if not ocr_text.strip():
            log_q.put(
                f"[{i}/{total}] ERROR: OCR produced no text; skipping PDF"
            )
            failed_files.append(pdf_file.name)
            progress_q.put((i, total))
            continue

        # ----------------------------
        # LOCAL MODEL EXTRACTION
        # ----------------------------

        if json_path.exists():
            log_q.put(
                f"[{i}/{total}] JSON cache found: {json_path.name}"
            )

            try:
                metadata_json = json.loads(
                    json_path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception as e:
                log_q.put(
                    f"[{i}/{total}] ERROR (JSON load) {json_path.name}: {e}"
                )

                failed_files.append(pdf_file.name)
                progress_q.put((i, total))
                continue

        else:
            log_q.put(
                f"[{i}/{total}] Sending metadata to {model_choice}"
            )

            try:
                metadata_json = extract_metadata_json(
                    ocr_text,
                    model_choice=model_choice,
                    prompt_mode=settings.prompt_mode,
                    prompt_file=settings.prompt_file,
                    log_callback=lambda msg: log_q.put(
                        f"[{i}/{total}] {msg}"
                    )
                )

                json_path.write_text(
                    json.dumps(
                        metadata_json,
                        indent=2,
                        ensure_ascii=False
                    ),
                    encoding="utf-8"
                )

                log_q.put(
                    f"[{i}/{total}] Metadata extraction finished"
                )

            except Exception as e:
                log_q.put(
                    f"[{i}/{total}] ERROR ({model_name}) {pdf_file.name}: {e}"
                )

                failed_files.append(pdf_file.name)
                progress_q.put((i, total))
                continue

        metadata_list.append(
            metadata_json
        )

        log_q.put(
            f"[{i}/{total}] Completed: {pdf_file.name}"
        )

        progress_q.put(
            (i, total)
        )

    # ----------------------------
    # WRITE OUTPUT
    # ----------------------------

    if not metadata_list:
        log_q.put(
            "No metadata collected — nothing to write."
        )

        if failed_files:
            log_q.put(
                f"Failed PDFs: {len(failed_files)}"
            )

        return

    if settings.output_ris:
        output_path = (
            pdf_dir
            / f"combined_{ocr_engine}.ris"
        )

        log_q.put(
            "Writing RIS..."
        )

        try:
            output_text = json_to_ris(
                metadata_list
            )

            output_path.write_text(
                output_text,
                encoding="utf-8"
            )

            log_q.put(
                f"RIS saved: {output_path}"
            )

        except Exception as e:
            log_q.put(
                f"ERROR (RIS write): {e}"
            )

    if settings.output_xml:
        output_path = (
            pdf_dir
            / f"combined_{ocr_engine}.xml"
        )

        log_q.put(
            "Writing XML..."
        )

        try:
            output_text = json_to_oai_pmh(
                metadata_list,
                event_override=event_override,
                pub_override=pub_override
            )

            output_path.write_text(
                output_text,
                encoding="utf-8"
            )

            log_q.put(
                f"XML saved: {output_path}"
            )

        except Exception as e:
            log_q.put(
                f"ERROR (XML write): {e}"
            )

    log_q.put(
        f"Batch finished: {len(metadata_list)}/{total} PDF(s) completed."
    )

    if failed_files:
        log_q.put(
            f"Skipped {len(failed_files)} PDF(s):"
        )

        for failed_name in failed_files:
            log_q.put(
                f"  - {failed_name}"
            )


# ----------------------------
# SETTINGS MODEL
# ----------------------------

class Settings:
    def __init__(self):
        self.pdf_dir = ""
        self.ocr_engine = "pymupdf"
        self.output_format = "ris"
        self.dpi = 300
        self.output_ris = True
        self.output_xml = True

        self.model_choice = MODEL_OPTIONS[1]
        self.env_file = str(env_path)

        self.prompt_mode = "base"
        self.prompt_file = ""
        self.clear_cache = True

        # OCR page range
        self.first_page = 1
        self.last_page = 2

        # Manual event/conference override fields
        self.event_acronym = ""
        self.event_name = ""
        self.event_place = ""
        self.event_country = ""
        self.event_start_date = ""
        self.event_end_date = ""

        # Manual publication detail fields
        self.publication_date = ""
        self.volume = ""
        self.edition = ""
        self.isbn_print = ""
        self.isbn_print_2 = ""
        self.isbn_online = ""
        self.publisher = ""
        self.host_title = ""
        self.host_subtitle = ""


# ----------------------------
# GUI
# ----------------------------

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.title("Aiamea")
        self.geometry("900x800")
        self.minsize(700, 500)

        self.settings = Settings()

        self.log_q = queue.Queue()
        self.progress_q = queue.Queue()

        self.build_ui()

        self.after(
            100,
            self.poll_queues
        )

        icon_path = resource_path(
            "Aiamea.ico"
        )

        if icon_path.exists():
            try:
                self.iconbitmap(
                    str(icon_path)
                )

            except Exception as e:
                print(
                    "Icon load failed:",
                    e
                )

    def build_ui(self):
        # =========================================================
        # ROOT WINDOW LAYOUT
        # =========================================================

        self.columnconfigure(0, weight=1)

        # Configuration area gets flexible height.
        # Log also gets flexible height.
        self.rowconfigure(0, weight=1)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)
        self.rowconfigure(3, weight=0)

        # =========================================================
        # SCROLLABLE CONFIGURATION AREA
        # =========================================================

        scroll_container = ttk.Frame(self)
        scroll_container.grid(
            row=0,
            column=0,
            sticky="nsew",
            padx=8,
            pady=(8, 4)
        )

        scroll_container.columnconfigure(0, weight=1)
        scroll_container.rowconfigure(0, weight=1)

        config_canvas = tk.Canvas(
            scroll_container,
            highlightthickness=0,
            borderwidth=0
        )

        config_scrollbar = ttk.Scrollbar(
            scroll_container,
            orient="vertical",
            command=config_canvas.yview
        )

        config_canvas.configure(
            yscrollcommand=config_scrollbar.set
        )

        config_canvas.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        config_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        # This frame contains all configuration controls.
        scroll_frame = ttk.Frame(
            config_canvas,
            padding=8
        )

        canvas_window = config_canvas.create_window(
            (0, 0),
            window=scroll_frame,
            anchor="nw"
        )

        scroll_frame.columnconfigure(
            0,
            weight=1
        )

        # =========================================================
        # CANVAS RESIZING
        # =========================================================

        def update_scrollregion(event=None):
            config_canvas.configure(
                scrollregion=config_canvas.bbox("all")
            )

        scroll_frame.bind(
            "<Configure>",
            update_scrollregion
        )

        def resize_inner_frame(event):
            config_canvas.itemconfigure(
                canvas_window,
                width=event.width
            )

        config_canvas.bind(
            "<Configure>",
            resize_inner_frame
        )

        # =========================================================
        # MOUSE WHEEL SUPPORT
        # =========================================================

        def on_mousewheel(event):
            # Windows / macOS
            if event.delta:
                config_canvas.yview_scroll(
                    int(-1 * (event.delta / 120)),
                    "units"
                )

        def on_linux_scroll_up(event):
            config_canvas.yview_scroll(
                -1,
                "units"
            )

        def on_linux_scroll_down(event):
            config_canvas.yview_scroll(
                1,
                "units"
            )

        def enter_scroll_area(event):
            config_canvas.bind_all(
                "<MouseWheel>",
                on_mousewheel
            )

            config_canvas.bind_all(
                "<Button-4>",
                on_linux_scroll_up
            )

            config_canvas.bind_all(
                "<Button-5>",
                on_linux_scroll_down
            )

        def leave_scroll_area(event):
            config_canvas.unbind_all(
                "<MouseWheel>"
            )

            config_canvas.unbind_all(
                "<Button-4>"
            )

            config_canvas.unbind_all(
                "<Button-5>"
            )

        scroll_container.bind(
            "<Enter>",
            enter_scroll_area
        )

        scroll_container.bind(
            "<Leave>",
            leave_scroll_area
        )

        # =========================================================
        # INPUT / MODEL
        # =========================================================

        top = ttk.LabelFrame(
            scroll_frame,
            text="Input & Model",
            padding=10
        )

        top.grid(
            row=0,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        top.columnconfigure(
            0,
            weight=1
        )

        # -------------------------
        # PDF FOLDER
        # -------------------------

        ttk.Label(
            top,
            text="PDF Folder"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.folder_var = tk.StringVar()

        ttk.Entry(
            top,
            textvariable=self.folder_var
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 8),
            pady=(3, 6)
        )

        ttk.Button(
            top,
            text="Browse",
            command=self.pick_folder
        ).grid(
            row=1,
            column=1,
            sticky="e",
            pady=(3, 6)
        )

        # -------------------------
        # ENVIRONMENT FILE
        # -------------------------

        ttk.Label(
            top,
            text="Environment File"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(5, 0)
        )

        self.env_var = tk.StringVar(
            value=str(env_path)
        )

        self.env_entry = ttk.Entry(
            top,
            textvariable=self.env_var
        )

        self.env_entry.grid(
            row=3,
            column=0,
            sticky="ew",
            padx=(0, 8),
            pady=(3, 6)
        )

        self.env_btn = ttk.Button(
            top,
            text="Browse",
            command=self.pick_env
        )

        self.env_btn.grid(
            row=3,
            column=1,
            sticky="e",
            pady=(3, 6)
        )

        # -------------------------
        # METADATA MODEL
        # -------------------------

        ttk.Label(
            top,
            text="Metadata Model"
        ).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(5, 0)
        )

        self.model_var = tk.StringVar(
            value=MODEL_OPTIONS[1]
        )

        self.model_combo = ttk.Combobox(
            top,
            values=MODEL_OPTIONS,
            textvariable=self.model_var,
            state="readonly"
        )

        self.model_combo.grid(
            row=5,
            column=0,
            sticky="ew",
            pady=(3, 6)
        )

        self.model_combo.bind(
            "<<ComboboxSelected>>",
            lambda event: self.toggle_model_controls()
        )

        # -------------------------
        # PROMPT FILE
        # -------------------------

        ttk.Label(
            top,
            text="Prompt File"
        ).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(5, 0)
        )

        self.prompt_var = tk.StringVar(
            value=""
        )

        self.prompt_entry = ttk.Entry(
            top,
            textvariable=self.prompt_var
        )

        self.prompt_entry.grid(
            row=7,
            column=0,
            sticky="ew",
            padx=(0, 8),
            pady=(3, 0)
        )

        self.prompt_btn = ttk.Button(
            top,
            text="Browse",
            command=self.pick_prompt
        )

        self.prompt_btn.grid(
            row=7,
            column=1,
            sticky="e"
        )

        # =========================================================
        # OPTIONS
        # =========================================================

        opts = ttk.LabelFrame(
            scroll_frame,
            text="Options",
            padding=10
        )

        opts.grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        for col in range(5):
            opts.columnconfigure(
                col,
                weight=1
            )

        # -------------------------
        # SCAN METHOD
        # -------------------------

        self.ocr_var = tk.StringVar(
            value="pymupdf"
        )

        ttk.Label(
            opts,
            text="Scan Method"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        ttk.Radiobutton(
            opts,
            text="Copy Text",
            variable=self.ocr_var,
            value="pymupdf"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )

        ttk.Radiobutton(
            opts,
            text="OCR Tesseract",
            variable=self.ocr_var,
            value="tesseract"
        ).grid(
            row=2,
            column=0,
            sticky="w"
        )

        # -------------------------
        # OUTPUT
        # -------------------------

        self.out_ris = tk.BooleanVar(
            value=True
        )

        self.out_xml = tk.BooleanVar(
            value=True
        )

        ttk.Label(
            opts,
            text="Output"
        ).grid(
            row=0,
            column=1,
            sticky="w"
        )

        ttk.Checkbutton(
            opts,
            text="RIS",
            variable=self.out_ris
        ).grid(
            row=1,
            column=1,
            sticky="w"
        )

        ttk.Checkbutton(
            opts,
            text="XML",
            variable=self.out_xml
        ).grid(
            row=2,
            column=1,
            sticky="w"
        )

        # -------------------------
        # DPI
        # -------------------------

        self.dpi_values = [
            200,
            300,
            400,
            450
        ]

        self.dpi_var = tk.IntVar(
            value=self.settings.dpi
        )

        ttk.Label(
            opts,
            text="DPI"
        ).grid(
            row=0,
            column=2,
            sticky="w"
        )

        self.dpi_combo = ttk.Combobox(
            opts,
            values=self.dpi_values,
            textvariable=self.dpi_var,
            state="readonly",
            width=8
        )

        self.dpi_combo.grid(
            row=1,
            column=2,
            sticky="w"
        )

        # -------------------------
        # PAGE RANGE
        # -------------------------

        self.first_page_var = tk.IntVar(
            value=1
        )

        self.last_page_var = tk.StringVar(
            value="2"
        )

        ttk.Label(
            opts,
            text="Page Range"
        ).grid(
            row=0,
            column=3,
            sticky="w"
        )

        page_range_frame = ttk.Frame(
            opts
        )

        page_range_frame.grid(
            row=1,
            column=3,
            rowspan=2,
            sticky="w"
        )

        ttk.Label(
            page_range_frame,
            text="First"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        ttk.Entry(
            page_range_frame,
            textvariable=self.first_page_var,
            width=5
        ).grid(
            row=0,
            column=1,
            padx=(4, 0)
        )

        ttk.Label(
            page_range_frame,
            text="Last"
        ).grid(
            row=1,
            column=0,
            sticky="w"
        )

        ttk.Entry(
            page_range_frame,
            textvariable=self.last_page_var,
            width=5
        ).grid(
            row=1,
            column=1,
            padx=(4, 0)
        )

        # -------------------------
        # PROMPT MODE
        # -------------------------

        self.prompt_mode_var = tk.StringVar(
            value="base"
        )

        ttk.Label(
            opts,
            text="Prompt Mode"
        ).grid(
            row=0,
            column=4,
            sticky="w"
        )

        ttk.Radiobutton(
            opts,
            text="Default Prompt",
            variable=self.prompt_mode_var,
            value="base",
            command=self.toggle_prompt_mode
        ).grid(
            row=1,
            column=4,
            sticky="w"
        )

        ttk.Radiobutton(
            opts,
            text="Custom Prompt",
            variable=self.prompt_mode_var,
            value="custom",
            command=self.toggle_prompt_mode
        ).grid(
            row=2,
            column=4,
            sticky="w"
        )

        # =========================================================
        # CONFERENCE / EVENT
        # =========================================================

        event_frame = ttk.LabelFrame(
            scroll_frame,
            text=(
                "Conference / Event "
                "(optional — overwrites extracted data in XML)"
            ),
            padding=10
        )

        event_frame.grid(
            row=2,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        for col in range(4):
            event_frame.columnconfigure(
                col,
                weight=1
            )

        self.event_acronym_var = tk.StringVar()
        self.event_name_var = tk.StringVar()
        self.event_place_var = tk.StringVar()
        self.event_country_var = tk.StringVar()
        self.event_start_date_var = tk.StringVar()
        self.event_end_date_var = tk.StringVar()

        # Row 0

        ttk.Label(
            event_frame,
            text="Acronym"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        ttk.Label(
            event_frame,
            text="Name"
        ).grid(
            row=0,
            column=1,
            sticky="w"
        )

        ttk.Label(
            event_frame,
            text="Place"
        ).grid(
            row=0,
            column=2,
            sticky="w"
        )

        ttk.Label(
            event_frame,
            text="Country (ISO code)"
        ).grid(
            row=0,
            column=3,
            sticky="w"
        )

        # Row 1

        ttk.Entry(
            event_frame,
            textvariable=self.event_acronym_var
        ).grid(
            row=1,
            column=0,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            event_frame,
            textvariable=self.event_name_var
        ).grid(
            row=1,
            column=1,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            event_frame,
            textvariable=self.event_place_var
        ).grid(
            row=1,
            column=2,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            event_frame,
            textvariable=self.event_country_var
        ).grid(
            row=1,
            column=3,
            sticky="ew"
        )

        # Row 2

        ttk.Label(
            event_frame,
            text="Start Date (YYYY-MM-DD)"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            event_frame,
            text="End Date (YYYY-MM-DD)"
        ).grid(
            row=2,
            column=1,
            sticky="w",
            pady=(8, 0)
        )

        # Row 3

        ttk.Entry(
            event_frame,
            textvariable=self.event_start_date_var
        ).grid(
            row=3,
            column=0,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            event_frame,
            textvariable=self.event_end_date_var
        ).grid(
            row=3,
            column=1,
            sticky="ew",
            padx=(0, 8)
        )

        # =========================================================
        # PUBLICATION DETAILS
        # =========================================================

        pubdetails_frame = ttk.LabelFrame(
            scroll_frame,
            text="Publication Details (optional)",
            padding=10
        )

        pubdetails_frame.grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(0, 8)
        )

        for col in range(3):
            pubdetails_frame.columnconfigure(
                col,
                weight=1
            )

        self.publication_date_var = tk.StringVar()
        self.volume_var = tk.StringVar()
        self.edition_var = tk.StringVar()
        self.isbn_print_var = tk.StringVar()
        self.isbn_print_2_var = tk.StringVar()
        self.isbn_online_var = tk.StringVar()
        self.publisher_var = tk.StringVar()
        self.host_title_var = tk.StringVar()
        self.host_subtitle_var = tk.StringVar()

        # -------------------------
        # TITLE
        # -------------------------

        ttk.Label(
            pubdetails_frame,
            text="Title"
        ).grid(
            row=0,
            column=0,
            sticky="w"
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.host_title_var
        ).grid(
            row=1,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(3, 0)
        )

        # -------------------------
        # SUBTITLE
        # -------------------------

        ttk.Label(
            pubdetails_frame,
            text="Subtitle"
        ).grid(
            row=2,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.host_subtitle_var
        ).grid(
            row=3,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(3, 0)
        )

        # -------------------------
        # DATE / VOLUME / EDITION
        # -------------------------

        ttk.Label(
            pubdetails_frame,
            text="Publication Date (year)"
        ).grid(
            row=4,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            pubdetails_frame,
            text="Volume"
        ).grid(
            row=4,
            column=1,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            pubdetails_frame,
            text="Edition"
        ).grid(
            row=4,
            column=2,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.publication_date_var
        ).grid(
            row=5,
            column=0,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.volume_var
        ).grid(
            row=5,
            column=1,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.edition_var
        ).grid(
            row=5,
            column=2,
            sticky="ew"
        )

        # -------------------------
        # ISBN
        # -------------------------

        ttk.Label(
            pubdetails_frame,
            text="ISBN (Print)"
        ).grid(
            row=6,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            pubdetails_frame,
            text="ISBN (Print) 2"
        ).grid(
            row=6,
            column=1,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Label(
            pubdetails_frame,
            text="ISBN (Online)"
        ).grid(
            row=6,
            column=2,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.isbn_print_var
        ).grid(
            row=7,
            column=0,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.isbn_print_2_var
        ).grid(
            row=7,
            column=1,
            sticky="ew",
            padx=(0, 8)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.isbn_online_var
        ).grid(
            row=7,
            column=2,
            sticky="ew"
        )

        # -------------------------
        # PUBLISHER
        # -------------------------

        ttk.Label(
            pubdetails_frame,
            text="Publisher"
        ).grid(
            row=8,
            column=0,
            sticky="w",
            pady=(8, 0)
        )

        ttk.Entry(
            pubdetails_frame,
            textvariable=self.publisher_var
        ).grid(
            row=9,
            column=0,
            columnspan=3,
            sticky="ew",
            pady=(3, 0)
        )

        # =========================================================
        # ACTIONS - FIXED AT BOTTOM OF CONFIG AREA
        # =========================================================

        actions = ttk.Frame(
            self,
            padding=(8, 4)
        )

        actions.grid(
            row=1,
            column=0,
            sticky="ew"
        )

        actions.columnconfigure(
            1,
            weight=1
        )

        self.run_btn = ttk.Button(
            actions,
            text="Extract Metadata",
            command=self.start_run
        )

        self.run_btn.grid(
            row=0,
            column=0,
            sticky="w"
        )

        self.pb = ttk.Progressbar(
            actions,
            mode="determinate"
        )

        self.pb.grid(
            row=0,
            column=1,
            sticky="ew",
            padx=(10, 0)
        )

        # =========================================================
        # LOG
        # =========================================================

        logf = ttk.LabelFrame(
            self,
            text="Log",
            padding=5
        )

        logf.grid(
            row=2,
            column=0,
            sticky="nsew",
            padx=8,
            pady=4
        )

        logf.columnconfigure(
            0,
            weight=1
        )

        logf.rowconfigure(
            0,
            weight=1
        )

        self.txt = tk.Text(
            logf,
            wrap="word"
        )

        self.txt.grid(
            row=0,
            column=0,
            sticky="nsew"
        )

        log_scrollbar = ttk.Scrollbar(
            logf,
            orient="vertical",
            command=self.txt.yview
        )

        log_scrollbar.grid(
            row=0,
            column=1,
            sticky="ns"
        )

        self.txt.configure(
            yscrollcommand=log_scrollbar.set
        )

        # =========================================================
        # ABOUT
        # =========================================================

        about = ttk.Frame(
            self,
            padding=(8, 4)
        )

        about.grid(
            row=3,
            column=0,
            sticky="ew"
        )

        about_text = (
            "- About -\n\n"
            "Aiamea v0.7.1\n\n"
            "Home: https://github.com/tu-delft-library/pdf-ris\n\n"
            "Metadata models: OpenAI gpt-4o-mini / Ollama qwen3:8b\n\n"
            "Copyright © 2026 TU Delft\n"
            "H.D. Nguyen and J. Stelma\n\n"
            "Aiamea is shared under a Creative Commons "
            "Attribution-NonCommercial-ShareAlike license "
            "(CC-BY-NC-SA).\n\n"
            "This program is distributed in the hope that it will "
            "be useful. However,\n"
            "THE LICENSED SOFTWARE IS PROVIDED \"AS IS\", "
            "WITHOUT WARRANTY OF ANY KIND.\n"
            "Metadata extraction: OpenAI gpt-4o-mini or "
            "local Ollama qwen3:8b.\n\n"
            "The user is solely responsible for ANYTHING that "
            "happens as a result of using this software."
        )

        ttk.Label(
            about,
            text=about_text,
            anchor="w",
            justify="left",
            wraplength=850,
            foreground="gray"
        ).pack(
            fill="x"
        )

        # =========================================================
        # INITIAL GUI STATE
        # =========================================================

        self.toggle_prompt_mode()
        self.toggle_model_controls()

        # Force the canvas to calculate its scroll region after
        # all widgets have been created.
        self.after_idle(
            update_scrollregion
        )

    # ----------------------------
    # FOLDER
    # ----------------------------

    def pick_folder(self):
        p = filedialog.askdirectory()

        if p:
            self.folder_var.set(p)

    # ----------------------------
    # START RUN
    # ----------------------------

    def start_run(self):
        if not self.folder_var.get():
            messagebox.showerror(
                "Missing folder",
                "Select a PDF folder first."
            )

            return

        # Check the selected metadata model
        selected_model = self.model_var.get()

        if selected_model not in MODEL_CONFIG:
            messagebox.showerror(
                "Invalid model",
                "Please select a metadata model."
            )
            return

        self.settings.model_choice = selected_model

        selected_backend = MODEL_CONFIG[selected_model]["backend"]

        if selected_backend == "ollama":
            try:
                from urllib.request import urlopen

                with urlopen(
                    "http://127.0.0.1:11434/api/tags",
                    timeout=10
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}"
                        )

                self.log_status(
                    f"Local model ready: {MODEL_CONFIG[selected_model]['model']}"
                )

            except Exception as e:
                self.log_status(
                    f"Cannot connect to Ollama at "
                    f"{OLLAMA_BASE_URL}: {e}",
                    "ERROR"
                )

                messagebox.showerror(
                    "Ollama not available",
                    "Cannot connect to Ollama.\n\n"
                    "Make sure Ollama is running and that the "
                    "selected local model is installed."
                )

                return

        else:
            selected_env = self.env_var.get().strip()

            if not selected_env:
                self.log_status(
                    "No .env file selected.",
                    "ERROR"
                )
                messagebox.showerror(
                    "Missing .env",
                    "Select an environment file for OpenAI."
                )
                return

            if not Path(selected_env).exists():
                self.log_status(
                    f"Environment file not found: {selected_env}",
                    "ERROR"
                )
                messagebox.showerror(
                    "Missing .env",
                    f"Environment file not found:\n{selected_env}"
                )
                return

            load_dotenv(selected_env, override=True)
            api_key = os.getenv("OPENAI_API_KEY", "").strip()

            if not api_key:
                self.log_status(
                    f"OPENAI_API_KEY missing in: {selected_env}",
                    "ERROR"
                )

                messagebox.showerror(
                    "OpenAI API key missing",
                    f"OPENAI_API_KEY was not found in:\n{selected_env}"
                )
                return

            self.settings.env_file = selected_env

            self.log_status(
                f"OpenAI ready: {OPENAI_MODEL}"
            )

        self.settings.pdf_dir = (
            self.folder_var.get()
        )

        self.settings.ocr_engine = (
            self.ocr_var.get()
        )

        self.settings.output_ris = (
            self.out_ris.get()
        )

        self.settings.output_xml = (
            self.out_xml.get()
        )

        self.pb["value"] = 0

        self.txt.delete(
            "1.0",
            "end"
        )

        self.run_btn.config(
            state="disabled"
        )

        self.settings.dpi = (
            self.dpi_var.get()
        )

        print(
            f"[DEBUG] Selected DPI = "
            f"{self.settings.dpi}"
        )

        # OCR page range
        try:
            first_page = int(self.first_page_var.get())

        except (TypeError, ValueError):
            messagebox.showerror(
                "Invalid page range",
                "First page must be an integer."
            )
            self.run_btn.config(state="normal")
            return

        last = (
            self.last_page_var.get()
            .strip()
        )

        try:
            last_page = int(last) if last else None

        except ValueError:
            messagebox.showerror(
                "Invalid page range",
                "Last page must be an integer or blank."
            )
            self.run_btn.config(state="normal")
            return

        if first_page < 1 or (last_page is not None and last_page < first_page):
            messagebox.showerror(
                "Invalid page range",
                "Please enter a valid First/Last page range."
            )
            self.run_btn.config(state="normal")
            return

        self.settings.first_page = first_page
        self.settings.last_page = last_page

        print(
            f"[DEBUG] Pages: "
            f"{self.settings.first_page} → "
            f"{self.settings.last_page}"
        )

        self.settings.output_ris = (
            self.out_ris.get()
        )

        self.settings.output_xml = (
            self.out_xml.get()
        )

        self.settings.prompt_file = (
            self.prompt_var.get()
        )

        self.settings.prompt_mode = (
            self.prompt_mode_var.get()
        )

        # Manual event/conference override values

        self.settings.event_acronym = (
            self.event_acronym_var.get()
        )

        self.settings.event_name = (
            self.event_name_var.get()
        )

        self.settings.event_place = (
            self.event_place_var.get()
        )

        self.settings.event_country = (
            self.event_country_var.get()
        )

        self.settings.event_start_date = (
            self.event_start_date_var.get()
        )

        self.settings.event_end_date = (
            self.event_end_date_var.get()
        )

        # Manual publication detail values

        self.settings.publication_date = (
            self.publication_date_var.get()
        )

        self.settings.volume = (
            self.volume_var.get()
        )

        self.settings.edition = (
            self.edition_var.get()
        )

        self.settings.isbn_print = (
            self.isbn_print_var.get()
        )

        self.settings.isbn_print_2 = (
            self.isbn_print_2_var.get()
        )

        self.settings.isbn_online = (
            self.isbn_online_var.get()
        )

        self.settings.publisher = (
            self.publisher_var.get()
        )

        self.settings.host_title = (
            self.host_title_var.get()
        )

        self.settings.host_subtitle = (
            self.host_subtitle_var.get()
        )

        threading.Thread(
            target=self._run_and_reenable,
            daemon=True
        ).start()

    # ----------------------------
    # RUN / REENABLE
    # ----------------------------

    def _run_and_reenable(self):
        """Wrapper so the button re-enables after the batch finishes."""

        try:
            run_batch(
                self.settings,
                self.log_q,
                self.progress_q
            )

        except Exception as e:
            self.log_q.put(
                f"FATAL BATCH ERROR: {e}"
            )

        finally:
            self.after(
                0,
                lambda: self.run_btn.config(
                    state="normal"
                )
            )

    # ----------------------------
    # QUEUES
    # ----------------------------

    def poll_queues(self):
        while not self.log_q.empty():
            msg = self.log_q.get_nowait()

            self.txt.insert(
                "end",
                msg + "\n"
            )

            self.txt.see(
                "end"
            )

        while not self.progress_q.empty():
            cur, total = (
                self.progress_q.get_nowait()
            )

            self.pb["maximum"] = total
            self.pb["value"] = cur

        self.after(
            100,
            self.poll_queues
        )

    # ----------------------------
    # MODEL / ENVIRONMENT FILE
    # ----------------------------

    def toggle_model_controls(self):
        selected_model = self.model_var.get()
        config = MODEL_CONFIG.get(selected_model, {})

        if config.get("backend") == "openai":
            self.env_entry.config(state="normal")
            self.env_btn.config(state="normal")
        else:
            self.env_entry.config(state="disabled")
            self.env_btn.config(state="disabled")

    def pick_env(self):
        p = filedialog.askopenfilename(
            title="Select Environment File",
            filetypes=[
                ("Environment file", "*.env"),
                ("All files", "*.*")
            ]
        )

        if not p:
            self.log_status("No .env file selected", "WARN")
            return

        self.env_var.set(p)
        self.settings.env_file = p
        self.log_status(f".env selected: {p}")

    # ----------------------------
    # PROMPT
    # ----------------------------

    def pick_prompt(self):
        p = filedialog.askopenfilename(
            title="Select Prompt File",
            filetypes=[
                ("Text files", "*.txt"),
                ("All files", "*.*")
            ]
        )

        if not p:
            return

        self.prompt_var.set(
            p
        )

        self.txt.delete(
            "1.0",
            "end"
        )

        try:
            with open(
                p,
                "r",
                encoding="utf-8"
            ) as f:
                prompt_text = f.read()

            self.txt.insert(
                "end",
                f"Prompt file: {p}\n"
            )

            self.txt.insert(
                "end",
                "=" * 80 + "\n"
            )

            self.txt.insert(
                "end",
                prompt_text
            )

        except Exception as e:
            self.txt.insert(
                "end",
                f"ERROR reading prompt file:\n{e}\n"
            )

    def toggle_prompt_mode(self):
        mode = (
            self.prompt_mode_var.get()
        )

        if mode == "base":
            self.prompt_entry.config(
                state="disabled"
            )

            self.prompt_btn.config(
                state="disabled"
            )

        else:
            self.prompt_entry.config(
                state="normal"
            )

            self.prompt_btn.config(
                state="normal"
            )

    # ----------------------------
    # STATUS
    # ----------------------------

    def log_status(
        self,
        msg,
        level="INFO"
    ):
        self.txt.insert(
            "end",
            f"[{level}] {msg}\n"
        )

        self.txt.see(
            "end"
        )


if __name__ == "__main__":
    mp.freeze_support()
    App().mainloop()
