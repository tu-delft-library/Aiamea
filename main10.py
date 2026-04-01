from pathlib import Path
import pandas as pd
import json
import xml.etree.ElementTree as ET
import uuid
import os
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# LOAD OPENAI API KEY
# ----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")
client = OpenAI(api_key=api_key)

# ----------------------------
# SETTINGS
# ----------------------------
PDF_DIR = Path(r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\batches\06")
FIRST_PAGE = 1
LAST_PAGE = 2

# OCR engine toggle: "tesseract" or "pymupdf"
OCR_ENGINE = "pymupdf"

# Output format toggle: "xml" or "ris"
OUTPUT_FORMAT = "ris"

# Tesseract-specific settings (only used if OCR_ENGINE = "tesseract")
OCR_DPI = 450
TESSERACT_CONFIG = "--oem 3 --psm 12"
POPPLER_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\poppler-25.11.0\Library\bin"

# COAR publication type mapping
COAR_TYPE_MAP = {
    "journal": "http://purl.org/coar/resource_type/c_6501",
    "conference": "http://purl.org/coar/resource_type/c_5794",
    "book": "http://purl.org/coar/resource_type/c_1790",
    "chapter": "http://purl.org/coar/resource_type/c_1791",
    "report": "http://purl.org/coar/resource_type/c_1897",
}

# RIS type mapping
RIS_TYPE_MAP = {
    "journal": "JOUR",
    "conference": "GEN",
    "book": "BOOK",
    "chapter": "CHAP",
    "report": "RPRT",
}

# ----------------------------
# OCR FUNCTIONS
# ----------------------------
def ocr_with_tesseract(pdf_path):
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    import numpy as np

    print(f"📄 [Tesseract] OCRing {pdf_path.name}...")
    pages = convert_from_path(
        str(pdf_path), OCR_DPI,
        poppler_path=POPPLER_PATH,
        first_page=FIRST_PAGE,
        last_page=LAST_PAGE
    )
    master_lines = []
    for i, page in enumerate(pages, start=FIRST_PAGE):
        gray = cv2.cvtColor(import_numpy_array(page), cv2.COLOR_RGB2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        data = pytesseract.image_to_data(thresh, config=TESSERACT_CONFIG, output_type="dict")
        df = pd.DataFrame(data)
        df = df[df["conf"].astype(float) > 0]
        page_dict = {}
        for _, row in df.iterrows():
            key = f"{i}_{row['par_num']}_{row['line_num']}"
            text = str(row["text"]).strip()
            if not text:
                continue
            page_dict.setdefault(key, "")
            page_dict[key] += (" " if page_dict[key] else "") + text
        for key in sorted(page_dict.keys()):
            master_lines.append(page_dict[key])
    return " ".join(master_lines)

def import_numpy_array(page):
    import numpy as np
    return np.array(page)

def ocr_with_pymupdf(pdf_path):
    import fitz  # pymupdf

    print(f"📄 [PyMuPDF] Extracting {pdf_path.name}...")
    doc = fitz.open(str(pdf_path))
    all_text = []

    # pymupdf pages are 0-indexed
    start = FIRST_PAGE - 1
    end = min(LAST_PAGE, len(doc))

    for page_num in range(start, end):
        page = doc[page_num]
        text = page.get_text("text")
        all_text.append(text)

    doc.close()
    return " ".join(all_text)

def ocr_pdf(pdf_path):
    if OCR_ENGINE == "pymupdf":
        return ocr_with_pymupdf(pdf_path)
    elif OCR_ENGINE == "tesseract":
        return ocr_with_tesseract(pdf_path)
    else:
        raise ValueError(f"Unknown OCR_ENGINE: '{OCR_ENGINE}'. Use 'tesseract' or 'pymupdf'.")

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
    """Always return a list, whether val is None, a string, or already a list."""
    if val is None:
        return []
    if isinstance(val, list):
        return [v for v in val if v is not None]
    return [val]

def get_affiliations(author):
    """
    Extract affiliations from an author dict.
    Handles both 'affiliations' and 'affiliation' key names.
    Splits on semicolons if multiple affiliations are in one string.
    Returns a list of affiliation strings.
    """
    raw = author.get("affiliations") or author.get("affiliation") or ""
    if isinstance(raw, list):
        # already a list — flatten and split each item on ";"
        result = []
        for item in raw:
            for part in safe_text(item).split(";"):
                part = part.strip()
                if part:
                    result.append(part)
        return result
    else:
        # single string — split on ";"
        return [a.strip() for a in safe_text(raw).split(";") if a.strip()]

# ----------------------------
# GPT EXTRACTION
# ----------------------------
def gpt_extract_json(ocr_text, snippet_length=8000):
    snippet = ocr_text[:snippet_length]

    prompt_json = f"""
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
- Think carefully about how authors and affiliations are connected before producing the final JSON

General rules:

- Return only valid JSON
- No explanations
- No markdown
- No comments
- Missing fields should be null

OCR text:
{snippet}
"""

    print("🤖 Extracting JSON via GPT...")
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_json}]
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    metadata_json = json.loads(raw_output)
    return metadata_json

# ----------------------------
# XML BUILDER
# ----------------------------
def json_to_oai_pmh(metadata_list):
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
        attrib={f"{{{ns_xsi}}}schemaLocation":
                f"{ns_oai} {ns_oai}/OAI-PMH.xsd {ns_cerif} https://www.openaire.eu/schema/cris/current/openaire-cerif-profile.xsd"}
    )

    list_records = ET.SubElement(root, f"{{{ns_oai}}}ListRecords")

    for metadata in metadata_list:
        record = ET.SubElement(list_records, f"{{{ns_oai}}}record")
        metadata_el = ET.SubElement(record, "metadata")

        pub_el = ET.SubElement(metadata_el, f"{{{ns_cerif}}}Publication", id=f"Publications/{uuid.uuid4()}")

        # Type
        for pub_type in as_list(metadata.get("publication_type")):
            pub_type_uri = COAR_TYPE_MAP.get(safe_text(pub_type).lower(), COAR_TYPE_MAP["conference"])
            ET.SubElement(pub_el, f"{{{ns_pubt}}}Type").text = pub_type_uri

        # Language (static, always "en")
        ET.SubElement(pub_el, f"{{{ns_cerif}}}Language").text = "en"

        # Title
        for title in as_list(metadata.get("title")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"}).text = safe_text(title)

        # Subtitle
        for subtitle in as_list(metadata.get("subtitle")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Subtitle", attrib={"xml:lang": "en"}).text = safe_text(subtitle)

        # Abstract
        for abstract in as_list(metadata.get("abstract")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Abstract", attrib={"xml:lang": "en"}).text = safe_text(abstract)

        # Keywords
        for kw in as_list(metadata.get("keywords")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Keyword", attrib={"xml:lang": "en"}).text = safe_text(kw)

        # DOI
        for doi in as_list(metadata.get("doi")):
            doi_raw = safe_text(doi)
            if doi_raw.startswith("https://doi.org/"):
                doi_raw = doi_raw.replace("https://doi.org/", "")
            ET.SubElement(pub_el, f"{{{ns_cerif}}}DOI").text = doi_raw

        # Authors + Affiliations
        authors_el = ET.SubElement(pub_el, f"{{{ns_cerif}}}Authors")
        for author in as_list(metadata.get("authors")):
            author_name = safe_text(author.get("name"))
            affiliations = get_affiliations(author)

            author_el = ET.SubElement(authors_el, f"{{{ns_cerif}}}Author")
            ET.SubElement(author_el, f"{{{ns_cerif}}}DisplayName").text = author_name

            person_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Person", id=str(uuid.uuid4()))
            name_el = ET.SubElement(person_el, f"{{{ns_cerif}}}PersonName")

            if " " in author_name:
                first, last = author_name.split(" ", 1)
            else:
                first, last = author_name, ""

            ET.SubElement(name_el, f"{{{ns_cerif}}}FirstNames").text = safe_text(first)
            ET.SubElement(name_el, f"{{{ns_cerif}}}FamilyNames").text = safe_text(last)

            # One Affiliation/OrgUnit per institution
            for aff in affiliations:
                aff_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Affiliation")
                org_el = ET.SubElement(aff_el, f"{{{ns_cerif}}}OrgUnit")
                ET.SubElement(org_el, f"{{{ns_cerif}}}Name", attrib={"xml:lang": "en"}).text = aff

        # Conference info
        for conf_name in as_list(metadata.get("conference_name")):
            presented_at = ET.SubElement(pub_el, f"{{{ns_cerif}}}PresentedAt")
            event = ET.SubElement(presented_at, "Event")
            for acronym in as_list(metadata.get("conference_acronym")):
                ET.SubElement(event, "Acronym").text = safe_text(acronym)
            ET.SubElement(event, "Name", attrib={"xml:lang": "en"}).text = safe_text(conf_name)
            for place in as_list(metadata.get("conference_place")):
                ET.SubElement(event, "Place").text = safe_text(place)
            for country in as_list(metadata.get("conference_country")):
                ET.SubElement(event, "Country").text = safe_text(country)
            dates = metadata.get("conference_dates") or {}
            for start in as_list(dates.get("start_date")):
                ET.SubElement(event, "StartDate").text = safe_text(start)
            for end in as_list(dates.get("end_date")):
                ET.SubElement(event, "EndDate").text = safe_text(end)

        # Journal (for journal-type publications)
        for journal in as_list(metadata.get("journal")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}PublishedIn").text = safe_text(journal)

        # Publisher
        for publisher in as_list(metadata.get("publisher")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Publisher").text = safe_text(publisher)

        # Publication year
        for year in as_list(metadata.get("year")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}PublicationDate").text = safe_text(year)

        # Status (static)
        ET.SubElement(pub_el, f"{{{ns_cerif}}}Status",
                      attrib={"scheme": "/dk/atira/pure/researchoutput/status"}).text = "published"

    indent(root)
    return ET.tostring(root, encoding="utf-8").decode("utf-8")

# ----------------------------
# RIS BUILDER
# ----------------------------
def json_to_ris(metadata_list):
    lines = []

    for metadata in metadata_list:
        pub_type_key = safe_text(metadata.get("publication_type")).lower()
        ris_type = RIS_TYPE_MAP.get(pub_type_key, "GEN")

        lines.append(f"TY  - {ris_type}")

        for title in as_list(metadata.get("title")):
            lines.append(f"T1  - {safe_text(title)}")

        for author in as_list(metadata.get("authors")):
            lines.append(f"AU  - {safe_text(author.get('name'))}")

        for year in as_list(metadata.get("year")):
            lines.append(f"PY  - {safe_text(year)}")
            lines.append(f"Y1  - {safe_text(year)}")

        for abstract in as_list(metadata.get("abstract")):
            lines.append(f"N2  - {safe_text(abstract)}")
            lines.append(f"AB  - {safe_text(abstract)}")

        for kw in as_list(metadata.get("keywords")):
            lines.append(f"KW  - {safe_text(kw)}")

        for doi in as_list(metadata.get("doi")):
            doi_raw = safe_text(doi)
            if doi_raw.startswith("https://doi.org/"):
                doi_raw = doi_raw.replace("https://doi.org/", "")
            if doi_raw:
                lines.append(f"U2  - {doi_raw}")
                lines.append(f"DO  - {doi_raw}")

        lines.append(f"M3  - Conference contribution")

        conf_names = as_list(metadata.get("conference_name"))
        if conf_names:
            for conf_name in conf_names:
                lines.append(f"BT  - {safe_text(conf_name)}")
                lines.append(f"T2  - {safe_text(conf_name)}")
            dates = metadata.get("conference_dates") or {}
            start = safe_text(as_list(dates.get("start_date"))[0]) if as_list(dates.get("start_date")) else ""
            end = safe_text(as_list(dates.get("end_date"))[0]) if as_list(dates.get("end_date")) else ""
            if start or end:
                lines.append(f"Y2  - {start} through {end}")
        else:
            for journal in as_list(metadata.get("journal")):
                lines.append(f"JO  - {safe_text(journal)}")
                lines.append(f"T2  - {safe_text(journal)}")

        for publisher in as_list(metadata.get("publisher")):
            lines.append(f"PB  - {safe_text(publisher)}")

        lines.append("ER  - ")
        lines.append("")

    return "\n".join(lines)

# ----------------------------
# BATCH PROCESS
# ----------------------------
metadata_list = []
for pdf_file in sorted(PDF_DIR.glob("*.pdf")):
    base_name = pdf_file.stem
    output_dir = pdf_file.parent
    ocr_total_path = output_dir / f"{base_name}_total_{OCR_ENGINE}.txt"
    json_path = output_dir / f"{base_name}_{OCR_ENGINE}.json"

    # OCR
    if ocr_total_path.exists():
        print(f"📄 OCR text exists: {ocr_total_path.name}")
        ocr_text = ocr_total_path.read_text(encoding="utf-8")
    else:
        ocr_text = ocr_pdf(pdf_file)
        ocr_total_path.write_text(ocr_text, encoding="utf-8")
        print(f"✅ OCR done: {ocr_total_path.name}")

    # JSON
    if json_path.exists():
        print(f"📄 JSON exists: {json_path.name}")
        try:
            metadata_json = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to load JSON {json_path.name}: {e}")
            continue
    else:
        try:
            metadata_json = gpt_extract_json(ocr_text)
            json_path.write_text(json.dumps(metadata_json, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"✅ JSON saved: {json_path.name}")
        except Exception as e:
            print(f"⚠️ GPT failed for {pdf_file.name}: {e}")
            continue

    metadata_list.append(metadata_json)

# ----------------------------
# WRITE OUTPUT
# ----------------------------
if metadata_list:
    if OUTPUT_FORMAT == "xml":
        output_path = PDF_DIR / f"combined_{OCR_ENGINE}.xml"
        print("📝 Writing combined XML...")
        output_text = json_to_oai_pmh(metadata_list)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"✅ Combined XML saved: {output_path}")

    elif OUTPUT_FORMAT == "ris":
        output_path = PDF_DIR / f"combined_{OCR_ENGINE}.ris"
        print("📝 Writing combined RIS...")
        output_text = json_to_ris(metadata_list)
        output_path.write_text(output_text, encoding="utf-8")
        print(f"✅ Combined RIS saved: {output_path}")

    else:
        raise ValueError(f"Unknown OUTPUT_FORMAT: '{OUTPUT_FORMAT}'. Use 'xml' or 'ris'.")
else:
    print("⚠️ No metadata to write.")