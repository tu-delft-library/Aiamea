from pdf2image import convert_from_path
from pathlib import Path
import pytesseract
import cv2
import pandas as pd
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
import xml.etree.ElementTree as ET
import xml.dom.minidom
import uuid

# ----------------------------
# LOAD OPENAI API KEY
# ----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")
client = OpenAI(api_key=api_key)

# ----------------------------
# INPUT PDF
# ----------------------------
pdf_path = Path(r"E:\XRZONE_Files\PDFExtractor\pdf-ris\samples\257.pdf")
base_name = pdf_path.stem
output_dir = pdf_path.parent
ocr_total_path = output_dir / f"{base_name}_total.txt"
json_path = output_dir / f"{base_name}.json"
xml_path = output_dir / f"{base_name}.xml"

# ----------------------------
# OCR SETTINGS
# ----------------------------
OCR_DPI = 450
TESSERACT_CONFIG = "--oem 3 --psm 12"
POPPLER_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\poppler-25.11.0\Library\bin"

# ----------------------------
# PAGE RANGE
# ----------------------------
first_page = 1
last_page = 2

# ----------------------------
# OCR OR LOAD EXISTING TEXT
# ----------------------------
if ocr_total_path.exists():
    ocr_text = ocr_total_path.read_text(encoding="utf-8")
else:
    pages = convert_from_path(str(pdf_path), OCR_DPI, poppler_path=POPPLER_PATH,
                              first_page=first_page, last_page=last_page)
    master_lines = []
    for i, page in enumerate(pages, start=first_page):
        gray = cv2.cvtColor(np.array(page), cv2.COLOR_RGB2GRAY)
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
    ocr_text = " ".join(master_lines)
    ocr_total_path.write_text(ocr_text, encoding="utf-8")

# ----------------------------
# JSON METADATA
# ----------------------------
if json_path.exists():
    metadata_json = json.loads(json_path.read_text(encoding="utf-8"))
else:
    snippet = ocr_text[:8000]
    prompt_json = f"""
Extract structured metadata from this research paper text.
Return ONLY valid JSON with the following fields:
- title
- subtitle (optional)
- authors (array of full names)
- affiliations
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

OCR text:
{snippet}
"""
    response = client.responses.create(model="gpt-4o-mini", input=prompt_json)
    raw_output = response.output_text.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    metadata_json = json.loads(raw_output)
    json_path.write_text(json.dumps(metadata_json, indent=2, ensure_ascii=False), encoding="utf-8")

# ----------------------------
# JSON → OAI-PMH CERIF XML (pretty-printed)
# ----------------------------

def json_to_oai_pmh(metadata, pdf_file=None):
    """
    Convert metadata JSON to OAI-PMH + CERIF/OpenAIRE XML
    matching the clean Pure XML format example.
    """
    import xml.etree.ElementTree as ET
    import uuid
    from xml.dom import minidom

    # Namespaces
    ns_oai = "http://www.openarchives.org/OAI/2.0/"
    ns_xsi = "http://www.w3.org/2001/XMLSchema-instance"
    ns_cerif = "https://www.openaire.eu/cerif-profile/1.2/"
    ns_pubt = "https://www.openaire.eu/cerif-profile/vocab/COAR_Publication_Types"

    # Register namespaces
    ET.register_namespace("", ns_oai)
    ET.register_namespace("cerif", ns_cerif)
    ET.register_namespace("pubt", ns_pubt)

    # Root OAI-PMH element
    root = ET.Element(
        f"{{{ns_oai}}}OAI-PMH",
        attrib={
            f"{{{ns_xsi}}}schemaLocation": (
                f"{ns_oai} {ns_oai}/OAI-PMH.xsd "
                f"{ns_cerif} https://www.openaire.eu/schema/cris/current/openaire-cerif-profile.xsd"
            )
        }
    )

    # ListRecords
    list_records = ET.SubElement(root, f"{{{ns_oai}}}ListRecords")
    record = ET.SubElement(list_records, f"{{{ns_oai}}}record")
    metadata_el = ET.SubElement(record, "metadata")

    # CERIF Publication
    pub_id = f"Publications/{str(uuid.uuid4())}"
    pub_el = ET.SubElement(metadata_el, f"{{{ns_cerif}}}Publication", id=pub_id)

    # Type
    pub_type = ET.SubElement(pub_el, f"{{{ns_pubt}}}Type")
    pub_type.text = metadata.get("publication_type", "http://purl.org/coar/resource_type/c_6501")

    # Language
    lang = ET.SubElement(pub_el, f"{{{ns_cerif}}}Language")
    lang.text = metadata.get("language", "en")

    # Titles and subtitles
    title_en = ET.SubElement(pub_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"})
    title_en.text = metadata.get("title_en", metadata.get("title", ""))
    if metadata.get("subtitle_en"):
        subtitle_en = ET.SubElement(pub_el, f"{{{ns_cerif}}}Subtitle", attrib={"xml:lang": "en"})
        subtitle_en.text = metadata.get("subtitle_en")
    if metadata.get("title_nl"):
        title_nl = ET.SubElement(pub_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "nl"})
        title_nl.text = metadata.get("title_nl")
    if metadata.get("subtitle_nl"):
        subtitle_nl = ET.SubElement(pub_el, f"{{{ns_cerif}}}Subtitle", attrib={"xml:lang": "nl"})
        subtitle_nl.text = metadata.get("subtitle_nl")

    # PublishedIn (journal)
    if metadata.get("journal"):
        published_in = ET.SubElement(pub_el, f"{{{ns_cerif}}}PublishedIn")
        journal = ET.SubElement(published_in, f"{{{ns_cerif}}}Publication", id=f"Publications/{str(uuid.uuid4())}")
        journal_type = ET.SubElement(journal, f"{{{ns_pubt}}}Type")
        journal_type.text = "http://purl.org/coar/resource_type/c_0640"
        journal_title = ET.SubElement(journal, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"})
        journal_title.text = metadata.get("journal")
        if metadata.get("issn"):
            ET.SubElement(journal, f"{{{ns_cerif}}}ISSN").text = metadata["issn"]
        if metadata.get("publisher"):
            publishers = ET.SubElement(journal, f"{{{ns_cerif}}}Publishers")
            publisher_el = ET.SubElement(publishers, f"{{{ns_cerif}}}Publisher")
            org_unit = ET.SubElement(publisher_el, f"{{{ns_cerif}}}OrgUnit")
            ET.SubElement(org_unit, f"{{{ns_cerif}}}Name", attrib={"xml:lang": "en"}).text = metadata["publisher"]

    # Publication details
    for field in ["PublicationDate", "Volume", "Issue", "StartPage", "EndPage", "DOI"]:
        if metadata.get(field.lower()):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}{field}").text = str(metadata[field.lower()])

    # # Authors
    # authors_el = ET.SubElement(pub_el, f"{{{ns_cerif}}}Authors")
    # for author in metadata.get("authors", []):
    #     author_el = ET.SubElement(authors_el, f"{{{ns_cerif}}}Author")
    #     ET.SubElement(author_el, f"{{{ns_cerif}}}DisplayName").text = author
    #     person_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Person", id=str(uuid.uuid4()))
    #     name_el = ET.SubElement(person_el, f"{{{ns_cerif}}}PersonName")
    #     if " " in author:
    #         first, last = author.split(" ", 1)
    #     else:
    #         first, last = author, ""
    #     ET.SubElement(name_el, f"{{{ns_cerif}}}FirstNames").text = first
    #     ET.SubElement(name_el, f"{{{ns_cerif}}}FamilyNames").text = last
    #     ET.SubElement(author_el, f"{{{ns_cerif}}}Affiliation")  # empty

    # Authors
    authors_el = ET.SubElement(pub_el, f"{{{ns_cerif}}}Authors")

    authors = metadata.get("authors", [])
    affiliations = metadata.get("affiliations", [])

    for i, author_name in enumerate(authors):

        author_el = ET.SubElement(authors_el, f"{{{ns_cerif}}}Author")

        # DisplayName
        ET.SubElement(author_el, f"{{{ns_cerif}}}DisplayName").text = author_name

        # Split name (best effort, since JSON gives full name only)
        if " " in author_name:
            first, last = author_name.split(" ", 1)
        else:
            first, last = author_name, ""

        # Person
        person_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Person", id=str(uuid.uuid4()))
        name_el = ET.SubElement(person_el, f"{{{ns_cerif}}}PersonName")

        ET.SubElement(name_el, f"{{{ns_cerif}}}FamilyNames").text = last
        ET.SubElement(name_el, f"{{{ns_cerif}}}FirstNames").text = first

        # Affiliation (matched by index)
        if i < len(affiliations):
            affiliation_text = affiliations[i]
            if affiliation_text:
                aff_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Affiliation")
                org_unit = ET.SubElement(aff_el, f"{{{ns_cerif}}}OrgUnit")
                ET.SubElement(
                    org_unit,
                    f"{{{ns_cerif}}}Name",
                    attrib={"xml:lang": "en"}
                ).text = affiliation_text

    # Keywords
    for kw in metadata.get("keywords", []):
        k = ET.SubElement(pub_el, f"{{{ns_cerif}}}Keyword", attrib={"xml:lang": "en"})
        k.text = kw

    # Abstract
    if metadata.get("abstract"):
        abstract = ET.SubElement(pub_el, f"{{{ns_cerif}}}Abstract", attrib={"xml:lang": "en"})
        abstract.text = metadata["abstract"]

    # PresentedAt (conference)
    if metadata.get("conference_name"):
        presented_at = ET.SubElement(pub_el, f"{{{ns_cerif}}}PresentedAt")
        event = ET.SubElement(presented_at, "Event")
        if metadata.get("conference_acronym"):
            ET.SubElement(event, "Acronym").text = metadata["conference_acronym"]
        ET.SubElement(event, "Name", attrib={"xml:lang": "en"}).text = metadata.get("conference_name")
        if metadata.get("conference_place"):
            ET.SubElement(event, "Place").text = metadata["conference_place"]
        if metadata.get("conference_country"):
            ET.SubElement(event, "Country").text = metadata["conference_country"]
        if metadata.get("conference_dates"):
            ET.SubElement(event, "StartDate").text = metadata["conference_dates"].get("start_date", "")
            ET.SubElement(event, "EndDate").text = metadata["conference_dates"].get("end_date", "")

    # Status
    ET.SubElement(pub_el, f"{{{ns_cerif}}}Status", attrib={"scheme": "/dk/atira/pure/researchoutput/status"}).text = "published"

    # Pretty-print
    xml_str = ET.tostring(root, encoding="unicode")
    xml_pretty = minidom.parseString(xml_str).toprettyxml(indent="    ")
    return xml_pretty

# ----------------------------
# SAVE XML
# ----------------------------
xml_text = json_to_oai_pmh(metadata_json)
xml_path.write_text(xml_text, encoding="utf-8")
print(f"✅ OAI-PMH / CERIF XML saved: {xml_path}")