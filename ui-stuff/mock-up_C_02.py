from pathlib import Path
import pandas as pd
import json
import xml.etree.ElementTree as ET
import uuid
import os
import threading
import queue
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# OPENAI CLIENT SETUP
# ----------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Check your .env file.")
client = OpenAI(api_key=api_key)

# ----------------------------
# CONSTANTS
# ----------------------------
FIRST_PAGE = 1
LAST_PAGE = 2

OCR_DPI = 450
TESSERACT_CONFIG = "--oem 3 --psm 12"
POPPLER_PATH = r"E:\XRZONE_Files\PDFExtractor\pdf-ris\poppler-25.11.0\Library\bin"

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
        return [a.strip() for a in safe_text(raw).split(";") if a.strip()]

# ----------------------------
# OCR FUNCTIONS
# ----------------------------
def ocr_with_tesseract(pdf_path):
    from pdf2image import convert_from_path
    import pytesseract
    import cv2
    import numpy as np

    pages = convert_from_path(
        str(pdf_path), OCR_DPI,
        poppler_path=POPPLER_PATH,
        first_page=FIRST_PAGE,
        last_page=LAST_PAGE
    )
    master_lines = []
    for i, page in enumerate(pages, start=FIRST_PAGE):
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
    return " ".join(master_lines)

def ocr_with_pymupdf(pdf_path):
    import fitz

    doc = fitz.open(str(pdf_path))
    all_text = []
    start = FIRST_PAGE - 1
    end = min(LAST_PAGE, len(doc))
    for page_num in range(start, end):
        page = doc[page_num]
        text = page.get_text("text")
        all_text.append(text)
    doc.close()
    return " ".join(all_text)

def ocr_pdf(pdf_path, ocr_engine):
    if ocr_engine == "pymupdf":
        return ocr_with_pymupdf(pdf_path)
    elif ocr_engine == "tesseract":
        return ocr_with_tesseract(pdf_path)
    else:
        raise ValueError(f"Unknown OCR engine: '{ocr_engine}'")

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

General rules:

- Return only valid JSON
- No explanations
- No markdown
- No comments
- Missing fields should be null

OCR text:
{snippet}
"""
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt_json}]
    )
    raw_output = response.choices[0].message.content.strip()
    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`").replace("json", "", 1).strip()
    return json.loads(raw_output)

# ----------------------------
# XML BUILDER
# ----------------------------
def json_to_oai_pmh(metadata_list):
    ns_oai  = "http://www.openarchives.org/OAI/2.0/"
    ns_xsi  = "http://www.w3.org/2001/XMLSchema-instance"
    ns_cerif = "https://www.openaire.eu/cerif-profile/1.2/"
    ns_pubt = "https://www.openaire.eu/cerif-profile/vocab/COAR_Publication_Types"
    ns_ar   = "http://purl.org/coar/access_right"

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

        for pub_type in as_list(metadata.get("publication_type")):
            pub_type_uri = COAR_TYPE_MAP.get(safe_text(pub_type).lower(), COAR_TYPE_MAP["conference"])
            ET.SubElement(pub_el, f"{{{ns_pubt}}}Type").text = pub_type_uri

        ET.SubElement(pub_el, f"{{{ns_cerif}}}Language").text = "en"

        for title in as_list(metadata.get("title")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Title", attrib={"xml:lang": "en"}).text = safe_text(title)

        for subtitle in as_list(metadata.get("subtitle")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Subtitle", attrib={"xml:lang": "en"}).text = safe_text(subtitle)

        for abstract in as_list(metadata.get("abstract")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Abstract", attrib={"xml:lang": "en"}).text = safe_text(abstract)

        for kw in as_list(metadata.get("keywords")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Keyword", attrib={"xml:lang": "en"}).text = safe_text(kw)

        for doi in as_list(metadata.get("doi")):
            doi_raw = safe_text(doi).replace("https://doi.org/", "")
            ET.SubElement(pub_el, f"{{{ns_cerif}}}DOI").text = doi_raw

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
            for aff in affiliations:
                aff_el = ET.SubElement(author_el, f"{{{ns_cerif}}}Affiliation")
                org_el = ET.SubElement(aff_el, f"{{{ns_cerif}}}OrgUnit")
                ET.SubElement(org_el, f"{{{ns_cerif}}}Name", attrib={"xml:lang": "en"}).text = aff

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

        for journal in as_list(metadata.get("journal")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}PublishedIn").text = safe_text(journal)

        for publisher in as_list(metadata.get("publisher")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}Publisher").text = safe_text(publisher)

        for year in as_list(metadata.get("year")):
            ET.SubElement(pub_el, f"{{{ns_cerif}}}PublicationDate").text = safe_text(year)

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
            doi_raw = safe_text(doi).replace("https://doi.org/", "")
            if doi_raw:
                lines.append(f"U2  - {doi_raw}")
                lines.append(f"DO  - {doi_raw}")

        lines.append("M3  - Conference contribution")

        conf_names = as_list(metadata.get("conference_name"))
        if conf_names:
            for conf_name in conf_names:
                lines.append(f"BT  - {safe_text(conf_name)}")
                lines.append(f"T2  - {safe_text(conf_name)}")
            dates = metadata.get("conference_dates") or {}
            start = safe_text(as_list(dates.get("start_date"))[0]) if as_list(dates.get("start_date")) else ""
            end   = safe_text(as_list(dates.get("end_date"))[0])   if as_list(dates.get("end_date"))   else ""
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
# BATCH RUNNER (runs in background thread)
# ----------------------------
def run_batch(settings, log_q, progress_q):
    pdf_dir       = Path(settings.pdf_dir)
    ocr_engine    = settings.ocr_engine
    output_format = settings.output_format

    files = sorted(pdf_dir.glob("*.pdf"))
    total = len(files)
    if not total:
        log_q.put("No PDF files found.")
        return

    metadata_list = []

    for i, pdf_file in enumerate(files, start=1):
        base_name      = pdf_file.stem
        output_dir     = pdf_file.parent
        ocr_cache_path = output_dir / f"{base_name}_total_{ocr_engine}.txt"
        json_path      = output_dir / f"{base_name}_{ocr_engine}.json"

        # --- OCR ---
        if ocr_cache_path.exists():
            log_q.put(f"OCR cache found: {ocr_cache_path.name}")
            ocr_text = ocr_cache_path.read_text(encoding="utf-8")
        else:
            log_q.put(f"OCRing {pdf_file.name} with {ocr_engine}...")
            try:
                ocr_text = ocr_pdf(pdf_file, ocr_engine)
                ocr_cache_path.write_text(ocr_text, encoding="utf-8")
                log_q.put(f"OCR done: {ocr_cache_path.name}")
            except Exception as e:
                log_q.put(f"ERROR (OCR) {pdf_file.name}: {e}")
                progress_q.put((i, total))
                continue

        # --- GPT extraction ---
        if json_path.exists():
            log_q.put(f"JSON cache found: {json_path.name}")
            try:
                metadata_json = json.loads(json_path.read_text(encoding="utf-8"))
            except Exception as e:
                log_q.put(f"ERROR (JSON load) {json_path.name}: {e}")
                progress_q.put((i, total))
                continue
        else:
            log_q.put(f"Extracting metadata via GPT: {pdf_file.name}...")
            try:
                metadata_json = gpt_extract_json(ocr_text)
                json_path.write_text(json.dumps(metadata_json, indent=2, ensure_ascii=False), encoding="utf-8")
                log_q.put(f"JSON saved: {json_path.name}")
            except Exception as e:
                log_q.put(f"ERROR (GPT) {pdf_file.name}: {e}")
                progress_q.put((i, total))
                continue

        metadata_list.append(metadata_json)
        progress_q.put((i, total))

    # --- Write output ---
    if not metadata_list:
        log_q.put("No metadata collected — nothing to write.")
        return

    if output_format == "xml":
        output_path = pdf_dir / f"combined_{ocr_engine}.xml"
        log_q.put("Writing combined XML...")
        try:
            output_text = json_to_oai_pmh(metadata_list)
            output_path.write_text(output_text, encoding="utf-8")
            log_q.put(f"Done. XML saved: {output_path}")
        except Exception as e:
            log_q.put(f"ERROR (XML write): {e}")

    elif output_format == "ris":
        output_path = pdf_dir / f"combined_{ocr_engine}.ris"
        log_q.put("Writing combined RIS...")
        try:
            output_text = json_to_ris(metadata_list)
            output_path.write_text(output_text, encoding="utf-8")
            log_q.put(f"Done. RIS saved: {output_path}")
        except Exception as e:
            log_q.put(f"ERROR (RIS write): {e}")

# ----------------------------
# SETTINGS MODEL
# ----------------------------
class Settings:
    def __init__(self):
        self.pdf_dir       = ''
        self.ocr_engine    = 'pymupdf'
        self.output_format = 'ris'

# ----------------------------
# GUI
# ----------------------------
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Aiamea')
        self.geometry('760x620')
        self.settings  = Settings()
        self.log_q      = queue.Queue()
        self.progress_q = queue.Queue()
        self.build_ui()
        self.after(100, self.poll_queues)

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')

        ttk.Label(top, text='PDF Folder').grid(row=0, column=0, sticky='w')
        self.folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.folder_var, width=70).grid(row=1, column=0, sticky='ew', padx=(0, 8))
        ttk.Button(top, text='Browse', command=self.pick_folder).grid(row=1, column=1)
        top.columnconfigure(0, weight=1)

        opts = ttk.LabelFrame(self, text='Options', padding=10)
        opts.pack(fill='x', padx=10, pady=8)

        self.ocr_var = tk.StringVar(value='pymupdf')
        ttk.Label(opts, text='OCR Engine').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(opts, text='PyMuPDF',  variable=self.ocr_var, value='pymupdf').grid(row=1, column=0, sticky='w')
        ttk.Radiobutton(opts, text='Tesseract', variable=self.ocr_var, value='tesseract').grid(row=2, column=0, sticky='w')

        self.out_var = tk.StringVar(value='ris')
        ttk.Label(opts, text='Output').grid(row=0, column=1, sticky='w', padx=(30, 0))
        ttk.Radiobutton(opts, text='RIS', variable=self.out_var, value='ris').grid(row=1, column=1, sticky='w', padx=(30, 0))
        ttk.Radiobutton(opts, text='XML', variable=self.out_var, value='xml').grid(row=2, column=1, sticky='w', padx=(30, 0))

        actions = ttk.Frame(self, padding=10)
        actions.pack(fill='x')
        self.run_btn = ttk.Button(actions, text='Extract Metadata', command=self.start_run)
        self.run_btn.pack(side='left')
        self.pb = ttk.Progressbar(actions, mode='determinate')
        self.pb.pack(side='left', fill='x', expand=True, padx=10)

        logf = ttk.LabelFrame(self, text='Log', padding=10)
        logf.pack(fill='both', expand=True, padx=10, pady=(0, 10))
        self.txt = tk.Text(logf, height=20)
        self.txt.pack(fill='both', expand=True)

    def pick_folder(self):
        p = filedialog.askdirectory()
        if p:
            self.folder_var.set(p)

    def start_run(self):
        if not self.folder_var.get():
            messagebox.showerror('Missing folder', 'Select a PDF folder first.')
            return
        self.settings.pdf_dir       = self.folder_var.get()
        self.settings.ocr_engine    = self.ocr_var.get()
        self.settings.output_format = self.out_var.get()
        self.pb['value'] = 0
        self.txt.delete('1.0', 'end')
        self.run_btn.config(state='disabled')
        threading.Thread(
            target=self._run_and_reenable,
            daemon=True
        ).start()

    def _run_and_reenable(self):
        """Wrapper so the button re-enables after the batch finishes."""
        try:
            run_batch(self.settings, self.log_q, self.progress_q)
        finally:
            self.after(0, lambda: self.run_btn.config(state='normal'))

    def poll_queues(self):
        while not self.log_q.empty():
            msg = self.log_q.get_nowait()
            self.txt.insert('end', msg + '\n')
            self.txt.see('end')
        while not self.progress_q.empty():
            cur, total = self.progress_q.get_nowait()
            self.pb['maximum'] = total
            self.pb['value']   = cur
        self.after(100, self.poll_queues)

if __name__ == '__main__':
    App().mainloop()