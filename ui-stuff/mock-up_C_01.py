import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading, queue, time

# ===== CORE SETTINGS MODEL =====
class Settings:
    def __init__(self):
        self.pdf_dir = ''
        self.ocr_engine = 'pymupdf'
        self.output_format = 'ris'

# ===== BACKEND PLACEHOLDERS (replace with your real functions) =====
def run_batch(settings, log, progress):
    folder = Path(settings.pdf_dir)
    files = list(folder.glob('*.pdf')) if folder.exists() else []
    total = len(files)
    if not total:
        log.put('No PDF files found.')
        return
    for i, pdf in enumerate(files, start=1):
        log.put(f'Processing {pdf.name} using {settings.ocr_engine}...')
        time.sleep(0.5)
        log.put(f'Exported metadata to {settings.output_format.upper()}')
        progress.put((i, total))
    log.put('Done.')

# ===== GUI =====
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('Aiamea')
        self.geometry('760x620')
        self.settings = Settings()
        self.log_q = queue.Queue()
        self.progress_q = queue.Queue()
        self.build_ui()
        self.after(100, self.poll_queues)

    def build_ui(self):
        top = ttk.Frame(self, padding=10)
        top.pack(fill='x')

        ttk.Label(top, text='PDF Folder').grid(row=0, column=0, sticky='w')
        self.folder_var = tk.StringVar()
        ttk.Entry(top, textvariable=self.folder_var, width=70).grid(row=1, column=0, sticky='ew', padx=(0,8))
        ttk.Button(top, text='Browse', command=self.pick_folder).grid(row=1, column=1)
        top.columnconfigure(0, weight=1)

        opts = ttk.LabelFrame(self, text='Options', padding=10)
        opts.pack(fill='x', padx=10, pady=8)

        self.ocr_var = tk.StringVar(value='pymupdf')
        ttk.Label(opts, text='OCR Engine').grid(row=0, column=0, sticky='w')
        ttk.Radiobutton(opts, text='PyMuPDF', variable=self.ocr_var, value='pymupdf').grid(row=1,column=0,sticky='w')
        ttk.Radiobutton(opts, text='Tesseract', variable=self.ocr_var, value='tesseract').grid(row=2,column=0,sticky='w')

        self.out_var = tk.StringVar(value='ris')
        ttk.Label(opts, text='Output').grid(row=0, column=1, sticky='w', padx=(30,0))
        ttk.Radiobutton(opts, text='RIS', variable=self.out_var, value='ris').grid(row=1,column=1,sticky='w', padx=(30,0))
        ttk.Radiobutton(opts, text='XML', variable=self.out_var, value='xml').grid(row=2,column=1,sticky='w', padx=(30,0))

        actions = ttk.Frame(self, padding=10)
        actions.pack(fill='x')
        ttk.Button(actions, text='Extract Metadata', command=self.start_run).pack(side='left')
        self.pb = ttk.Progressbar(actions, mode='determinate')
        self.pb.pack(side='left', fill='x', expand=True, padx=10)

        logf = ttk.LabelFrame(self, text='Log', padding=10)
        logf.pack(fill='both', expand=True, padx=10, pady=(0,10))
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
        self.settings.pdf_dir = self.folder_var.get()
        self.settings.ocr_engine = self.ocr_var.get()
        self.settings.output_format = self.out_var.get()
        self.pb['value'] = 0
        threading.Thread(target=run_batch, args=(self.settings, self.log_q, self.progress_q), daemon=True).start()

    def poll_queues(self):
        while not self.log_q.empty():
            msg = self.log_q.get_nowait()
            self.txt.insert('end', msg + '\n')
            self.txt.see('end')
        while not self.progress_q.empty():
            cur, total = self.progress_q.get_nowait()
            self.pb['maximum'] = total
            self.pb['value'] = cur
        self.after(100, self.poll_queues)

if __name__ == '__main__':
    App().mainloop()
