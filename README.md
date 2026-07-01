# Aiamea


With Aiamea the TU Delft Library aims to develop an Artificial Intelligence Assisted Meta-data Extraction Application.

For this purpose Aiamea communicates with a Large Language Model trough API.

It utilizes OCR (Tesseract) and LLM text recognition abilities to extract metadata from .pdf scientific research papers in bulk, and writes these to .json file. 

The collected metadata can then be exported to a range of standard metadata file formats, allowing the metadata to be imported in bulk into a Current Research Information System (CRIS). Since LLMs can also generate text based on a provided source, Aiamea can also output keywords even if non were explicitly defined in the research paper. 



Currently supported export formats: 

\- OpenAIRE Cerif XML

\- RIS

Export formats planned for the future:

\- BibTex



Aiamea is developed by TU Delft Library employees H.D Nguyen and J. Stelma.

Aiamea is shared under a Creative Commons Attribution-NonCommercial-ShareAlike license (CC-BY-NC-SA).

(Tesseract is distributed under the Apache License)

