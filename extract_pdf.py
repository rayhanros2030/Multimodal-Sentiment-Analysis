import sys
import os
import io

# Set UTF-8 encoding for stdout
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

pdf_path = r"C:\Users\PC\Downloads\Dynamic_Emotion_Project (3).pdf"

# Try PyPDF2 first
try:
    import PyPDF2
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        # Save to file first
        output_file = r"C:\Users\PC\Downloads\Github Folders 2\pdf_text_extracted.txt"
        with open(output_file, 'w', encoding='utf-8', errors='replace') as out:
            out.write(text)
        print(f"Extracted {len(text)} characters. First 10000 chars:")
        print(text[:10000])
        sys.exit(0)
except ImportError:
    pass
except Exception as e:
    print(f"PyPDF2 error: {e}")

# Try pdfplumber
try:
    import pdfplumber
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        output_file = r"C:\Users\PC\Downloads\Github Folders 2\pdf_text_extracted.txt"
        with open(output_file, 'w', encoding='utf-8', errors='replace') as out:
            out.write(text)
        print(f"Extracted {len(text)} characters. First 10000 chars:")
        print(text[:10000])
        sys.exit(0)
except ImportError:
    pass
except Exception as e:
    print(f"pdfplumber error: {e}")

# Fallback: try to install PyPDF2
try:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2", "-q"])
    import PyPDF2
    with open(pdf_path, 'rb') as f:
        reader = PyPDF2.PdfReader(f)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n\n"
        output_file = r"C:\Users\PC\Downloads\Github Folders 2\pdf_text_extracted.txt"
        with open(output_file, 'w', encoding='utf-8', errors='replace') as out:
            out.write(text)
        print(f"Extracted {len(text)} characters. First 10000 chars:")
        print(text[:10000])
        sys.exit(0)
except Exception as e:
    print(f"Could not extract PDF: {e}")
    sys.exit(1)

