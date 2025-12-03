"""
Extract Future Works and Limitations sections from PDF
"""

import PyPDF2
import sys
import io

def extract_sections(pdf_path):
    """Extract Future Works and Limitations sections"""
    
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        
        print(f"Total pages: {len(pdf.pages)}\n")
        print("="*80)
        
        full_text = ""
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            full_text += f"\n--- Page {i+1} ---\n{text}\n"
        
        # Search for Future Works
        print("SEARCHING FOR FUTURE WORKS:")
        print("="*80)
        
        future_keywords = ['future work', 'future works', 'future directions', 'future research']
        for keyword in future_keywords:
            idx = full_text.lower().find(keyword.lower())
            if idx != -1:
                print(f"\nFound '{keyword}' at position {idx}")
                start = max(0, idx - 100)
                end = min(len(full_text), idx + 1500)
                print(full_text[start:end])
                print("\n" + "="*80)
                break
        
        # Search for Limitations
        print("\nSEARCHING FOR LIMITATIONS:")
        print("="*80)
        
        limitation_keywords = ['limitation', 'limitations', 'constraint', 'constraints', 'shortcoming']
        for keyword in limitation_keywords:
            idx = full_text.lower().find(keyword.lower())
            if idx != -1:
                print(f"\nFound '{keyword}' at position {idx}")
                start = max(0, idx - 100)
                end = min(len(full_text), idx + 1500)
                print(full_text[start:end])
                print("\n" + "="*80)
                break
        
        # Print last few pages in case sections are there
        print("\nLAST 2 PAGES (in case sections are there):")
        print("="*80)
        for i in range(max(0, len(pdf.pages)-2), len(pdf.pages)):
            print(f"\n--- Page {i+1} ---")
            print(pdf.pages[i].extract_text())

if __name__ == '__main__':
    pdf_path = r"C:\Users\PC\Downloads\Dynamic_Emotion_Project (4).pdf"
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    
    extract_sections(pdf_path)

