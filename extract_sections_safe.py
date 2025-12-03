"""
Extract Future Works and Limitations sections from PDF and save to file
"""

import PyPDF2
import sys

def extract_sections(pdf_path, output_path):
    """Extract Future Works and Limitations sections"""
    
    with open(pdf_path, 'rb') as f:
        pdf = PyPDF2.PdfReader(f)
        
        full_text = ""
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            full_text += f"\n--- Page {i+1} ---\n{text}\n"
    
    # Save full text to file
    with open(output_path, 'w', encoding='utf-8', errors='replace') as f:
        # Search for Future Works
        f.write("="*80 + "\n")
        f.write("FUTURE WORKS SECTION:\n")
        f.write("="*80 + "\n\n")
        
        future_keywords = ['future work', 'future works', 'future directions', 'future research']
        found_future = False
        for keyword in future_keywords:
            idx = full_text.lower().find(keyword.lower())
            if idx != -1:
                f.write(f"Found '{keyword}' at position {idx}\n\n")
                # Extract section (assume it goes until next section or end)
                start = max(0, idx - 200)
                end = min(len(full_text), idx + 2000)
                section = full_text[start:end]
                
                # Try to find end of section (next section header or end)
                end_markers = ['\n6 ', '\n7 ', 'limitation', 'conclusion', 'acknowledgment', 'reference']
                for marker in end_markers:
                    marker_idx = section.lower().find(marker, 500)  # Start searching after first 500 chars
                    if marker_idx != -1:
                        section = section[:marker_idx]
                        break
                
                f.write(section)
                f.write("\n\n" + "="*80 + "\n\n")
                found_future = True
                break
        
        if not found_future:
            f.write("FUTURE WORKS SECTION NOT FOUND\n\n")
        
        # Search for Limitations
        f.write("="*80 + "\n")
        f.write("LIMITATIONS SECTION:\n")
        f.write("="*80 + "\n\n")
        
        limitation_keywords = ['limitation', 'limitations', 'constraint', 'constraints']
        found_limitations = False
        for keyword in limitation_keywords:
            idx = full_text.lower().find(keyword.lower())
            if idx != -1:
                f.write(f"Found '{keyword}' at position {idx}\n\n")
                # Extract section
                start = max(0, idx - 200)
                end = min(len(full_text), idx + 2000)
                section = full_text[start:end]
                
                # Try to find end of section
                end_markers = ['\n6 ', '\n7 ', 'future', 'conclusion', 'acknowledgment', 'reference']
                for marker in end_markers:
                    marker_idx = section.lower().find(marker, 500)
                    if marker_idx != -1:
                        section = section[:marker_idx]
                        break
                
                f.write(section)
                f.write("\n\n" + "="*80 + "\n\n")
                found_limitations = True
                break
        
        if not found_limitations:
            f.write("LIMITATIONS SECTION NOT FOUND\n\n")
    
    print(f"Extracted sections saved to: {output_path}")

if __name__ == '__main__':
    pdf_path = r"C:\Users\PC\Downloads\Dynamic_Emotion_Project (4).pdf"
    output_path = r"C:\Users\PC\Downloads\Github Folders 2\future_limitations_extracted.txt"
    
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    extract_sections(pdf_path, output_path)




