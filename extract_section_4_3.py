"""
Extract section 4.3 from the PDF and verify accuracy.
"""

import PyPDF2
import pdfplumber
import sys
import io

pdf_path = r"C:\Users\PC\Downloads\Dynamic_Emotion_Project (7).pdf"

print("Extracting section 4.3 from PDF...")
print("="*80)

try:
    # Try with pdfplumber first (better text extraction)
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"
        
        # Find section 4.3
        lines = full_text.split('\n')
        in_section_4_3 = False
        section_4_3_text = []
        section_4_3_found = False
        
        for i, line in enumerate(lines):
            # Look for section 4.3 header
            if '4.3' in line and ('Dataset' in line or 'Data' in line or 'Experiments' in line):
                in_section_4_3 = True
                section_4_3_found = True
                section_4_3_text.append(line)
                print(f"Found section 4.3 header: {line}")
                continue
            
            # If we're in section 4.3, collect lines
            if in_section_4_3:
                # Stop at next major section (4.4, 5.0, etc.)
                if line.strip() and (line.strip().startswith('4.4') or 
                                     line.strip().startswith('5.') or 
                                     line.strip().startswith('4.4') or
                                     (line.strip().startswith('4.') and '4.3' not in line and len(line.strip()) < 10)):
                    break
                
                section_4_3_text.append(line)
        
        if section_4_3_found:
            print("\n" + "="*80)
            print("SECTION 4.3 CONTENT:")
            print("="*80)
            print('\n'.join(section_4_3_text))
            print("="*80)
            
            # Save to file
            output_path = r"C:\Users\PC\Downloads\Github Folders 2\section_4_3_extracted.txt"
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write('\n'.join(section_4_3_text))
            print(f"\nSection 4.3 saved to: {output_path}")
        else:
            print("Section 4.3 not found. Searching for '4.3' or 'Dataset'...")
            # Search for any mention of 4.3
            for i, line in enumerate(lines):
                if '4.3' in line:
                    print(f"Line {i}: {line}")
                    # Print context
                    for j in range(max(0, i-2), min(len(lines), i+10)):
                        print(f"  {j}: {lines[j]}")
                    break

except Exception as e:
    print(f"Error with pdfplumber: {e}")
    print("Trying PyPDF2...")
    
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            full_text = ""
            for page in pdf_reader.pages:
                full_text += page.extract_text() + "\n"
            
            # Find section 4.3
            lines = full_text.split('\n')
            in_section_4_3 = False
            section_4_3_text = []
            
            for i, line in enumerate(lines):
                if '4.3' in line and ('Dataset' in line or 'Data' in line):
                    in_section_4_3 = True
                    section_4_3_text.append(line)
                    continue
                
                if in_section_4_3:
                    if line.strip().startswith('4.4') or line.strip().startswith('5.'):
                        break
                    section_4_3_text.append(line)
            
            if section_4_3_text:
                print("\n" + "="*80)
                print("SECTION 4.3 CONTENT:")
                print("="*80)
                print('\n'.join(section_4_3_text))
                print("="*80)
    except Exception as e2:
        print(f"Error with PyPDF2: {e2}")




