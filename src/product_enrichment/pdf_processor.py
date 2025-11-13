"""PDF text extraction functionality."""

import logging
from pathlib import Path
from typing import List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed

import pypdf
from rich.console import Console

from .models import ProcessingConfig, ExtractedText

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


class PDFProcessor:
    """Handles PDF text extraction and processing."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize PDF processor with configuration.
        
        Args:
            config: Processing configuration including batch size
        """
        self.config = config
    
    def extract_text_from_pdf(self, pdf_path: Path) -> ExtractedText:
        """Extract text from a single PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            ExtractedText object containing the extracted content
            
        Raises:
            Exception: If PDF cannot be read or processed
        """
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract text from all pages
                text_content = ""
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():  # Only add non-empty pages
                            text_content += f"\n--- Page {page_num + 1} ---\n"
                            text_content += page_text
                    except Exception as e:
                        logger.warning(f"Failed to extract text from page {page_num + 1} of {pdf_path}: {e}")
                        continue
                
                if not text_content.strip():
                    raise ValueError(f"No text could be extracted from {pdf_path}")
                
                return ExtractedText(
                    file_path=str(pdf_path),
                    text_content=text_content.strip(),
                    page_count=len(pdf_reader.pages)
                )
                
        except Exception as e:
            error_console.print(f"[red]Error processing {pdf_path}: {e}[/red]")
            raise
    
    def extract_texts_from_pdfs(self, pdf_paths: List[Path]) -> List[ExtractedText]:
        """Extract text from multiple PDF files in parallel.
        
        Args:
            pdf_paths: List of paths to PDF files
            
        Returns:
            List of ExtractedText objects
        """
        extracted_texts = []
        
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            # Submit all PDF processing tasks
            future_to_path = {
                executor.submit(self.extract_text_from_pdf, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    extracted_text = future.result()
                    extracted_texts.append(extracted_text)
                    console.print(f"[green]✓[/green] Extracted text from {pdf_path.name}")
                except Exception as e:
                    error_console.print(f"[red]✗ Failed to process {pdf_path.name}: {e}[/red]")
                    continue
        
        if not extracted_texts:
            raise ValueError("No PDFs could be processed successfully")
        
        console.print(f"[green]Successfully extracted text from {len(extracted_texts)} PDFs[/green]")
        return extracted_texts
