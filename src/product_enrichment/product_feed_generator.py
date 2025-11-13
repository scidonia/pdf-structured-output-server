"""Product feed generation using BookWyrm API and CSV output."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from bookwyrm import BookWyrmClient
from bookwyrm.models import SummaryResponse, TextResult, TextSpanResult, PDFPage
from bookwyrm.utils import collect_phrases_from_stream, collect_summary_from_stream, collect_pdf_pages_from_stream, create_pdf_text_mapping_from_pages
from rich.console import Console
from rich.progress import Progress

from .models import (
    ProcessingConfig, 
    ValidationResult
)

# Import user-configurable product model
import sys
from pathlib import Path
import importlib

# Add the project root to Python path to find models directory
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


class ProductFeedGenerator:
    """Generates product feed data using BookWyrm API."""
    
    def _load_extraction_model(self, model_name: str):
        """Dynamically load the specified extraction model.
        
        Args:
            model_name: Name of the model class to load from models.models
            
        Returns:
            The loaded Pydantic model class
        """
        try:
            models_module = importlib.import_module("models.models")
            model_class = getattr(models_module, model_name)
            
            # Verify it's a Pydantic model
            if not hasattr(model_class, 'model_fields'):
                raise ValueError(f"{model_name} is not a valid Pydantic model")
            
            return model_class
            
        except ImportError:
            raise ImportError("Product extraction models not found. Please ensure models/models.py exists.")
        except AttributeError:
            raise AttributeError(f"Model '{model_name}' not found in models/models.py. Available models should be defined as Pydantic BaseModel classes.")
    
    def __init__(self, config: ProcessingConfig):
        """Initialize generator with configuration.
        
        Args:
            config: Processing configuration including API key and model name
        """
        self.config = config
        if config.api_key != "dummy":  # Allow dummy key for validation-only usage
            self.client = BookWyrmClient(api_key=config.api_key)
        
        # Dynamically import the specified model
        self.extraction_model = self._load_extraction_model(config.model_name)
    
    def _extract_pdf_with_bookwyrm(self, pdf_path: Path) -> str:
        """Extract text from PDF using BookWyrm API.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Extracted text content
        """
        try:
            # Read PDF as bytes
            pdf_bytes = pdf_path.read_bytes()
            
            # Use BookWyrm's PDF extraction
            stream = self.client.stream_extract_pdf(
                pdf_bytes=pdf_bytes,
                filename=pdf_path.name,
                start_page=1,
                num_pages=None  # Extract all pages
            )
            
            # Collect PDF pages using BookWyrm utility
            pages, metadata = collect_pdf_pages_from_stream(stream, verbose=False)
            
            if not pages:
                raise ValueError("No pages extracted from PDF")
            
            # Convert PDF pages to text mapping
            mapping = create_pdf_text_mapping_from_pages(pages)
            
            return mapping.raw_text
            
        except Exception as e:
            error_console.print(f"[red]Error extracting PDF with BookWyrm for {pdf_path}: {e}[/red]")
            raise

    def _process_text_to_phrases(self, text_content: str, file_path: str) -> List[TextResult]:
        """Convert raw text to phrasal format using BookWyrm API.
        
        Args:
            text_content: Raw text extracted from PDF
            file_path: Source file path for debugging
            
        Returns:
            List of phrasal text results
        """
        try:
            # Use BookWyrm's text processing to create phrases
            stream = self.client.stream_process_text(
                text=text_content,
                response_format="WITH_OFFSETS"  # Include character offsets
            )
            
            # Collect phrases using BookWyrm utility
            phrases = collect_phrases_from_stream(stream, verbose=False)
            
            return phrases
            
        except Exception as e:
            error_console.print(f"[red]Error processing text to phrases for {file_path}: {e}[/red]")
            raise

    def _extract_product_data_from_pdf(self, pdf_path: Path) -> Dict[str, Any]:
        """Extract structured product data from PDF using complete BookWyrm workflow.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Dictionary containing extracted product information
        """
        try:
            # Step 1: Extract text from PDF using BookWyrm
            text_content = self._extract_pdf_with_bookwyrm(pdf_path)
            
            if not text_content or not text_content.strip():
                raise ValueError("No text content extracted from PDF")
            
            # Step 2: Convert raw text to phrasal format
            phrases = self._process_text_to_phrases(text_content, str(pdf_path))
            
            if not phrases:
                raise ValueError("No phrases generated from text")
            
            # Step 3: Use structured summarization with Pydantic model
            
            stream = self.client.stream_summarize(
                phrases=phrases,
                summary_class=self.extraction_model,
                model_strength="wise",
                debug=False
            )
            
            # Collect the structured summary
            final_result = collect_summary_from_stream(stream, verbose=False)
            
            if not final_result or not final_result.summary:
                raise ValueError("No structured summary received from BookWyrm API")
            
            # Convert the result to dictionary - handle any Pydantic model generically
            if hasattr(final_result.summary, 'model_dump'):
                # It's a Pydantic model (could be ProductExtractionModel or any other)
                product_data = final_result.summary.model_dump()
            elif isinstance(final_result.summary, dict):
                product_data = final_result.summary
            elif isinstance(final_result.summary, str):
                # Handle case where API returns JSON as a string - parse it
                try:
                    # Try to parse as JSON first
                    json_data = json.loads(final_result.summary)
                    
                    # Create the user's model from the parsed JSON
                    model_instance = self.extraction_model(**json_data)
                    product_data = model_instance.model_dump()
                    
                except json.JSONDecodeError as json_error:
                    # If it's not valid JSON, treat as plain text - save debug info silently
                    debug_file = Path(f"debug_string_response_{pdf_path.stem}.txt")
                    try:
                        with open(debug_file, 'w', encoding='utf-8') as f:
                            f.write(f"=== NON-JSON STRING RESPONSE FROM BOOKWYRM API ===\n")
                            f.write(f"File: {pdf_path.name}\n")
                            f.write(f"JSON Parse Error: {json_error}\n")
                            f.write(f"Response length: {len(final_result.summary)} characters\n\n")
                            f.write(final_result.summary)
                    except Exception:
                        pass  # Silently ignore debug file creation errors
                    
                    # Create minimal fallback using model fields if available
                    fallback_data = {
                        "id": pdf_path.stem,
                        "title": f"Product from {pdf_path.name}",
                        "description": final_result.summary[:1000] if final_result.summary else "No description available"
                    }
                    # Add None values for any other fields defined in the model
                    try:
                        model_fields = self.extraction_model.model_fields
                        for field_name in model_fields:
                            if field_name not in fallback_data:
                                fallback_data[field_name] = None
                    except Exception:
                        # If we can't inspect the model, just use basic fallback
                        pass
                    product_data = fallback_data
                    
                except Exception as model_error:
                    # If model creation fails, use the JSON data directly (log silently)
                    product_data = json_data
            else:
                raise ValueError(f"Unexpected summary type: {type(final_result.summary)}")
            
            # Add metadata
            product_data["source_file"] = str(pdf_path)
            product_data["page_count"] = len(text_content.split('\n'))  # Rough estimate
            
            return product_data
            
        except Exception as e:
            error_console.print(f"[red]Error extracting product data from {pdf_path}: {e}[/red]")
            # Return minimal fallback data structure
            return {
                "id": pdf_path.stem,
                "title": f"Product from {pdf_path.name}",
                "description": f"Product information extracted from PDF document. Processing failed: {str(e)[:200]}",
                "source_file": str(pdf_path),
                "page_count": 0,
                "brand": None,
                "product_category": None,
                "price": None,
                "material": None,
                "weight": None
            }
    
    def _extract_with_schema(self, phrases: List[TextResult], schema_name: str, schema_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data using JSON schema directly (for server API).
        
        Args:
            phrases: List of phrasal text results
            schema_name: Name for the extraction schema
            schema_dict: JSON schema definition
            
        Returns:
            Extracted data as dictionary
        """
        try:
            # Use BookWyrm's summarization with JSON schema directly
            stream = self.client.stream_summarize(
                phrases=phrases,
                model_name=schema_name,
                model_schema_json=json.dumps(schema_dict),
                model_strength="wise",
                debug=False
            )
            
            # Collect the structured summary
            final_result = collect_summary_from_stream(stream, verbose=False)
            
            if not final_result or not final_result.summary:
                raise ValueError("No structured summary received from BookWyrm API")
            
            # Convert result to dictionary
            if hasattr(final_result.summary, 'model_dump'):
                return final_result.summary.model_dump()
            elif isinstance(final_result.summary, dict):
                return final_result.summary
            elif isinstance(final_result.summary, str):
                # Try to parse as JSON
                try:
                    return json.loads(final_result.summary)
                except json.JSONDecodeError:
                    return {"raw_response": final_result.summary}
            else:
                return {"raw_response": str(final_result.summary)}
                
        except Exception as e:
            logger.error(f"Error in schema-based extraction: {e}")
            raise
    
    
    def _convert_to_product_feed_item(self, product_data: Dict[str, Any]) -> Dict[str, Any]:
        """Convert extracted product data to dictionary for CSV output.
        
        Args:
            product_data: Raw product data from API (structure matches user's Pydantic model)
            
        Returns:
            Dictionary containing the complete product data
        """
        # The API returns data that matches the user's Pydantic model JSON schema
        # We have a complete object already, so we just return it as-is
        return product_data
    
    def generate_product_feed(
        self, 
        pdf_paths: List[Path], 
        progress: Optional[Progress] = None, 
        task_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """Generate product feed items from PDF files using complete BookWyrm workflow.
        
        Args:
            pdf_paths: List of PDF file paths to process
            progress: Optional Progress instance for tracking
            task_id: Optional task ID for progress updates
            
        Returns:
            List of product data dictionaries
        """
        product_items = []
        
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            # Submit all extraction tasks
            future_to_path = {
                executor.submit(self._extract_product_data_from_pdf, pdf_path): pdf_path
                for pdf_path in pdf_paths
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_path):
                pdf_path = future_to_path[future]
                try:
                    product_data = future.result()
                    feed_item = self._convert_to_product_feed_item(product_data)
                    product_items.append(feed_item)
                    
                    # Update progress if provided
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                        
                except Exception as e:
                    error_console.print(f"[red]✗ Failed to generate product data for {pdf_path}: {e}[/red]")
                    
                    # Still update progress for failed items
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                    continue
        
        if not product_items:
            raise ValueError("No product feed items could be generated")
        
        console.print(f"[green]Successfully generated {len(product_items)} product feed items[/green]")
        return product_items
    
    def save_to_csv(self, product_items: List[Dict[str, Any]], output_path: Path) -> None:
        """Save product feed items to CSV file using reflection.
        
        Args:
            product_items: List of product data dictionaries
            output_path: Path for output CSV file
        """
        try:
            # Use reflection to get all possible fields from the user's extraction model
            model_fields = {}
            try:
                model_fields = self.extraction_model.model_fields
            except Exception:
                # If we can't get model fields, we'll just use the data as-is
                pass
            
            # Convert to list of dictionaries using reflection
            data = []
            for item in product_items:
                # Use reflection to iterate over attributes and their types
                item_dict = {}
                for field_name, field_value in item.items():
                    if field_value is not None:
                        # Get field type from model if available
                        field_type = None
                        if model_fields and field_name in model_fields:
                            field_type = model_fields[field_name].annotation
                        
                        # Convert based on type or use string representation
                        if isinstance(field_value, (list, dict)):
                            # Convert complex types to JSON strings
                            item_dict[field_name] = json.dumps(field_value)
                        elif hasattr(field_value, '__str__'):
                            item_dict[field_name] = str(field_value)
                        else:
                            item_dict[field_name] = field_value
                    else:
                        item_dict[field_name] = ""
                data.append(item_dict)
            
            # Create DataFrame and save to CSV
            df = pd.DataFrame(data)
            df.to_csv(output_path, index=False)
            
            console.print(f"[green]✓ Saved {len(product_items)} products to {output_path}[/green]")
            
        except Exception as e:
            error_console.print(f"[red]Error saving CSV: {e}[/red]")
            raise
    
    def validate_csv(self, csv_path: Path) -> ValidationResult:
        """Validate a CSV file against the product feed specification.
        
        Args:
            csv_path: Path to CSV file to validate
            
        Returns:
            ValidationResult with validation status and errors
        """
        errors = []
        warnings = []
        total_rows = 0
        
        try:
            console.print(f"[blue]📋[/blue] Loading CSV file: {csv_path}")
            df = pd.read_csv(csv_path)
            total_rows = len(df)
            
            console.print(f"[blue]🔍[/blue] Validating {total_rows} rows...")
            
            # Check required columns
            required_fields = [
                'enable_search', 'enable_checkout', 'id', 'title', 'description', 
                'link', 'price', 'availability', 'inventory_quantity', 
                'seller_name', 'seller_url', 'return_policy', 'return_window'
            ]
            
            missing_fields = [field for field in required_fields if field not in df.columns]
            if missing_fields:
                errors.append(f"Missing required fields: {', '.join(missing_fields)}")
            
            # Validate each row with progress
            from rich.progress import Progress, BarColumn, TaskProgressColumn, TimeElapsedColumn, TextColumn
            
            with Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                validation_task = progress.add_task("Validating rows...", total=total_rows)
                
                for idx, row in df.iterrows():
                    row_errors = []
                    
                    # Check required field values
                    if pd.isna(row.get('id')) or str(row.get('id')).strip() == '':
                        row_errors.append(f"Row {idx + 1}: 'id' is required")
                    
                    if pd.isna(row.get('title')) or str(row.get('title')).strip() == '':
                        row_errors.append(f"Row {idx + 1}: 'title' is required")
                    
                    # Check field length constraints
                    if not pd.isna(row.get('title')) and len(str(row.get('title'))) > 150:
                        row_errors.append(f"Row {idx + 1}: 'title' exceeds 150 characters")
                    
                    if not pd.isna(row.get('description')) and len(str(row.get('description'))) > 5000:
                        row_errors.append(f"Row {idx + 1}: 'description' exceeds 5000 characters")
                    
                    errors.extend(row_errors)
                    progress.update(validation_task, advance=1)
            
            is_valid = len(errors) == 0
            
            return ValidationResult(
                is_valid=is_valid,
                total_rows=total_rows,
                errors=errors,
                warnings=warnings
            )
            
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                total_rows=total_rows,
                errors=[f"Error reading CSV file: {e}"],
                warnings=warnings
            )
