"""Product feed generation using BookWyrm API and CSV output."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from bookwyrm import BookWyrmClient
from bookwyrm.models import SummaryResponse, TextResult, TextSpanResult
from bookwyrm.utils import collect_phrases_from_stream, collect_summary_from_stream
from rich.console import Console
from rich.progress import Progress

from .models import (
    ProcessingConfig, 
    ExtractedText, 
    ProductFeedItem, 
    ValidationResult,
    EnableSearchEnum,
    EnableCheckoutEnum,
    AvailabilityEnum,
    ProductExtractionModel
)

logger = logging.getLogger(__name__)
console = Console()
error_console = Console(stderr=True)


class ProductFeedGenerator:
    """Generates product feed data using BookWyrm API."""
    
    def __init__(self, config: ProcessingConfig):
        """Initialize generator with configuration.
        
        Args:
            config: Processing configuration including API key
        """
        self.config = config
        if config.api_key != "dummy":  # Allow dummy key for validation-only usage
            self.client = BookWyrmClient(api_key=config.api_key)
    
    def _process_text_to_phrases(self, text_content: str, file_path: str) -> List[TextResult]:
        """Convert raw text to phrasal format using BookWyrm API.
        
        Args:
            text_content: Raw text extracted from PDF
            file_path: Source file path for debugging
            
        Returns:
            List of phrasal text results
        """
        try:
            console.print(f"[blue]📝[/blue] Converting text to phrases for {Path(file_path).name}...")
            
            # Use BookWyrm's text processing to create phrases
            stream = self.client.stream_process_text(
                text=text_content,
                chunk_size=1000,  # Reasonable chunk size for product documents
                response_format="WITH_OFFSETS"  # Include character offsets
            )
            
            # Collect phrases using BookWyrm utility
            phrases = collect_phrases_from_stream(stream, verbose=False)
            
            console.print(f"[green]✓[/green] Created {len(phrases)} phrases from {Path(file_path).name}")
            return phrases
            
        except Exception as e:
            error_console.print(f"[red]Error processing text to phrases for {file_path}: {e}[/red]")
            raise

    def _extract_product_data_from_text(self, extracted_text: ExtractedText) -> Dict[str, Any]:
        """Extract structured product data from text using proper BookWyrm workflow.
        
        Args:
            extracted_text: Extracted text from PDF
            
        Returns:
            Dictionary containing extracted product information
        """
        try:
            console.print(f"[blue]🤖[/blue] Processing {Path(extracted_text.file_path).name} with BookWyrm API...")
            
            # Step 1: Convert raw text to phrasal format
            phrases = self._process_text_to_phrases(
                extracted_text.text_content, 
                extracted_text.file_path
            )
            
            if not phrases:
                raise ValueError("No phrases generated from text")
            
            # Step 2: Use structured summarization with Pydantic model
            console.print(f"[blue]📊[/blue] Extracting structured product data from {len(phrases)} phrases...")
            
            stream = self.client.stream_summarize(
                phrases=phrases,
                summary_class=ProductExtractionModel,
                model_strength="swift",
                debug=False
            )
            
            # Collect the structured summary
            final_result = collect_summary_from_stream(stream, verbose=False)
            
            if not final_result or not final_result.summary:
                raise ValueError("No structured summary received from BookWyrm API")
            
            # Convert the Pydantic model result to dictionary
            if isinstance(final_result.summary, ProductExtractionModel):
                product_data = final_result.summary.model_dump()
            elif isinstance(final_result.summary, dict):
                product_data = final_result.summary
            else:
                raise ValueError(f"Unexpected summary type: {type(final_result.summary)}")
            
            # Add metadata
            product_data["source_file"] = extracted_text.file_path
            product_data["page_count"] = extracted_text.page_count
            
            console.print(f"[green]✓[/green] Successfully extracted product data for {Path(extracted_text.file_path).name}")
            return product_data
            
        except Exception as e:
            error_console.print(f"[red]Error extracting product data from {extracted_text.file_path}: {e}[/red]")
            # Return minimal fallback data structure
            return {
                "id": Path(extracted_text.file_path).stem,
                "title": f"Product from {Path(extracted_text.file_path).name}",
                "description": f"Product information extracted from PDF document. Processing failed: {str(e)[:200]}",
                "source_file": extracted_text.file_path,
                "page_count": extracted_text.page_count,
                "brand": None,
                "product_category": None,
                "price": None,
                "material": None,
                "weight": None
            }
    
    def _convert_to_product_feed_item(self, product_data: Dict[str, Any]) -> ProductFeedItem:
        """Convert extracted product data to ProductFeedItem model.
        
        Args:
            product_data: Raw product data from API
            
        Returns:
            Validated ProductFeedItem instance
        """
        # Generate required fields with defaults
        product_id = product_data.get("id", "UNKNOWN")
        title = product_data.get("title", "Unknown Product")[:150]
        description = product_data.get("description", "No description available")[:5000]
        
        # Create basic product feed item
        feed_item = ProductFeedItem(
            # OpenAI Flags
            enable_search=EnableSearchEnum.TRUE,
            enable_checkout=EnableCheckoutEnum.FALSE,  # Default to false for safety
            
            # Basic Product Data
            id=product_id,
            title=title,
            description=description,
            link="https://example.com/product/" + product_id,  # Placeholder URL
            
            # Optional fields
            brand=product_data.get("brand"),
            product_category=product_data.get("product_category"),
            material=product_data.get("material"),
            weight=product_data.get("weight"),
            
            # Price (required)
            price=product_data.get("price", "0.00 USD"),
            
            # Availability (required)
            availability=AvailabilityEnum.IN_STOCK,
            inventory_quantity=1,  # Default quantity
            
            # Merchant Info (required)
            seller_name="Example Store",  # Placeholder
            seller_url="https://example.com/store",  # Placeholder
            
            # Returns (required)
            return_policy="https://example.com/returns",  # Placeholder
            return_window=30,  # Default 30 days
        )
        
        return feed_item
    
    def generate_product_feed(
        self, 
        extracted_texts: List[ExtractedText], 
        progress: Optional[Progress] = None, 
        task_id: Optional[int] = None
    ) -> List[ProductFeedItem]:
        """Generate product feed items from extracted texts.
        
        Args:
            extracted_texts: List of extracted text from PDFs
            progress: Optional Progress instance for tracking
            task_id: Optional task ID for progress updates
            
        Returns:
            List of ProductFeedItem objects
        """
        product_items = []
        
        with ThreadPoolExecutor(max_workers=self.config.batch_size) as executor:
            # Submit all extraction tasks
            future_to_text = {
                executor.submit(self._extract_product_data_from_text, text): text
                for text in extracted_texts
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_text):
                extracted_text = future_to_text[future]
                try:
                    product_data = future.result()
                    feed_item = self._convert_to_product_feed_item(product_data)
                    product_items.append(feed_item)
                    console.print(f"[green]✓[/green] Generated product data for {Path(extracted_text.file_path).name}")
                    
                    # Update progress if provided
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                        
                except Exception as e:
                    error_console.print(f"[red]✗ Failed to generate product data for {extracted_text.file_path}: {e}[/red]")
                    
                    # Still update progress for failed items
                    if progress and task_id is not None:
                        progress.update(task_id, advance=1)
                    continue
        
        if not product_items:
            raise ValueError("No product feed items could be generated")
        
        console.print(f"[green]Successfully generated {len(product_items)} product feed items[/green]")
        return product_items
    
    def save_to_csv(self, product_items: List[ProductFeedItem], output_path: Path) -> None:
        """Save product feed items to CSV file.
        
        Args:
            product_items: List of ProductFeedItem objects
            output_path: Path for output CSV file
        """
        try:
            # Convert to list of dictionaries
            data = []
            for item in product_items:
                # Convert Pydantic model to dict, handling enums and URLs
                item_dict = {}
                for field_name, field_value in item.model_dump().items():
                    if field_value is not None:
                        # Convert HttpUrl objects to strings
                        if hasattr(field_value, '__str__'):
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
