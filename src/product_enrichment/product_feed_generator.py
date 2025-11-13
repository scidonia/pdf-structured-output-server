"""Product feed generation using BookWyrm API and CSV output."""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from bookwyrm import BookWyrmClient
from bookwyrm.models import SummaryResponse
from rich.console import Console
from rich.progress import Progress

from .models import (
    ProcessingConfig, 
    ExtractedText, 
    ProductFeedItem, 
    ValidationResult,
    EnableSearchEnum,
    EnableCheckoutEnum,
    AvailabilityEnum
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
    
    def _clean_text_for_json(self, text: str) -> str:
        """Clean text content to ensure it's safe for JSON serialization.
        
        Args:
            text: Raw text content
            
        Returns:
            Cleaned text safe for JSON
        """
        import re
        
        if not text or not text.strip():
            return "No text content available"
        
        # Remove null bytes and other problematic characters
        cleaned = text.replace('\x00', '')
        
        # Replace various types of quotes and special characters that can break JSON
        cleaned = cleaned.replace('\u201c', '"').replace('\u201d', '"')  # Smart quotes
        cleaned = cleaned.replace('\u2018', "'").replace('\u2019', "'")  # Smart apostrophes
        cleaned = cleaned.replace('\u2013', '-').replace('\u2014', '-')  # En/em dashes
        cleaned = cleaned.replace('\u00a0', ' ')  # Non-breaking space
        
        # Remove or replace other problematic Unicode characters
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', cleaned)
        
        # Normalize line endings
        cleaned = cleaned.replace('\r\n', '\n').replace('\r', '\n')
        
        # Remove excessive whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)  # Multiple blank lines
        cleaned = re.sub(r'[ \t]+', ' ', cleaned)  # Multiple spaces/tabs
        
        # Ensure we have some content
        cleaned = cleaned.strip()
        if not cleaned:
            return "No readable text content found in document"
            
        return cleaned

    def _create_product_extraction_prompt(self, text_content: str) -> str:
        """Create a prompt for extracting product information from text.
        
        Args:
            text_content: The extracted text from PDF
            
        Returns:
            Formatted prompt for the BookWyrm API
        """
        # Clean text content thoroughly
        cleaned_text = self._clean_text_for_json(text_content)
        
        # Truncate to avoid token limits and ensure we don't break mid-word
        truncated_text = cleaned_text[:4000]
        if len(cleaned_text) > 4000:
            # Find the last complete sentence or line break
            last_period = truncated_text.rfind('.')
            last_newline = truncated_text.rfind('\n')
            cutoff = max(last_period, last_newline)
            if cutoff > 3000:  # Only use if it's not too short
                truncated_text = truncated_text[:cutoff + 1]
        
        # Escape any remaining problematic characters for the f-string
        safe_text = truncated_text.replace('\\', '\\\\').replace('{', '{{').replace('}', '}}')
        
        return f"""Please analyze the following document text and extract structured product information suitable for an e-commerce product feed. 

Based on the content, identify and extract the following information in JSON format:

{{
    "id": "unique product identifier or SKU",
    "title": "product name/title (max 150 chars)",
    "description": "detailed product description (max 5000 chars)",
    "brand": "brand name if mentioned",
    "product_category": "category path using > separator",
    "price": "price with currency if mentioned (e.g., '99.99 USD')",
    "material": "primary materials if mentioned",
    "weight": "weight with unit if mentioned",
    "key_features": ["list", "of", "key", "features"],
    "target_audience": "intended users/demographic",
    "use_cases": ["primary", "use", "cases"],
    "specifications": {{"key": "value pairs of technical specs"}},
    "benefits": ["main", "benefits", "or", "value", "propositions"]
}}

If any information is not available in the text, use null for that field.
Focus on extracting factual, specific information rather than marketing language.

Document text:
{safe_text}"""
    
    def _create_simple_prompt(self, text_content: str) -> str:
        """Create a simple fallback prompt for basic product extraction.
        
        Args:
            text_content: The extracted text from PDF
            
        Returns:
            Simple prompt for the BookWyrm API
        """
        # Use only the first 1000 characters and clean them thoroughly
        cleaned_text = self._clean_text_for_json(text_content)[:1000]
        
        return f"Extract the product name, brand, and brief description from this text: {cleaned_text}"

    def _extract_product_data_from_text(self, extracted_text: ExtractedText) -> Dict[str, Any]:
        """Extract structured product data from text using BookWyrm API.
        
        Args:
            extracted_text: Extracted text from PDF
            
        Returns:
            Dictionary containing extracted product information
        """
        try:
            prompt = self._create_product_extraction_prompt(extracted_text.text_content)
            
            # Use BookWyrm API for summarization/extraction
            summary_response = None
            console.print(f"[blue]🤖[/blue] Processing {Path(extracted_text.file_path).name} with BookWyrm API...")
            
            try:
                # Validate prompt before sending
                if not prompt or not prompt.strip():
                    raise ValueError("Empty prompt generated")
                
                # Debug: Check prompt length and content
                console.print(f"[blue]📝[/blue] Prompt length: {len(prompt)} chars for {Path(extracted_text.file_path).name}")
                
                # Test JSON serialization of the prompt
                try:
                    import json
                    test_payload = {"content": prompt, "max_tokens": self.config.max_tokens}
                    json.dumps(test_payload)  # This will fail if there are JSON issues
                except (TypeError, ValueError) as json_error:
                    raise ValueError(f"Prompt contains characters that break JSON serialization: {json_error}")
                
                for response in self.client.stream_summarize(
                    content=prompt,
                    max_tokens=self.config.max_tokens,
                    model_strength="swift"  # Use swift for faster processing
                ):
                    if isinstance(response, SummaryResponse):
                        summary_response = response.summary
                        break
            except Exception as api_error:
                error_console.print(f"[red]BookWyrm API error for {Path(extracted_text.file_path).name}: {api_error}[/red]")
                
                # Try with a simpler prompt as fallback
                console.print(f"[yellow]🔄[/yellow] Trying simple prompt for {Path(extracted_text.file_path).name}...")
                try:
                    simple_prompt = self._create_simple_prompt(extracted_text.text_content)
                    for response in self.client.stream_summarize(
                        content=simple_prompt,
                        max_tokens=500,  # Use fewer tokens for simple prompt
                        model_strength="swift"
                    ):
                        if isinstance(response, SummaryResponse):
                            summary_response = response.summary
                            break
                    
                    if summary_response:
                        console.print(f"[green]✓[/green] Simple prompt worked for {Path(extracted_text.file_path).name}")
                except Exception as simple_error:
                    console.print(f"[red]Simple prompt also failed for {Path(extracted_text.file_path).name}: {simple_error}[/red]")
                
                # Debug: Save problematic content for inspection
                debug_file = Path(f"debug_{Path(extracted_text.file_path).stem}_prompt.txt")
                try:
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        f.write(f"Error: {api_error}\n\n")
                        f.write(f"Prompt length: {len(prompt)}\n\n")
                        f.write(f"Original text length: {len(extracted_text.text_content)}\n\n")
                        f.write("Prompt content:\n")
                        f.write(prompt)
                        f.write("\n\n" + "="*50 + "\n")
                        f.write("Original text content (first 2000 chars):\n")
                        f.write(extracted_text.text_content[:2000])
                    console.print(f"[yellow]Debug info saved to {debug_file}[/yellow]")
                except Exception:
                    pass  # Don't fail if we can't write debug file
                
                # If we still don't have a response, return fallback data structure
                if not summary_response:
                    return {
                        "id": Path(extracted_text.file_path).stem,
                        "title": f"Product from {Path(extracted_text.file_path).name}",
                        "description": f"Product information extracted from PDF document. API processing failed: {str(api_error)[:200]}",
                        "source_file": extracted_text.file_path,
                        "page_count": extracted_text.page_count
                    }
            
            if not summary_response:
                raise ValueError("No summary response received from BookWyrm API")
            
            # Try to parse JSON response
            try:
                product_data = json.loads(summary_response)
            except json.JSONDecodeError:
                # If not valid JSON, create a basic structure
                console.print(f"[yellow]Warning: Could not parse JSON from API response for {extracted_text.file_path}[/yellow]")
                product_data = {
                    "id": Path(extracted_text.file_path).stem,
                    "title": f"Product from {Path(extracted_text.file_path).name}",
                    "description": summary_response[:1000],  # Use first part of response
                    "brand": None,
                    "product_category": None,
                    "price": None,
                    "material": None,
                    "weight": None
                }
            
            # Add metadata
            product_data["source_file"] = extracted_text.file_path
            product_data["page_count"] = extracted_text.page_count
            
            return product_data
            
        except Exception as e:
            error_console.print(f"[red]Error extracting product data from {extracted_text.file_path}: {e}[/red]")
            # Return minimal data structure
            return {
                "id": Path(extracted_text.file_path).stem,
                "title": f"Product from {Path(extracted_text.file_path).name}",
                "description": "Product information extracted from PDF document",
                "source_file": extracted_text.file_path,
                "page_count": extracted_text.page_count
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
