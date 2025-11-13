"""Product enrichment CLI for processing PDFs and generating product feed CSV."""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .pdf_processor import PDFProcessor
from .product_feed_generator import ProductFeedGenerator
from .models import ProcessingConfig

app = typer.Typer(help="Process PDF documents and generate product feed CSV using BookWyrm API")
console = Console()
error_console = Console(stderr=True)


@app.command()
def process(
    docs_dir: Path = typer.Argument(
        ...,
        help="Directory containing PDF documents to process",
        exists=True,
        file_okay=False,
        dir_okay=True,
    ),
    output_file: Path = typer.Option(
        "product_feed.csv",
        "--output", "-o",
        help="Output CSV file path"
    ),
    api_key: Optional[str] = typer.Option(
        None,
        "--api-key",
        envvar="BOOKWYRM_API_KEY",
        help="BookWyrm API key (can also be set via BOOKWYRM_API_KEY env var)"
    ),
    max_tokens: int = typer.Option(
        2000,
        "--max-tokens",
        help="Maximum tokens for summarization"
    ),
    batch_size: int = typer.Option(
        5,
        "--batch-size",
        help="Number of PDFs to process in parallel"
    ),
) -> None:
    """Process PDF documents and generate a product feed CSV file.
    
    This command will:
    1. Extract text from all PDF files in the specified directory
    2. Use BookWyrm API to generate product summaries
    3. Extract structured product data based on OpenAI commerce feed spec
    4. Output results to a CSV file
    """
    if not api_key:
        error_console.print("[red]Error: BookWyrm API key is required. Set BOOKWYRM_API_KEY environment variable or use --api-key option.[/red]")
        raise typer.Exit(1)
    
    # Find all PDF files
    pdf_files = list(docs_dir.glob("*.pdf"))
    if not pdf_files:
        error_console.print(f"[red]No PDF files found in {docs_dir}[/red]")
        raise typer.Exit(1)
    
    console.print(f"[green]Found {len(pdf_files)} PDF files to process[/green]")
    
    config = ProcessingConfig(
        api_key=api_key,
        max_tokens=max_tokens,
        batch_size=batch_size
    )
    
    processor = PDFProcessor(config)
    generator = ProductFeedGenerator(config)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        # Extract text from PDFs
        extract_task = progress.add_task("Extracting text from PDFs...", total=None)
        extracted_texts = processor.extract_texts_from_pdfs(pdf_files)
        progress.update(extract_task, completed=True)
        
        # Generate summaries and product data
        process_task = progress.add_task("Processing with BookWyrm API...", total=None)
        product_data = generator.generate_product_feed(extracted_texts)
        progress.update(process_task, completed=True)
        
        # Save to CSV
        save_task = progress.add_task("Saving to CSV...", total=None)
        generator.save_to_csv(product_data, output_file)
        progress.update(save_task, completed=True)
    
    console.print(f"[green]✓ Successfully processed {len(pdf_files)} PDFs and saved results to {output_file}[/green]")


@app.command()
def validate(
    csv_file: Path = typer.Argument(
        ...,
        help="CSV file to validate against OpenAI commerce feed spec",
        exists=True,
    )
) -> None:
    """Validate a CSV file against the OpenAI commerce feed specification."""
    generator = ProductFeedGenerator(ProcessingConfig(api_key="dummy"))
    
    try:
        validation_results = generator.validate_csv(csv_file)
        
        if validation_results.is_valid:
            console.print(f"[green]✓ CSV file is valid! Processed {validation_results.total_rows} rows.[/green]")
        else:
            console.print(f"[red]✗ CSV file has validation errors:[/red]")
            for error in validation_results.errors:
                console.print(f"  [red]• {error}[/red]")
            raise typer.Exit(1)
            
    except Exception as e:
        error_console.print(f"[red]Error validating CSV: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
