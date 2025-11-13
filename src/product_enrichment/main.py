"""Product enrichment CLI for processing PDFs and generating product feed CSV."""

import typer
from typing import Optional
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn, TimeElapsedColumn

from .product_feed_generator import ProductFeedGenerator
from .models import ProcessingConfig
from .server import ProductEnrichmentServer

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
    1. Extract PDF structure and text using BookWyrm API
    2. Convert text to phrasal format using BookWyrm API
    3. Generate structured product data using BookWyrm API with Pydantic models
    4. Output results to a CSV file based on OpenAI commerce feed spec
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
    
    generator = ProductFeedGenerator(config)
    
    # Create a separate console for progress to avoid interference
    from rich.console import Console as ProgressConsole
    progress_console = ProgressConsole()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        console=progress_console,
    ) as progress:
        # Process PDFs directly with BookWyrm API (PDF extraction + text processing + summarization)
        process_task = progress.add_task("Processing PDFs with BookWyrm API...", total=len(pdf_files))
        product_data = generator.generate_product_feed(pdf_files, progress, process_task)
        
        # Save to CSV
        save_task = progress.add_task("Saving to CSV...", total=1)
        generator.save_to_csv(product_data, output_file)
        progress.update(save_task, advance=1)
    
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


@app.command()
def serve(
    host: str = typer.Option(
        "0.0.0.0",
        "--host",
        help="Host to bind the server to"
    ),
    port: int = typer.Option(
        8000,
        "--port",
        help="Port to bind the server to"
    ),
    workers: int = typer.Option(
        5,
        "--workers",
        help="Number of parallel workers for processing PDFs"
    ),
) -> None:
    """Start the product enrichment API server.
    
    The server provides an endpoint that accepts:
    - Multiple PDF files via multipart form
    - A JSON schema for extraction
    - A schema name
    - Authorization header with Bearer token for BookWyrm API
    
    Returns streaming SSE responses with extraction results.
    """
    console.print(f"[green]Starting Product Enrichment API server...[/green]")
    console.print(f"[blue]Host: {host}[/blue]")
    console.print(f"[blue]Port: {port}[/blue]")
    console.print(f"[blue]Workers: {workers}[/blue]")
    console.print(f"[yellow]API endpoint: http://{host}:{port}/process[/yellow]")
    console.print(f"[yellow]Health check: http://{host}:{port}/health[/yellow]")
    console.print(f"[cyan]Note: API requests must include 'Authorization: Bearer <your-bookwyrm-api-key>' header[/cyan]")
    
    try:
        server = ProductEnrichmentServer(max_workers=workers)
        server.run(host=host, port=port)
    except KeyboardInterrupt:
        console.print(f"[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        error_console.print(f"[red]Error starting server: {e}[/red]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
