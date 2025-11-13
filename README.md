# Product Enrichment

A CLI tool for processing PDF documents and generating structured product feed CSV files using the BookWyrm API. This tool extracts text from PDFs, uses AI to identify product information, and outputs data conforming to the OpenAI commerce feed specification.

## Features

- **PDF Text Extraction**: Parallel processing of multiple PDF documents
- **AI-Powered Analysis**: Uses BookWyrm API for intelligent product data extraction
- **Structured Output**: Generates CSV files following OpenAI commerce feed specification
- **Validation**: Built-in validation for generated CSV files
- **Batch Processing**: Configurable parallel processing for efficiency
- **Rich CLI**: User-friendly command-line interface with progress indicators

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Make sure you have uv installed, then:

```bash
uv sync
```

## Configuration

Set your BookWyrm API key as an environment variable:

```bash
export BOOKWYRM_API_KEY=your_api_key_here
```

Alternatively, you can pass the API key directly using the `--api-key` option.

## Usage

### Process PDFs

Process all PDF files in a directory and generate a product feed CSV:

```bash
product-enrichment process ./docs --output products.csv
```

#### Options

- `--output, -o`: Output CSV file path (default: `product_feed.csv`)
- `--api-key`: BookWyrm API key (can also use `BOOKWYRM_API_KEY` env var)
- `--max-tokens`: Maximum tokens for summarization (default: 2000)
- `--batch-size`: Number of PDFs to process in parallel (default: 5)

#### Example

```bash
# Process PDFs with custom settings
product-enrichment process ./documents \
  --output my_products.csv \
  --max-tokens 3000 \
  --batch-size 3
```

### Validate CSV

Validate a generated CSV file against the OpenAI commerce feed specification:

```bash
product-enrichment validate products.csv
```

This will check for:
- Required fields presence
- Field length constraints
- Data type validation
- Business rule compliance

## Output Format

The generated CSV includes fields from the OpenAI commerce feed specification:

### Required Fields
- `enable_search`: Controls ChatGPT search visibility
- `enable_checkout`: Enables direct purchase in ChatGPT
- `id`: Unique product identifier
- `title`: Product name (max 150 chars)
- `description`: Product description (max 5000 chars)
- `link`: Product detail page URL
- `price`: Price with currency code
- `availability`: Stock status
- `inventory_quantity`: Available quantity
- `seller_name`: Merchant name
- `seller_url`: Merchant page URL
- `return_policy`: Return policy URL
- `return_window`: Return period in days

### Optional Fields
- `gtin`: Universal product identifier
- `brand`: Product brand
- `product_category`: Category hierarchy
- `material`: Primary materials
- `weight`: Product weight
- `image_link`: Main product image
- `color`: Product color
- `size`: Product size
- And many more...

## How It Works

1. **PDF Processing**: Extracts text from all PDF files in the specified directory
2. **AI Analysis**: Uses BookWyrm API to analyze text and extract structured product information
3. **Data Mapping**: Maps extracted data to OpenAI commerce feed schema
4. **CSV Generation**: Outputs validated CSV file with all required fields
5. **Validation**: Optionally validates output against specification

## API Integration

This tool integrates with the [BookWyrm API](https://bookwyrm-client.readthedocs.io/) for intelligent text analysis and product data extraction. The API is used to:

- Analyze PDF content for product information
- Extract structured data like titles, descriptions, prices
- Identify product categories and specifications
- Generate marketing-friendly descriptions

## Error Handling

The tool includes comprehensive error handling:

- **PDF Processing Errors**: Continues processing other files if individual PDFs fail
- **API Errors**: Graceful fallback with basic product data structure
- **Validation Errors**: Clear error messages with specific field issues
- **File System Errors**: Proper error reporting for missing files/directories

## Development

### Project Structure

```
src/product_enrichment/
├── __init__.py              # Package initialization
├── main.py                  # CLI application entry point
├── models.py                # Pydantic models for data validation
├── pdf_processor.py         # PDF text extraction logic
└── product_feed_generator.py # BookWyrm API integration and CSV generation
```

### Adding New Fields

To add new fields to the product feed:

1. Update the `ProductFeedItem` model in `models.py`
2. Modify the extraction prompt in `product_feed_generator.py`
3. Update the field mapping in `_convert_to_product_feed_item()`
4. Add validation rules if needed

### Testing

Run the CLI to test functionality:

```bash
# Test help output
product-enrichment --help

# Test with sample PDFs
mkdir -p test_docs
# Add some PDF files to test_docs/
product-enrichment process test_docs --output test_output.csv

# Validate the output
product-enrichment validate test_output.csv
```

## Requirements

- Python 3.12+
- BookWyrm API key
- PDF files for processing

## Dependencies

- `bookwyrm`: API client for text analysis
- `typer`: CLI framework
- `rich`: Terminal formatting and progress bars
- `pydantic`: Data validation and serialization
- `pypdf`: PDF text extraction
- `pandas`: CSV generation and manipulation

## License

[Add your license information here]

## Contributing

[Add contribution guidelines here]
