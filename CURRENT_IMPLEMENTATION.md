# Current BookWyrm API Implementation Analysis

## What We Have Implemented

### Current Workflow
1. **PDF Text Extraction** - Using `PDFProcessor` to extract raw text from PDFs
2. **Text Cleaning** - Cleaning extracted text for JSON safety
3. **Prompt Creation** - Creating text prompts asking for product information
4. **BookWyrm API Call** - Using `client.stream_summarize()` with text prompts
5. **Response Parsing** - Attempting to parse JSON from summary responses

### Current Issues
- ❌ **Wrong API Usage**: Using `stream_summarize()` with raw text and prompts
- ❌ **No Phrasal Processing**: Skipping the required phrasal text step
- ❌ **Manual Prompting**: Creating custom prompts instead of using Pydantic models
- ❌ **Wrong Endpoint**: Using `/summarize/sse` instead of appropriate endpoints

## What Needs to Change

### Required Changes
1. **Add Phrasal Processing Step**
   - Use `client.stream_process_text()` to convert raw text to phrases
   - Save phrases as JSONL format

2. **Create Product Extraction Pydantic Model**
   - Define a Pydantic model for the desired product structure
   - Remove manual prompt creation

3. **Use Correct API Method**
   - Use `client.stream_summarize()` with phrases and Pydantic model
   - Remove custom prompting

4. **Update Workflow**
   - Raw Text → Phrasal Processing → Structured Summarization → Product Data

### New Workflow Should Be
1. **PDF Text Extraction** (keep current)
2. **Phrasal Text Processing** (NEW - convert raw text to phrases)
3. **Structured Summarization** (CHANGE - use phrases + Pydantic model)
4. **Product Feed Generation** (keep current conversion logic)

### Files That Need Changes
- `src/product_enrichment/models.py` - Add product extraction Pydantic model
- `src/product_enrichment/product_feed_generator.py` - Complete rewrite of API usage
