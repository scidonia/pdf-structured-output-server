"""FastAPI server for product enrichment with streaming SSE responses."""

import asyncio
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional
from concurrent.futures import ThreadPoolExecutor
import tempfile
import os

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, create_model
from typing import Optional
import uvicorn

from .product_feed_generator import ProductFeedGenerator
from .models import ProcessingConfig

# Configure logging to ensure we see debug messages
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_app():
    """Factory function to create the FastAPI app for reload mode."""
    server = ProductEnrichmentServer(max_workers=5)
    return server.app


class SchemaRequest(BaseModel):
    """Request model for schema definition."""
    schema_name: str
    json_schema: Dict[str, Any]


class ProductEnrichmentServer:
    """FastAPI server for product enrichment with streaming responses."""
    
    def __init__(self, max_workers: int = 5):
        """Initialize the server.
        
        Args:
            max_workers: Number of parallel workers for processing
        """
        self.max_workers = max_workers
        self.app = FastAPI(
            title="Product Enrichment API",
            description="Process PDFs and extract structured product data using BookWyrm API",
            version="1.0.0"
        )
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.post("/process")
        async def process_pdfs(
            request: Request,
            files: List[UploadFile] = File(..., description="PDF files to process"),
            schema_name: str = Form(..., description="Name for the extraction schema"),
            json_schema: str = Form(..., description="JSON schema for extraction")
        ):
            """Process PDF files and return streaming results.
            
            Args:
                request: FastAPI Request object to access headers
                files: List of PDF files
                schema_name: Name for the extraction schema
                json_schema: JSON schema definition as string
                
            Returns:
                Streaming SSE response with extraction results
            """
            try:
                # Extract API key from Authorization header using request object
                authorization = request.headers.get("authorization")
                
                if not authorization:
                    raise HTTPException(status_code=401, detail="Authorization header is missing")
                
                if not authorization.startswith("Bearer "):
                    raise HTTPException(status_code=401, detail="Authorization header must start with 'Bearer '")
                
                api_key = authorization[7:].strip()  # Remove "Bearer " prefix and strip whitespace
                if not api_key:
                    raise HTTPException(status_code=401, detail="Bearer token is missing or empty")
                
                # Validate JSON schema
                try:
                    schema_dict = json.loads(json_schema)
                except json.JSONDecodeError as e:
                    raise HTTPException(status_code=400, detail=f"Invalid JSON schema: {e}")
                
                # Validate that files are PDFs
                pdf_files = []
                for file in files:
                    if not file.filename.lower().endswith('.pdf'):
                        raise HTTPException(status_code=400, detail=f"File {file.filename} is not a PDF")
                    pdf_files.append(file)
                
                if not pdf_files:
                    raise HTTPException(status_code=400, detail="No PDF files provided")
                
                # Create streaming response
                return StreamingResponse(
                    self._process_pdfs_stream(pdf_files, schema_name, schema_dict, api_key),
                    media_type="text/event-stream",
                    headers={
                        "Cache-Control": "no-cache",
                        "Connection": "keep-alive",
                        "Access-Control-Allow-Origin": "*",
                        "Access-Control-Allow-Headers": "Cache-Control"
                    }
                )
                
            except HTTPException:
                # Re-raise HTTP exceptions (like 401) without logging
                raise
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint."""
            return {"status": "healthy", "workers": self.max_workers}
    
    async def _process_pdfs_stream(
        self, 
        pdf_files: List[UploadFile], 
        schema_name: str, 
        schema_dict: Dict[str, Any],
        api_key: str
    ):
        """Stream processing results for PDF files.
        
        Args:
            pdf_files: List of uploaded PDF files
            schema_name: Name for the extraction schema
            schema_dict: JSON schema definition
            api_key: BookWyrm API key from Authorization header
            
        Yields:
            SSE formatted messages with processing results
        """
        try:
            # Send initial status
            yield f"data: {json.dumps({'type': 'status', 'message': f'Starting processing of {len(pdf_files)} files'})}\n\n"
            
            # Create dynamic Pydantic model from JSON schema
            logger.info(f"Creating dynamic model from schema: {schema_dict}")
            dynamic_model = self._create_dynamic_model(schema_name, schema_dict)
            logger.info(f"Created dynamic model: {dynamic_model}")
            
            # Create temporary directory for PDF files
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save uploaded files to temporary directory
                pdf_paths = []
                for file in pdf_files:
                    file_path = temp_path / file.filename
                    content = await file.read()
                    with open(file_path, 'wb') as f:
                        f.write(content)
                    pdf_paths.append(file_path)
                
                # Create generator with API key from request
                config = ProcessingConfig(
                    api_key=api_key,
                    max_tokens=2000,
                    batch_size=self.max_workers
                )
                generator = ProductFeedGenerator(config)
                
                # Process files in parallel and stream results
                loop = asyncio.get_event_loop()
                
                # Submit all processing tasks
                tasks = []
                for pdf_path in pdf_paths:
                    task = loop.run_in_executor(
                        self.executor,
                        self._process_single_pdf,
                        generator,
                        pdf_path,
                        dynamic_model
                    )
                    tasks.append((pdf_path.name, task))
                
                # Stream results as they complete
                completed_count = 0
                for filename, task in tasks:
                    try:
                        result = await task
                        completed_count += 1
                        
                        # Send progress update
                        progress_data = {
                            'type': 'progress',
                            'completed': completed_count,
                            'total': len(pdf_files),
                            'percentage': round((completed_count / len(pdf_files)) * 100, 1)
                        }
                        yield f"data: {json.dumps(progress_data)}\n\n"
                        
                        # Send result
                        result_data = {
                            'type': 'result',
                            'filename': filename,
                            'data': result
                        }
                        yield f"data: {json.dumps(result_data)}\n\n"
                        
                    except Exception as e:
                        # Send error for this file
                        error_data = {
                            'type': 'error',
                            'filename': filename,
                            'error': str(e)
                        }
                        yield f"data: {json.dumps(error_data)}\n\n"
                
                # Send completion message
                completion_data = {
                    'type': 'complete',
                    'message': f'Processed {completed_count} of {len(pdf_files)} files successfully'
                }
                yield f"data: {json.dumps(completion_data)}\n\n"
                
        except Exception as e:
            error_data = {
                'type': 'fatal_error',
                'error': str(e)
            }
            yield f"data: {json.dumps(error_data)}\n\n"
    
    
    def _create_dynamic_model(self, schema_name: str, schema_dict: Dict[str, Any]):
        """Create a dynamic Pydantic model from JSON schema.
        
        Args:
            schema_name: Name for the model
            schema_dict: JSON schema definition
            
        Returns:
            Dynamic Pydantic model class
        """
        try:
            # Extract properties from JSON schema
            properties = schema_dict.get('properties', {})
            
            if not properties:
                raise ValueError(f"No properties found in schema: {schema_dict}")
            
            # Convert JSON schema properties to Pydantic field definitions
            field_definitions = {}
            for field_name, field_schema in properties.items():
                field_type = self._json_type_to_python_type(field_schema)
                field_definitions[field_name] = (field_type, None)  # (type, default)
            
            # Create dynamic model
            dynamic_model = create_model(schema_name, **field_definitions)
            return dynamic_model
            
        except Exception as e:
            logger.error(f"Error creating dynamic model: {e}")
            raise Exception(f"Failed to create dynamic model: {e}")
    
    def _json_type_to_python_type(self, field_schema: Dict[str, Any]):
        """Convert JSON schema type to Python type.
        
        Args:
            field_schema: JSON schema field definition
            
        Returns:
            Python type for Pydantic field
        """
        json_type = field_schema.get('type', 'string')
        
        type_mapping = {
            'string': Optional[str],
            'integer': Optional[int],
            'number': Optional[float],
            'boolean': Optional[bool],
            'array': Optional[List[str]],  # Simplified - assume string arrays
            'object': Optional[Dict[str, Any]]
        }
        
        return type_mapping.get(json_type, Optional[str])

    def _process_single_pdf(
        self, 
        generator: ProductFeedGenerator, 
        pdf_path: Path, 
        dynamic_model
    ) -> Dict[str, Any]:
        """Process a single PDF file.
        
        Args:
            generator: ProductFeedGenerator instance
            pdf_path: Path to PDF file
            dynamic_model: Dynamic Pydantic model for extraction
            
        Returns:
            Extracted product data
        """
        try:
            # Extract text from PDF using BookWyrm
            text_content = generator._extract_pdf_with_bookwyrm(pdf_path)
            
            if not text_content or not text_content.strip():
                raise ValueError("No text content extracted from PDF")
            
            # Convert raw text to phrasal format
            phrases = generator._process_text_to_phrases(text_content, str(pdf_path))
            
            if not phrases:
                raise ValueError("No phrases generated from text")
            
            # Use structured summarization with dynamic model
            logger.info(f"Using dynamic model {dynamic_model.__name__} for PDF: {pdf_path.name}")
            stream = generator.client.stream_summarize(
                phrases=phrases,
                summary_class=dynamic_model,
                model_strength="wise",
                debug=False
            )
            
            # Collect the structured summary
            from bookwyrm.utils import collect_summary_from_stream
            final_result = collect_summary_from_stream(stream, verbose=False)
            
            if not final_result or not final_result.summary:
                raise ValueError("No structured summary received from BookWyrm API")
            
            # Convert result to dictionary
            if hasattr(final_result.summary, 'model_dump'):
                product_data = final_result.summary.model_dump()
            elif isinstance(final_result.summary, dict):
                product_data = final_result.summary
            elif isinstance(final_result.summary, str):
                # Try to parse as JSON
                try:
                    product_data = json.loads(final_result.summary)
                except json.JSONDecodeError:
                    product_data = {"raw_response": final_result.summary}
            else:
                product_data = {"raw_response": str(final_result.summary)}
            
            # Add metadata
            product_data["source_file"] = pdf_path.name
            product_data["page_count"] = len(text_content.split('\n'))
            
            return product_data
            
        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {e}")
            raise
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the server.
        
        Args:
            host: Host to bind to
            port: Port to bind to
            reload: Enable auto-reload on code changes
        """
        if reload:
            # For reload to work, we need to pass the app as an import string
            uvicorn.run(
                "product_enrichment.server:create_app", 
                host=host, 
                port=port, 
                reload=reload,
                factory=True
            )
        else:
            uvicorn.run(self.app, host=host, port=port, reload=reload)
