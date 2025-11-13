"""Product enrichment package for processing PDFs and generating product feeds."""

__version__ = "0.1.0"

from .main import app
from .models import ProductFeedItem, ProcessingConfig
from .pdf_processor import PDFProcessor
from .product_feed_generator import ProductFeedGenerator

__all__ = [
    "app",
    "ProductFeedItem", 
    "ProcessingConfig",
    "PDFProcessor",
    "ProductFeedGenerator"
]
