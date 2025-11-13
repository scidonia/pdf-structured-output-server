"""User-configurable product extraction model.

This model defines the structure that BookWyrm API should extract
from phrasal text when processing product documents.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class ProductExtractionModel(BaseModel):
    """Pydantic model for structured product extraction from text.
    
    This model defines the structure that BookWyrm API should extract
    from phrasal text when processing product documents.
    """
    
    id: Optional[str] = Field(
        None,
        description="Unique product identifier, SKU, or model number found in the document"
    )
    
    title: Optional[str] = Field(
        None,
        max_length=150,
        description="Product name or title as mentioned in the document"
    )
    
    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed product description, features, and specifications from the document"
    )
    
    brand: Optional[str] = Field(
        None,
        max_length=70,
        description="Brand or manufacturer name mentioned in the document"
    )
    
    product_category: Optional[str] = Field(
        None,
        description="Product category or classification, use '>' separator for hierarchical categories"
    )
    
    price: Optional[str] = Field(
        None,
        description="Product price with currency if mentioned (e.g., '99.99 USD')"
    )
    
    material: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary materials or composition mentioned in the document"
    )
    
    weight: Optional[str] = Field(
        None,
        description="Product weight with unit if specified in the document"
    )
    
    key_features: Optional[List[str]] = Field(
        None,
        description="List of key features, benefits, or selling points mentioned"
    )
    
    specifications: Optional[Dict[str, str]] = Field(
        None,
        description="Technical specifications as key-value pairs found in the document"
    )
    
    target_audience: Optional[str] = Field(
        None,
        description="Intended users, demographic, or target market mentioned"
    )
    
    use_cases: Optional[List[str]] = Field(
        None,
        description="Primary use cases, applications, or purposes mentioned"
    )
