"""User-configurable product extraction model.

This model defines the structure that BookWyrm API should extract
from phrasal text when processing product documents.
"""

from typing import Optional, List, Dict
from pydantic import BaseModel, Field


class TechnicalSpecifications(BaseModel):
    """Technical specifications for appliances based on CSV data patterns."""
    
    energy_efficiency_class: Optional[str] = Field(
        None,
        description="Energy efficiency rating (e.g., 'A', 'B', 'C')"
    )
    
    energy_consumption: Optional[str] = Field(
        None,
        description="Energy consumption per cycle or per 100 cycles (e.g., '49 kWh/100 cycles')"
    )
    
    water_consumption: Optional[str] = Field(
        None,
        description="Water consumption per cycle (e.g., '50 l/cycle')"
    )
    
    spin_speed: Optional[str] = Field(
        None,
        description="Maximum spin speed (e.g., '0-1400 rpm')"
    )
    
    noise_level: Optional[str] = Field(
        None,
        description="Noise level during operation (e.g., '72 dB(A) re 1pW')"
    )
    
    dimensions: Optional[str] = Field(
        None,
        description="Product dimensions in mm (e.g., '845x598x590 mm')"
    )
    
    voltage: Optional[str] = Field(
        None,
        description="Operating voltage (e.g., '220-240 V')"
    )
    
    frequency: Optional[str] = Field(
        None,
        description="Operating frequency (e.g., '50 Hz')"
    )
    
    programme_duration: Optional[str] = Field(
        None,
        description="Standard programme duration (e.g., '3:55 h')"
    )
    
    load_capacity: Optional[str] = Field(
        None,
        description="Maximum load capacity (e.g., '1-10 kg')"
    )


class ProductExtractionModel(BaseModel):
    """Pydantic model for structured product extraction from text.

    This model defines the structure that BookWyrm API should extract
    from phrasal text when processing product documents.
    """

    id: Optional[str] = Field(
        None,
        description="Unique product identifier, SKU, or model number found in the document",
    )

    title: Optional[str] = Field(
        None,
        max_length=150,
        description="Product name or title as mentioned in the document",
    )

    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Detailed product description, features, and specifications from the document",
    )

    brand: Optional[str] = Field(
        None,
        max_length=70,
        description="Brand or manufacturer name mentioned in the document",
    )

    product_category: Optional[str] = Field(
        None,
        description="Product category or classification, use '>' separator for hierarchical categories",
    )

    price: Optional[str] = Field(
        None, description="Product price with currency if mentioned (e.g., '99.99 USD')"
    )

    material: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary materials or composition mentioned in the document",
    )

    weight: Optional[str] = Field(
        None, description="Product weight with unit if specified in the document"
    )

    key_features: Optional[List[str]] = Field(
        None, description="List of key features, benefits, or selling points mentioned"
    )

    specifications: Optional[TechnicalSpecifications] = Field(
        None,
        description="Structured technical specifications found in the document"
    )

    target_audience: Optional[str] = Field(
        None, description="Intended users, demographic, or target market mentioned"
    )

    use_cases: Optional[List[str]] = Field(
        None, description="Primary use cases, applications, or purposes mentioned"
    )
