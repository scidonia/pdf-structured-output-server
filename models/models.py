"""User-configurable product extraction model.

This model defines the structure that BookWyrm API should extract
from phrasal text when processing product documents.

Based on OpenAI Commerce Feed Specification.
"""

from typing import Optional, List
from pydantic import BaseModel, Field


class ProductExtractionModel(BaseModel):
    """Pydantic model for structured product extraction from text.

    This model follows the OpenAI Commerce Feed Specification and defines
    the structure that BookWyrm API should extract from phrasal text when
    processing product documents.
    """

    # OpenAI Flags (Required)
    enable_search: Optional[str] = Field(
        "true",
        description="Controls whether the product can be surfaced in ChatGPT search results ('true' or 'false')",
    )

    enable_checkout: Optional[str] = Field(
        "false", description="Allows direct purchase inside ChatGPT ('true' or 'false')"
    )

    # Basic Product Data (Required)
    id: Optional[str] = Field(
        None,
        max_length=100,
        description="Merchant product ID (unique) - SKU or model number found in the document",
    )

    title: Optional[str] = Field(
        None, max_length=150, description="Product title as mentioned in the document"
    )

    description: Optional[str] = Field(
        None,
        max_length=5000,
        description="Full product description, features, and specifications from the document",
    )

    link: Optional[str] = Field(
        None, description="Product detail page URL (will be generated if not found)"
    )

    # Basic Product Data (Recommended/Optional)
    gtin: Optional[str] = Field(
        None, description="Universal product identifier (GTIN, UPC, ISBN) if mentioned"
    )

    mpn: Optional[str] = Field(
        None, max_length=70, description="Manufacturer part number if mentioned"
    )

    # Item Information
    condition: Optional[str] = Field(
        "new", description="Condition of product ('new', 'refurbished', 'used')"
    )

    product_category: Optional[str] = Field(
        None,
        description="Product category or classification, use '>' separator for hierarchical categories",
    )

    brand: Optional[str] = Field(
        None,
        max_length=70,
        description="Brand or manufacturer name mentioned in the document",
    )

    material: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary material(s) mentioned in the document",
    )

    dimensions: Optional[str] = Field(
        None,
        description="Overall dimensions with units (e.g., '12x8x5 in') if mentioned",
    )

    weight: Optional[str] = Field(
        None, description="Product weight with unit if specified in the document"
    )

    age_group: Optional[str] = Field(
        None,
        description="Target demographic ('newborn', 'infant', 'toddler', 'kids', 'adult')",
    )

    # Media
    image_link: Optional[str] = Field(
        None, description="Main product image URL if mentioned or can be inferred"
    )

    additional_image_link: Optional[str] = Field(
        None, description="Additional product images (comma-separated URLs)"
    )

    # Price & Promotions (Required)
    price: Optional[str] = Field(
        None,
        description="Regular price with currency code if mentioned (e.g., '79.99 USD')",
    )

    sale_price: Optional[str] = Field(
        None, description="Discounted price with currency code if mentioned"
    )

    # Availability & Inventory (Required)
    availability: Optional[str] = Field(
        "in_stock",
        description="Product availability status ('in_stock', 'out_of_stock', 'preorder')",
    )

    inventory_quantity: Optional[int] = Field(
        1, description="Stock count (default to 1 if not specified)"
    )

    # Variants
    item_group_id: Optional[str] = Field(
        None, max_length=70, description="Variant group ID if product has variants"
    )

    color: Optional[str] = Field(
        None, max_length=40, description="Variant color if mentioned"
    )

    size: Optional[str] = Field(
        None, max_length=20, description="Variant size if mentioned"
    )

    gender: Optional[str] = Field(
        None, description="Gender target ('male', 'female', 'unisex') if applicable"
    )

    # Merchant Info (Required)
    seller_name: Optional[str] = Field(
        "Example Store",
        max_length=70,
        description="Seller name (will use default if not found)",
    )

    seller_url: Optional[str] = Field(
        None, description="Seller page URL (will be generated if not found)"
    )

    # Returns (Required)
    return_policy: Optional[str] = Field(
        None, description="Return policy URL (will be generated if not found)"
    )

    return_window: Optional[int] = Field(
        30, description="Days allowed for return (default 30 if not specified)"
    )

    # Performance Signals (Recommended)
    popularity_score: Optional[float] = Field(
        None, description="Popularity indicator (0-5 scale) if mentioned"
    )

    # Reviews and Q&A (Recommended)
    product_review_count: Optional[int] = Field(
        None, description="Number of product reviews if mentioned"
    )

    product_review_rating: Optional[float] = Field(
        None, description="Average review score (0-5 scale) if mentioned"
    )

    # Technical Specifications (Custom for appliances)
    energy_efficiency_class: Optional[str] = Field(
        None, description="Energy efficiency rating (e.g., 'A', 'B', 'C') if mentioned"
    )

    energy_consumption: Optional[str] = Field(
        None,
        description="Energy consumption per cycle or per 100 cycles (e.g., '49 kWh/100 cycles') if mentioned",
    )

    water_consumption: Optional[str] = Field(
        None,
        description="Water consumption per cycle (e.g., '50 l/cycle') if mentioned",
    )

    spin_speed: Optional[str] = Field(
        None, description="Maximum spin speed (e.g., '0-1400 rpm') if mentioned"
    )

    noise_level: Optional[str] = Field(
        None,
        description="Noise level during operation (e.g., '72 dB(A) re 1pW') if mentioned",
    )

    voltage: Optional[str] = Field(
        None, description="Operating voltage (e.g., '220-240 V') if mentioned"
    )

    frequency: Optional[str] = Field(
        None, description="Operating frequency (e.g., '50 Hz') if mentioned"
    )

    programme_duration: Optional[str] = Field(
        None, description="Standard programme duration (e.g., '3:55 h') if mentioned"
    )

    load_capacity: Optional[str] = Field(
        None, description="Maximum load capacity (e.g., '1-10 kg') if mentioned"
    )

    # Additional fields for comprehensive extraction
    key_features: Optional[List[str]] = Field(
        None, description="List of key features, benefits, or selling points mentioned"
    )

    target_audience: Optional[str] = Field(
        None, description="Intended users, demographic, or target market mentioned"
    )

    use_cases: Optional[List[str]] = Field(
        None, description="Primary use cases, applications, or purposes mentioned"
    )

    # Metadata (added by system)
    source_file: Optional[str] = Field(
        None, description="Source PDF file path (added by system)"
    )

    page_count: Optional[int] = Field(
        None, description="Number of pages in source document (added by system)"
    )


class MultiProductExtractionModel(BaseModel):
    """Wrapper model for extracting multiple products from a single document.
    
    Use this model when processing catalogs, brochures, or documents that contain
    information about multiple products. The AI will identify and extract each
    product separately within the same document.
    """
    
    products: List[ProductExtractionModel] = Field(
        ...,
        description="List of products found in the document. Extract each distinct product as a separate item."
    )
