"""Pydantic models for product feed specification and processing configuration."""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, HttpUrl, validator
from enum import Enum
from datetime import date


class ProcessingConfig(BaseModel):
    """Configuration for PDF processing and API calls."""
    
    api_key: str = Field(
        ...,
        description="BookWyrm API key for summarization requests"
    )
    max_tokens: int = Field(
        2000,
        description="Maximum tokens for summarization API calls"
    )
    batch_size: int = Field(
        5,
        description="Number of PDFs to process in parallel"
    )


class EnableSearchEnum(str, Enum):
    """Enum for enable_search field."""
    TRUE = "true"
    FALSE = "false"


class EnableCheckoutEnum(str, Enum):
    """Enum for enable_checkout field."""
    TRUE = "true"
    FALSE = "false"


class ConditionEnum(str, Enum):
    """Enum for product condition."""
    NEW = "new"
    REFURBISHED = "refurbished"
    USED = "used"


class AvailabilityEnum(str, Enum):
    """Enum for product availability."""
    IN_STOCK = "in_stock"
    OUT_OF_STOCK = "out_of_stock"
    PREORDER = "preorder"


class AgeGroupEnum(str, Enum):
    """Enum for age group."""
    NEWBORN = "newborn"
    INFANT = "infant"
    TODDLER = "toddler"
    KIDS = "kids"
    ADULT = "adult"


class GenderEnum(str, Enum):
    """Enum for gender."""
    MALE = "male"
    FEMALE = "female"
    UNISEX = "unisex"


class PickupMethodEnum(str, Enum):
    """Enum for pickup method."""
    IN_STORE = "in_store"
    RESERVE = "reserve"
    NOT_SUPPORTED = "not_supported"


class RelationshipTypeEnum(str, Enum):
    """Enum for relationship type."""
    PART_OF_SET = "part_of_set"
    REQUIRED_PART = "required_part"
    OFTEN_BOUGHT_WITH = "often_bought_with"
    SUBSTITUTE = "substitute"
    DIFFERENT_BRAND = "different_brand"
    ACCESSORY = "accessory"


class ProductFeedItem(BaseModel):
    """Complete product feed item based on OpenAI commerce feed specification."""
    
    # OpenAI Flags (Required)
    enable_search: EnableSearchEnum = Field(
        ...,
        description="Controls whether the product can be surfaced in ChatGPT search results"
    )
    enable_checkout: EnableCheckoutEnum = Field(
        ...,
        description="Allows direct purchase inside ChatGPT"
    )
    
    # Basic Product Data (Required)
    id: str = Field(
        ...,
        max_length=100,
        description="Merchant product ID (unique)"
    )
    title: str = Field(
        ...,
        max_length=150,
        description="Product title"
    )
    description: str = Field(
        ...,
        max_length=5000,
        description="Full product description"
    )
    link: HttpUrl = Field(
        ...,
        description="Product detail page URL"
    )
    
    # Basic Product Data (Recommended/Optional)
    gtin: Optional[str] = Field(
        None,
        description="Universal product identifier (GTIN, UPC, ISBN)"
    )
    mpn: Optional[str] = Field(
        None,
        max_length=70,
        description="Manufacturer part number"
    )
    
    # Item Information
    condition: Optional[ConditionEnum] = Field(
        ConditionEnum.NEW,
        description="Condition of product"
    )
    product_category: Optional[str] = Field(
        None,
        description="Category path using '>' separator"
    )
    brand: Optional[str] = Field(
        None,
        max_length=70,
        description="Product brand"
    )
    material: Optional[str] = Field(
        None,
        max_length=100,
        description="Primary material(s)"
    )
    weight: Optional[str] = Field(
        None,
        description="Product weight with unit"
    )
    age_group: Optional[AgeGroupEnum] = Field(
        None,
        description="Target demographic"
    )
    
    # Media (Required/Optional)
    image_link: Optional[HttpUrl] = Field(
        None,
        description="Main product image URL"
    )
    additional_image_link: Optional[str] = Field(
        None,
        description="Extra images (comma-separated URLs)"
    )
    
    # Price & Promotions (Required)
    price: str = Field(
        ...,
        description="Regular price with currency code (e.g., '79.99 USD')"
    )
    sale_price: Optional[str] = Field(
        None,
        description="Discounted price with currency code"
    )
    
    # Availability & Inventory (Required)
    availability: AvailabilityEnum = Field(
        ...,
        description="Product availability status"
    )
    inventory_quantity: int = Field(
        ...,
        ge=0,
        description="Stock count"
    )
    
    # Variants (Optional)
    item_group_id: Optional[str] = Field(
        None,
        max_length=70,
        description="Variant group ID"
    )
    color: Optional[str] = Field(
        None,
        max_length=40,
        description="Variant color"
    )
    size: Optional[str] = Field(
        None,
        max_length=20,
        description="Variant size"
    )
    gender: Optional[GenderEnum] = Field(
        None,
        description="Gender target"
    )
    
    # Merchant Info (Required)
    seller_name: str = Field(
        ...,
        max_length=70,
        description="Seller name"
    )
    seller_url: HttpUrl = Field(
        ...,
        description="Seller page URL"
    )
    
    # Returns (Required)
    return_policy: HttpUrl = Field(
        ...,
        description="Return policy URL"
    )
    return_window: int = Field(
        ...,
        gt=0,
        description="Days allowed for return"
    )
    
    # Performance Signals (Recommended)
    popularity_score: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Popularity indicator (0-5 scale)"
    )
    
    # Reviews and Q&A (Recommended)
    product_review_count: Optional[int] = Field(
        None,
        ge=0,
        description="Number of product reviews"
    )
    product_review_rating: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Average review score (0-5 scale)"
    )


class ValidationResult(BaseModel):
    """Result of CSV validation."""
    
    is_valid: bool = Field(
        ...,
        description="Whether the CSV passes validation"
    )
    total_rows: int = Field(
        ...,
        description="Total number of rows processed"
    )
    errors: List[str] = Field(
        default_factory=list,
        description="List of validation errors"
    )
    warnings: List[str] = Field(
        default_factory=list,
        description="List of validation warnings"
    )


class ExtractedText(BaseModel):
    """Extracted text from a PDF file."""
    
    file_path: str = Field(
        ...,
        description="Path to the source PDF file"
    )
    text_content: str = Field(
        ...,
        description="Extracted text content"
    )
    page_count: int = Field(
        ...,
        description="Number of pages in the PDF"
    )
