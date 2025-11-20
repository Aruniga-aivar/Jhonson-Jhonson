"""
J&J Invoice Processing Script - Enhanced Version with S3 Clustering and Document-Specific Templates

This script processes invoices with advanced PDF clustering and separate document processing using AWS Textract 
and AWS Bedrock for AI-powered data extraction.

Features:
- AWS Textract for superior PDF text extraction (key-value pairs, tables, forms)
- LLM-based PDF clustering to identify and separate different document types
- Automatic S3 storage of clustered PDFs under main PDF name folder
- Separate LLM converse calls for each clustered document
- Document-wise JSON results stored in separate folders (main_pdf_name_json_documentname)
- Document-specific schemas and prompt templates for different document types
- Freight invoices use S3-based templates (carrier-specific)
- Other documents (commercial invoice, air waybill, bill of lading, packing list) use hardcoded templates
- AWS Bedrock integration for AI-powered data extraction
- Local and S3 file output (JSON format)
- Automatic cleanup of temporary S3 files

Requirements:
- AWS credentials configured for Bedrock, Textract, and S3 access
- boto3: pip install boto3
- PyPDF2 or pdfplumber (fallback): pip install PyPDF2 or pip install pdfplumber
- PyMuPDF and PIL for advanced PDF processing: pip install PyMuPDF Pillow

Configuration:
- Update CONFIGURATION SETTINGS section with your AWS details
- Update DEFAULT_INPUT_FILE with your default PDF file path
- Update OUTPUT_DIR with your preferred output directory
- Update OUTPUT_BUCKET with your S3 bucket for clustered PDFs and results
- Update S3_BUCKET and S3_PREFIX for freight invoice prompt templates

Template Handling:
- Freight Invoices: Templates loaded from S3 (carrier-specific)
- Commercial Invoices: Hardcoded templates in the code
- Air Waybills: Hardcoded templates in the code
- Bills of Lading: Hardcoded templates in the code
- Packing Lists: Hardcoded templates in the code

Usage:
1. Run with hardcoded values: python J_J.py
2. Override file path: python J_J.py <file_path>
3. Override file path and carrier: python J_J.py <file_path> <carrier>
4. Override file path, carrier, and S3 bucket: python J_J.py <file_path> <carrier> <s3_bucket>

Example:
    python J_J.py invoice.pdf AAMRO my-s3-bucket

S3 Structure:
- clustered_pdfs/{main_pdf_name}/cluster_{id}_{doc_type}.pdf
- extractions/{main_pdf_name}_json_{doc_type}/{main_pdf_name}_{doc_type}_{timestamp}.json
- summary_reports/{main_pdf_name}_summary_{timestamp}.json

S3 Template Structure (for freight invoices):
- S3_BUCKET/S3_PREFIX contains carrier-specific templates
- Each carrier can have document-specific templates under 'document_templates'
"""

import json
import logging
import boto3
import os
import time
import jsonschema
import re
import sys
from typing import Dict, Any, Optional, List, Tuple
from botocore.exceptions import EndpointConnectionError, ClientError, ReadTimeoutError
from botocore.config import Config
from urllib.parse import unquote_plus
from datetime import datetime
from io import BytesIO
import pandas as pd

# Retry configuration for LLM calls
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds
RETRY_BACKOFF_MULTIPLIER = 2

def retry_llm_call(func, *args, **kwargs):
    """
    Retry wrapper for LLM calls to handle timeouts and temporary failures.
    
    Args:
        func: The function to retry
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    delay = RETRY_DELAY
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"LLM call attempt {attempt + 1}/{MAX_RETRIES}")
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"LLM call succeeded on attempt {attempt + 1}")
            return result
            
        except (ReadTimeoutError, EndpointConnectionError, ClientError) as e:
            last_exception = e
            error_type = type(e).__name__
            logger.warning(f"LLM call attempt {attempt + 1} failed with {error_type}: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= RETRY_BACKOFF_MULTIPLIER
            else:
                logger.error(f"All {MAX_RETRIES} LLM call attempts failed")
                
        except Exception as e:
            # For non-retryable errors, fail immediately
            logger.error(f"Non-retryable error in LLM call: {str(e)}")
            raise e
    
    # If we get here, all retries failed
    raise last_exception

def retry_textract_call(func, *args, **kwargs):
    """
    Retry wrapper for Textract calls to handle timeouts and temporary failures.
    
    Args:
        func: The Textract function to retry
        *args: Positional arguments for the function
        **kwargs: Keyword arguments for the function
    
    Returns:
        The result of the function call
        
    Raises:
        Exception: If all retries fail
    """
    last_exception = None
    delay = RETRY_DELAY
    
    for attempt in range(MAX_RETRIES):
        try:
            logger.info(f"Textract call attempt {attempt + 1}/{MAX_RETRIES}")
            result = func(*args, **kwargs)
            if attempt > 0:
                logger.info(f"Textract call succeeded on attempt {attempt + 1}")
            return result
            
        except (ReadTimeoutError, EndpointConnectionError, ClientError) as e:
            last_exception = e
            error_type = type(e).__name__
            logger.warning(f"Textract call attempt {attempt + 1} failed with {error_type}: {str(e)}")
            
            if attempt < MAX_RETRIES - 1:
                logger.info(f"Retrying Textract call in {delay} seconds...")
                time.sleep(delay)
                delay *= RETRY_BACKOFF_MULTIPLIER
            else:
                logger.error(f"All {MAX_RETRIES} Textract call attempts failed")
                
        except Exception as e:
            # For non-retryable errors, fail immediately
            logger.error(f"Non-retryable error in Textract call: {str(e)}")
            raise e
    
    # If we get here, all retries failed
    raise last_exception

# # Local PDF processing imports
# try:
#     import PyPDF2
#     PDF_LIBRARY = "PyPDF2"
# except ImportError:
#     try:
#         import pdfplumber
#         PDF_LIBRARY = "pdfplumber"
#     except ImportError:
#         print("Error: Please install either PyPDF2 or pdfplumber for PDF processing")
#         print("Run: pip install PyPDF2 or pip install pdfplumber")
#         sys.exit(1)

# PDF page extraction imports
try:
    import fitz  # PyMuPDF
    import io
    from PIL import Image
    import base64
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    import threading
    PDF_EXTRACTION_AVAILABLE = True
except ImportError as e:
    print(f"Warning: PDF page extraction not available: {e}")
    print("Install PyMuPDF and PIL for advanced PDF processing: pip install PyMuPDF Pillow")
    PDF_EXTRACTION_AVAILABLE = False

# ---------- CONFIGURE LOGGING ----------
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# ---------- CONFIGURATION SETTINGS ----------
# Core AWS configuration - REQUIRED
BEDROCK_MODEL_ID = 'us.anthropic.claude-3-7-sonnet-20250219-v1:0'  # Bedrock model ID for AI processing
REGION = 'us-east-1'  # AWS region for all services

# S3 configuration for prompt templates - REQUIRED
S3_BUCKET = 'pando-j-and-j-invoice'  # S3 bucket containing prompt templates
S3_PREFIX = 'templates/invoice/freight_prompt.py'  # S3 key for prompt template file

# Runtime configuration - OVERRIDDEN by lambda_function.py
OUTPUT_BUCKET = 'pando-j-and-j-output'  # Default S3 bucket (overridden by event in Lambda)
OUTPUT_DIR = '/Users/aruniga/Downloads/J_J output'  # Default local directory (overridden in Lambda)

# ---------- INITIALIZE AWS CLIENTS ----------
# Initialize all AWS clients including Textract for better PDF processing
custom_config = Config(
    connect_timeout=30,
    read_timeout=400
)
bedrock = boto3.client('bedrock-runtime', region_name=REGION, config=custom_config)
s3 = boto3.client('s3')
textract = boto3.client('textract', region_name=REGION)

logger.info("AWS services (S3, Textract, Bedrock) initialized successfully")

# ---------- GLOBAL PROMPT CACHE ----------
_PROMPT_TEMPLATE_CACHE = {}
_PROMPT_TEMPLATE_CACHE_TIMESTAMP = {}
_PROMPT_TEMPLATE_CACHE_TTL = 300  # in seconds, e.g., 5 minutes

# ---------- PROMPT TEMPLATE HANDLING ----------
def load_prompt_template_from_s3(bucket_name, key, force_refresh=False):
    """Load a prompt template from an S3 bucket with time-based cache invalidation."""
    cache_key = f"{bucket_name}/{key}"
    current_time = datetime.now().timestamp()
    
    # Check if we need to refresh the cache
    cache_expired = False
    if cache_key in _PROMPT_TEMPLATE_CACHE_TIMESTAMP:
        last_update_time = _PROMPT_TEMPLATE_CACHE_TIMESTAMP.get(cache_key, 0)
        cache_expired = (current_time - last_update_time) > _PROMPT_TEMPLATE_CACHE_TTL
    
    # Use cache only if it exists, isn't expired, and we're not forcing a refresh
    if not force_refresh and cache_key in _PROMPT_TEMPLATE_CACHE and not cache_expired:
        logger.info("Using cached prompt templates")
        return _PROMPT_TEMPLATE_CACHE[cache_key]

    # -------- ACTUAL LOAD from S3 if not cached --------
    try:
        logger.info(f"Loading prompt templates from S3: s3://{bucket_name}/{key}")
        obj = s3.get_object(Bucket=bucket_name, Key=key)
        content = obj['Body'].read().decode('utf-8')
        logger.info(f"S3 file content length: {len(content)} characters")
        logger.info(f"S3 file content preview (first 500 chars): {content[:500]}...")
        
        local_vars = {}
        exec(content, {}, local_vars)
        templates = local_vars.get("PROMPT_TEMPLATES", {})
        
        # Log the loaded templates
        logger.info(f"Available carriers: {list(templates.keys())}")
        logger.info(f"Template structure: {list(templates.keys()) if templates else 'No templates found'}")
        
        # Save to cache
        _PROMPT_TEMPLATE_CACHE[cache_key] = templates
        _PROMPT_TEMPLATE_CACHE_TIMESTAMP[cache_key] = current_time
        logger.info("Loaded and cached prompt templates from S3")
        return templates
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', '')
        if error_code == 'NoSuchKey':
            logger.warning(f"S3 template file not found: s3://{bucket_name}/{key}")
            logger.info("This is expected if S3 templates are not set up. System will use fallback templates.")
        else:
            logger.error(f"AWS ClientError loading prompt template from S3: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading prompt template from S3: {e}", exc_info=True)
        return {}


def get_prompt_template(carrier_name: str, document_type: str = None) -> Optional[str]:
    """Get the prompt template for a specific carrier and document type."""
    try:
        logger.info(f"Getting prompt template for carrier: {carrier_name}, document_type: {document_type}")
        
        # Normalize document type
        if document_type:
            doc_type_key = document_type.lower().replace(' ', '_').replace('-', '_')
        else:
            doc_type_key = None
        
        # Check if this is a freight invoice - use S3 templates ONLY for freight invoices
        is_freight_invoice = (
            doc_type_key in ['freight_invoice', 'freight_invoices']
        )
        
        if is_freight_invoice:
            logger.info("Using S3 template for freight invoice")
            return get_freight_invoice_template_from_s3(carrier_name, document_type)
        else:
            # Use hardcoded templates for non-freight documents
            logger.info(f"Using hardcoded template for document type: {document_type}")
            return get_hardcoded_document_template(document_type)
        
    except Exception as e:
        logger.error(f"Error loading prompt template for {carrier_name}: {str(e)}", exc_info=True)
        return None

def get_freight_invoice_template_from_s3(carrier_name: str, document_type: str = None) -> Optional[str]:
    """Get freight invoice template from S3 with fallback to hardcoded template."""
    try:
        # Load all templates from S3
        all_templates = load_prompt_template_from_s3(S3_BUCKET, S3_PREFIX)
        logger.info(f"Loaded templates from S3. Available carriers: {list(all_templates.keys())}")
        
        # Try exact match first
        carrier_data = all_templates.get(carrier_name)
        if not carrier_data:
            # Try case-insensitive match
            logger.info(f"No exact match for '{carrier_name}', trying case-insensitive match...")
            for template_carrier in all_templates.keys():
                if template_carrier.upper() == carrier_name.upper():
                    carrier_data = all_templates[template_carrier]
                    logger.info(f"Found case-insensitive match: '{template_carrier}' for '{carrier_name}'")
                    break
        
        if not carrier_data:
            logger.warning(f"No prompt template found for carrier: {carrier_name} in S3")
            logger.warning(f"Available carriers in S3: {list(all_templates.keys())}")
            logger.info("Falling back to hardcoded freight invoice template")
            return FALLBACK_FREIGHT_INVOICE_TEMPLATE
        
        # If document_type is specified, try to get document-specific template
        if document_type:
            # Normalize document type
            doc_type_key = document_type.lower().replace(' ', '_').replace('-', '_')
            
            # Check for document-specific templates
            if isinstance(carrier_data, dict):
                # Look for document-specific templates within carrier data
                doc_templates = carrier_data.get('document_templates', {})
                if doc_templates and doc_type_key in doc_templates:
                    prompt_template = doc_templates[doc_type_key]
                    logger.info(f"Found document-specific template for {carrier_name} - {document_type}")
                    return prompt_template
                
                # Try alternative document type keys
                alternative_keys = [
                    doc_type_key,
                    doc_type_key.replace('_', ''),
                    doc_type_key.replace('_invoice', ''),
                    doc_type_key.replace('freight', 'invoice')
                ]
                
                for alt_key in alternative_keys:
                    if alt_key in doc_templates:
                        prompt_template = doc_templates[alt_key]
                        logger.info(f"Found document-specific template for {carrier_name} - {document_type} (using key: {alt_key})")
                        return prompt_template
            
            logger.info(f"No document-specific template found for {document_type}, using default template")
        
        # Use default template - handle new structure where template is nested under 'prompt_template'
        if isinstance(carrier_data, dict):
            prompt_template = carrier_data.get('prompt_template')
        else:
            # Handle legacy structure where carrier_data might be the template string directly
            prompt_template = carrier_data
            
        if not prompt_template:
            logger.warning(f"Missing prompt template for {carrier_name} in S3")
            logger.info("Falling back to hardcoded freight invoice template")
            return FALLBACK_FREIGHT_INVOICE_TEMPLATE
            
        logger.info(f"Successfully loaded S3 prompt template for {carrier_name} (length: {len(prompt_template)} chars)")
        # Log a preview of the template
        logger.info(f"Prompt template preview (first 200 chars): {prompt_template[:200]}...")
        return prompt_template
        
    except Exception as e:
        logger.error(f"Error loading S3 prompt template for {carrier_name}: {str(e)}")
        logger.info("Falling back to hardcoded freight invoice template")
        return FALLBACK_FREIGHT_INVOICE_TEMPLATE

def get_hardcoded_document_template(document_type: str) -> Optional[str]:
    """Get hardcoded template for non-freight document types."""
    try:
        if not document_type:
            logger.error("Document type is required for hardcoded templates")
            return None

        # Normalize document type
        doc_type_key = document_type.lower().replace(' ', '_').replace('-', '_')
        
        # Try exact match first
        if doc_type_key in HARDCODED_DOCUMENT_TEMPLATES:
            template = HARDCODED_DOCUMENT_TEMPLATES[doc_type_key]
            logger.info(f"Found hardcoded template for document type: {doc_type_key}")
            return template
        
        # Try alternative keys
        alternative_keys = [
            doc_type_key.replace('_', ''),
            doc_type_key.replace('_invoice', ''),
            doc_type_key.replace('_waybill', ''),
            doc_type_key.replace('bill_of_lading', 'air_waybill'),  # Map bill_of_lading to air_waybill (same as schema mapping)
            doc_type_key.replace('packing_list', 'packing'),
            'air_waybill',  # Fallback for bill_of_lading and similar document types
            'airway_bill'   # Additional fallback variant
        ]
        
        for alt_key in alternative_keys:
            if alt_key in HARDCODED_DOCUMENT_TEMPLATES:
                template = HARDCODED_DOCUMENT_TEMPLATES[alt_key]
                logger.info(f"Found hardcoded template for document type: {doc_type_key} (using key: {alt_key})")
                return template
        
        logger.error(f"No hardcoded template found for document type: {document_type}")
        logger.error(f"Available hardcoded templates: {list(HARDCODED_DOCUMENT_TEMPLATES.keys())}")
        return None
        
    except Exception as e:
        logger.error(f"Error loading hardcoded template for {document_type}: {str(e)}", exc_info=True)
        return None

# ---------- DEFINE JSON SCHEMAS FOR DIFFERENT DOCUMENT TYPES ----------

# Base schema structure for all documents
def create_base_schema_field():
    return {
    "type": "object",
    "properties": {
            "value": {"type": ["string", "number", "null"]},
                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                "explanation": {"type": "string"}
            },
            "required": ["value", "confidence", "explanation"]
    }

# Freight Invoice Schema
FREIGHT_INVOICE_SCHEMA = {
            "type": "object",
    "required": ["data"],
            "properties": {
        "data": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "invoice_number": create_base_schema_field(),
                    "invoice_date": create_base_schema_field(),
        # "payment_due_date": create_base_schema_field(),
        "payment_terms": create_base_schema_field(),
                    "vendor_reference_id": create_base_schema_field(),
        "currency": create_base_schema_field(),
                    "total_invoice_value": create_base_schema_field(),
                    "total_tax_amount": create_base_schema_field(),
                    "bill_of_lading_number": create_base_schema_field(),
                    "bill_to_name": create_base_schema_field(),
                    "bill_to_gst": create_base_schema_field(),
                    "bill_to_address": create_base_schema_field(),
                    "bill_to_phone_number": create_base_schema_field(),
                    "bill_to_email": create_base_schema_field(),
                    "cost_center": create_base_schema_field(),
                    "billing_entity_name": create_base_schema_field(),
                    
                    "shipments": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "shipment_number": create_base_schema_field(),
                                "mode": create_base_schema_field(),
                                "pro_number": create_base_schema_field(),
                                "source_name": create_base_schema_field(),
                                "destination_name": create_base_schema_field(),
                                "source_code": create_base_schema_field(),
                                "source_city": create_base_schema_field(),
                                "source_state": create_base_schema_field(),
                                "source_country": create_base_schema_field(),
                                "source_zone": create_base_schema_field(),
                                "source_zip_code": create_base_schema_field(),
                                "source_address": create_base_schema_field(),
                                "destination_code": create_base_schema_field(),
                                "destination_city": create_base_schema_field(),
                                "destination_state": create_base_schema_field(),
                                "destination_country": create_base_schema_field(),
                                "destination_zone": create_base_schema_field(),
                                "destination_zip_code": create_base_schema_field(),
                                "destination_address": create_base_schema_field(),
                                "shipment_weight": create_base_schema_field(),
                                "shipment_weight_uom": create_base_schema_field(),
                                "shipment_volume": create_base_schema_field(),
                                "shipment_volume_uom": create_base_schema_field(),
                                "shipment_distance": create_base_schema_field(),
                                "shipment_distance_uom": create_base_schema_field(),
                                "shipment_total_value": create_base_schema_field(),
                                "shipment_tax_value": create_base_schema_field(),
                                "shipment_creation_date": create_base_schema_field(),
        "charges": {
            "type": "array",
            "items": {
            "type": "object",
            "properties": {
                    "charge_code": create_base_schema_field(),
                    "charge_name": create_base_schema_field(),
                    "charge_gross_amount": create_base_schema_field(),
                                            "charge_tax_amount": create_base_schema_field(),
                                            "currency": create_base_schema_field()
                },
                "required": ["charge_code", "charge_name", "charge_gross_amount"]
            }
        },
                                "port_of_loading": create_base_schema_field(),
                                "origin_service_type": create_base_schema_field(),
                                "destination_service_type": create_base_schema_field(),
                                "port_of_discharge": create_base_schema_field(),
                                "custom": {
            "type": "object",
                                    "properties": {
                                        "service_code": create_base_schema_field(),
                                        "container_type": create_base_schema_field(),
                                        "special_handling": create_base_schema_field(),
                                        "unnumber_count": create_base_schema_field(),
                                        "data_loggercount": create_base_schema_field(),
                                        "container_count": create_base_schema_field(),
                                        "origin_terminal_handling_days_count": create_base_schema_field(),
                                        "destination_terminal_handling_days_count": create_base_schema_field(),
                                        "thermal_blanket": create_base_schema_field(),
                                        "uld_extra_lease_day": create_base_schema_field(),
                                        "hazardous_material": create_base_schema_field(),
                                        "actual_weight": create_base_schema_field(),
                                        "actual_weight_uom": create_base_schema_field(),
                                        "total_package": create_base_schema_field(),
                                        "cargo": create_base_schema_field(),
                                        "temperature_control": create_base_schema_field(),
                                        "lane_id": create_base_schema_field()
                                    }
                                }
                            },
                            "required": ["shipment_number", "source_name", "destination_name"]
                        }
                    },
                    "taxes": {
            "type": "array",
            "items": {
            "type": "object",
            "properties": {
                                "tax_code": create_base_schema_field(),
                                "tax_name": create_base_schema_field(),
                                "tax_percentage": create_base_schema_field(),
                                "tax_amount": create_base_schema_field()
                            }
                        }
                    },
                    "custom_charges": {
            "type": "array",
            "items": {
                "type": "object",
                            "properties": {
                                "charge_name": create_base_schema_field(),
                                "charge_gross_amount": create_base_schema_field(),
                                "currency": create_base_schema_field()
                            }
                        }
                    },
                    "custom": {
                        "type": "object",
                        "properties": {
                            "reference_number": create_base_schema_field(),
                            "special_instructions": create_base_schema_field(),
                            "priority": create_base_schema_field(),
                            "pay_as_present": create_base_schema_field()
                        }
                    },
                    "shipment_identifiers": {
                        "type": "object",
                        "properties": {
                            "booking_number": create_base_schema_field(),
                            "container_numbers": {
            "type": "array",
            "items": {
                                    "type": "string"
                                }
                            }
                        }
                    }
                },
                "required": ["invoice_number", "invoice_date", "total_invoice_value", "shipments"]
            }
        }
    }
}

# Commercial Invoice Schema
COMMERCIAL_INVOICE_SCHEMA ={
    "type": "object",
  "required": ["commercial_invoice"],
                        "properties": {
    "commercial_invoice": {
            "type": "array",
            "items": {
                        "type": "object",
        "properties": {
          "bill_to_name": create_base_schema_field(),
          "bill_to_address": create_base_schema_field(),
          "bill_to_city": create_base_schema_field(),
          "bill_to_country": create_base_schema_field(),
          "ship_to_name": create_base_schema_field(),
          "ship_to_address": create_base_schema_field(),
          "ship_to_city": create_base_schema_field(),
          "ship_to_country": create_base_schema_field(),
          "purchase_order_no": create_base_schema_field(),
          "ship_quantity": create_base_schema_field(),
          "product_number": create_base_schema_field(),
          "product_name": create_base_schema_field(),
          "unit_price": create_base_schema_field(),
          "total_price": create_base_schema_field(),
          "UoM": create_base_schema_field()
        },
        "required": ["bill_to_name", "ship_to_name"]
      }
    }
  }
}

# Air Waybill Schema
AIR_WAYBILL_SCHEMA = {
                        "type": "object",
  "required": ["airway_bill"],
                        "properties": {
    "airway_bill": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
          "shipper_name": create_base_schema_field(),
          "shipper_address": create_base_schema_field(),
          "shipper_city": create_base_schema_field(),
          "shipper_state": create_base_schema_field(),
          "shipper_country": create_base_schema_field(),
          "shipper_zipcode": create_base_schema_field(),
          "consignee_name": create_base_schema_field(),
          "consignee_address": create_base_schema_field(),
          "consignee_city": create_base_schema_field(),
          "consignee_state": create_base_schema_field(),
          "consignee_country": create_base_schema_field(),
          "consignee_zipcode": create_base_schema_field(),
          "airport_of_departure": create_base_schema_field(),
          "airport_of_destination": create_base_schema_field(),
          "flight_code": create_base_schema_field(),
          "flight_number": create_base_schema_field(),
          "airway_bill_number": create_base_schema_field(),
          "handling_information": create_base_schema_field(),
          "number_of_pieces": create_base_schema_field(),
          "gross_weight": create_base_schema_field(),
          "chargeable_weight": create_base_schema_field(),
          "airway_bill_parsed_text": {
                        "type": "object",
                "additionalProperties": True
          }
        },
        "required": ["shipper_name", "consignee_name", "airway_bill_number"]
      }
    }
  }
}

# Bill of Lading Schema


# Packing List Schema
PACKING_LIST_SCHEMA = {
    "type": "object",
  "required": ["packing_list"],
    "properties": {
    "packing_list": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
          "invoice_number": create_base_schema_field(),
          "invoice_date": create_base_schema_field(),
          "origin_city": create_base_schema_field(),
          "destination_city": create_base_schema_field(),
          "number_of_pallets": create_base_schema_field(),
          "gross_weight": create_base_schema_field(),
          "net_volume": create_base_schema_field(),
          "volume_uom": create_base_schema_field(),
          "weight_uom": create_base_schema_field(),
          "ship_Id": create_base_schema_field(),
          "Ship_Date": create_base_schema_field()
        },
        "required": ["invoice_number", "gross_weight"]
      }
    }
  }
}
# Import Declaration Schema

# Receipt Schema
DELIVERY_NOTE_SCHEMA = {
    "type": "object",
  "required": ["delivery_note"],
                "properties": {
    "delivery_note": {
            "type": "array",
            "items": {
                        "type": "object",
                        "properties": {
          "incoterms": create_base_schema_field(),
          "delivery_number": create_base_schema_field(),
          "po_number": create_base_schema_field(),
          "bill_to_name": create_base_schema_field(),
          "bill_to_address": create_base_schema_field(),
          "bill_to_city": create_base_schema_field(),
          "bill_to_state": create_base_schema_field(),
          "sold_to_name": create_base_schema_field(),
          "sold_to_city": create_base_schema_field(),
          "sold_to_country": create_base_schema_field(),
          "sold_to_name": create_base_schema_field(),
          "sold_to_city": create_base_schema_field(),
          "sold_to_country": create_base_schema_field(),
          "product_number": create_base_schema_field(),
          "product_name": create_base_schema_field(),
          "qty_ordered": create_base_schema_field(),
          "product_uom": create_base_schema_field(),
          "qty_picked": create_base_schema_field()
        },
        "required": ["delivery_number", "bill_to_name"]
            }
        }
    }
}

# Document Schema Mapping
DOCUMENT_SCHEMAS = {
    "freight_invoice": FREIGHT_INVOICE_SCHEMA,
    "freight_invoices": FREIGHT_INVOICE_SCHEMA,
    "commercial_invoice": COMMERCIAL_INVOICE_SCHEMA,
    "commercial_invoices": COMMERCIAL_INVOICE_SCHEMA,
    "air_waybill": AIR_WAYBILL_SCHEMA,
    "air_waybills": AIR_WAYBILL_SCHEMA,
    "airway_bill": AIR_WAYBILL_SCHEMA,  # Add this line
    "airway_bills": AIR_WAYBILL_SCHEMA,
    "sea_waybills": AIR_WAYBILL_SCHEMA,
    "packing_list": PACKING_LIST_SCHEMA,
    "packing_lists": PACKING_LIST_SCHEMA,
    "delivery_note": DELIVERY_NOTE_SCHEMA,
    "delivery_notes": DELIVERY_NOTE_SCHEMA
}

# Legacy schema for backward compatibility
INVOICE_SCHEMA = FREIGHT_INVOICE_SCHEMA

# ---------- HARDCODED PROMPT TEMPLATES FOR NON-FREIGHT DOCUMENTS ----------

# Commercial Invoice Prompt Template
COMMERCIAL_INVOICE_PROMPT_TEMPLATE = """
You are an expert document analyzer specializing in extracting structured information from commercial invoices.

Your task is to analyze the provided commercial invoice text and extract key information in a structured JSON format according to the specified schema.

## Commercial Invoice Text:
{pdf_text}

## Required JSON Schema:
You must extract the following fields and return them in this structure:

## Instructions:
1. Carefully analyze the commercial invoice text to locate all required fields
2. For Bill To and Ship To information, extract the complete name, address, city, and country
3. If a field contains multiple values (like product numbers), include all of them in a comma-separated string
4. Sum all quantities to get the total ship_quantity
5. If information is not available, return an empty string for that field
6. For purchase_order_no, extract all values if there are multiple orders referenced
7. Ensure all extracted information is complete and accurate

## Field-by-Field Extraction Guidelines:
- bill_to_name: Look for "Bill To", "Sold To", or "Customer" sections
- bill_to_address: Extract the street address from the Bill To section
- bill_to_city: Extract only the city name from the Bill To section
- bill_to_country: Extract only the country name from the Bill To section
- ship_to_name: Look for "Ship To", "Deliver To", or "Consignee" sections
- ship_to_address: Extract the street address from the Ship To section
- ship_to_city: Extract only the city name from the Ship To section
- ship_to_country: Extract only the country name from the Ship To section
- purchase_order_no: Look for "Order No", "PO Number", "Purchase Order" fields
- ship_quantity: Sum all quantities listed in the product table
- product_number: List all the product numbers from Product table
- product_name: List the name of the product from Product description
- unit_price: Extract all unit prices for each product
- total_price: Extract all extended or total prices for each product
- UoM: Take value from UoM, that indicates the unit of measure (e.g., EA, PCS, KG)

Please ensure your extraction is accurate and follows the exact schema structure provided.
"""


# Air Waybill Prompt Template
AIRWAY_BILL_PROMPT_TEMPLATE = """
You are an expert document analyzer specializing in extracting structured information from Airway Bills (AWBs).

Your task is to analyze the provided Airway Bill text and extract key information in a structured JSON format according to the specified schema.

## Airway Bill Text:
{pdf_text}


## Instructions:
1. Carefully analyze the Airway Bill text to locate all required fields
2. For Shipper and Consignee information, extract the complete name, address, city, state, country, and zipcode separately
3. Pay attention to the formatting of addresses to correctly separate city, state, country, and zipcode
4. If information is not available, return an empty string for that field
5. For airway_bill_parsed_text, return an empty object {}
6. Ensure all extracted information is complete and accurate

## Field-by-Field Extraction Guidelines:
- shipper_name: Extract the company/individual name from the shipper section
- shipper_address: Extract only the street address from the shipper section
- shipper_city: Extract only the city name from the shipper section
- shipper_state: Extract only the state/province from the shipper section
- shipper_country: Extract only the country name or code from the shipper section
- shipper_zipcode: Extract only the postal/zip code from the shipper section
- consignee_name: Extract the company/individual name from the consignee section
- consignee_address: Extract only the street address from the consignee section
- consignee_city: Extract only the city name from the consignee section
- consignee_state: Extract only the state/province from the consignee section
- consignee_country: Extract only the country name or code from the consignee section
- consignee_zipcode: Extract only the postal/zip code from the consignee section
- airport_of_departure: Look for "Airport of Departure", "From", "Origin", etc.
- airport_of_destination: Look for "Airport of Destination", "To", "Destination", etc.
- flight_code: Extract the airline code, typically 2 letters (e.g., "UA" for United Airlines)
- flight_number: Extract the complete flight number including the airline code (e.g., "UA0930")
- airway_bill_number: In the Air Way Bill (AWB) text section, extract the airway bill number by:
  Searching for labels such as "MAWB:", "AWB No:", "Air Waybill No:", and extracting the number after the label.
  If no labeled airway bill number exists, search for a standalone alphanumeric string matching airway bill number formats (e.g., HHKG10061348, 160-56563783).
  If not found, return "".
- handling_information: Extract any special instructions or handling notes
- number_of_pieces: Take data from No.of Pieces.
- gross_weight: Take data from Gross weight.
- chargeable_weight: Take data from Chargeable weight.

Please ensure your extraction is accurate and follows the exact schema structure provided.
"""


# Packing List Prompt Template
PACKING_LIST_PROMPT_TEMPLATE = """
You are an expert document analyzer specializing in extracting structured information from Packing Lists.

Your task is to analyze the provided Packing List text and extract key information in a structured JSON format according to the specified schema.

## Packing List Text:
{pdf_text}



## Instructions:
1. Carefully analyze the Packing List text to locate all required fields
2. Look for shipping information including origin, destination, weights, and dates
3. If information is not available, return an empty string for that field
4. Pay special attention to extracting the Ship ID and Ship Date
5. Ensure all extracted information is complete and accurate

## Field-by-Field Extraction Guidelines:
- invoice_number: Look for references to "Invoice", "Invoice No.", "Inv#", etc.
- invoice_date: Look for dates associated with invoices
- origin_city: Extract city name from the "Ship From" or "Origin" section
- destination_city: Extract city name from the "Ship To" or "Destination" section
- number_of_pallets: Look for "Pallets", "Pallet Count", "No. of Pallets", etc.
- gross_weight: Look for "Weight", "Gross Wt", "Total Weight" in pallet dimensions section
- net_volume: Look for "Volume", "Net Vol", "Cubic Measurement", etc.
- volume_uom: Extract the unit of measure for volume (e.g., "CBM", "cu ft")
- weight_uom: Extract the unit of measure for weight, typically "KG" or "LB"
- ship_Id: Look for "Ship ID", "Shipment ID", "SID", etc., e.g. "SID5958808"
- Ship_Date: Look for "Ship Date", "Shipping Date", "Date of Shipment", etc., e.g. "4/16/25"

Please ensure your extraction is accurate and follows the exact schema structure provided.
"""

DELIVERY_NOTE_PROMPT_TEMPLATE = """
You are an expert document analyzer specializing in extracting structured information from Delivery Notes.

Your task is to analyze the provided Delivery Note text and extract key information in a structured JSON format according to the specified schema.

## Delivery Note Text:
{pdf_text}


## Instructions:
1. Carefully analyze the Delivery Note text to locate all required fields
2. Extract all company information (Bill-To, Sold-To, and Ship-To) separately
3. For product information (product_number, product_name, qty_ordered, product_uom, qty_picked), compile all items as comma-separated lists in their respective fields
4. If information is not available, return an empty string for that field
5. Ensure all extracted information is complete and accurate

## Field-by-Field Extraction Guidelines:
- incoterms: Look for "IncoTerms", "Terms", "Shipping Terms", etc., e.g. "Ex-Works"
- delivery_number: Look for "Delivery Number", e.g. "801201169"
- po_number: Look for "Customer PO Number", "PO No.", etc., e.g. "4200130425"
- bill_to_name: Extract company name from the billing information at the top of the document
- bill_to_address: Extract street address from the billing information at the top of the document
- bill_to_city: Extract city from the billing information at the top of the document
- bill_to_state: Extract state/province from the billing information at the top of the document
- sold_to_name: Extract company name from the "Sold To" section
- sold_to_city: Extract city from the "Sold To" section
- sold_to_country: Extract country from the "Sold To" section
- ship_to_name: Extract company name from the "Ship To" section
- ship_to_city: Extract city from the "Ship To" section
- ship_to_country: Extract country from the "Ship To" section
- product_number: Extract all material numbers from the "Mat.No" column, as a comma-separated list
- product_name: Extract all descriptions from the "Description" column, as a comma-separated list
- qty_ordered: Extract all quantities from the "Qty ordered" column, as a comma-separated list
- product_uom: Extract  units of measure from the "UoM" or U/M column.
- qty_picked: Extract all quantities from the "QtyPicked" column, as a comma-separated list

Please ensure your extraction is accurate and follows the exact schema structure provided. For array fields (product information), compile multiple entries as comma-separated values.
"""

# Fallback Freight Invoice Template (used when S3 templates are not available)
FALLBACK_FREIGHT_INVOICE_TEMPLATE = """
You are an expert document analyzer specializing in extracting structured information from freight invoices.

Your task is to analyze the provided freight invoice text and extract key information in a structured JSON format.

## Instructions:
1. Carefully analyze the freight invoice text below
2. Extract all relevant information including invoice details, charges, shipping information, etc.
3. Provide confidence scores (0.0 to 1.0) for each extracted field
4. Include explanations for your extractions
5. Return the data in the exact JSON schema format specified

## Freight Invoice Text:
{freight_invoice_text}
## Air Way Bill (AWB) Text:
{air_way_bill_text}
## Commercial Invoice Text:
{commercial_invoices_text}

## Required JSON Schema:
The response must follow this exact structure with a 'data' array containing invoice objects.
"""

# Document Template Mapping
HARDCODED_DOCUMENT_TEMPLATES = {
    "commercial_invoice": COMMERCIAL_INVOICE_PROMPT_TEMPLATE,
    "commercial_invoices": COMMERCIAL_INVOICE_PROMPT_TEMPLATE,
    "air_waybill": AIRWAY_BILL_PROMPT_TEMPLATE,
    "air_waybills": AIRWAY_BILL_PROMPT_TEMPLATE,
    "airway_bill": AIRWAY_BILL_PROMPT_TEMPLATE,  # Add this line
    "airway_bills": AIRWAY_BILL_PROMPT_TEMPLATE,
    "sea_waybills": AIRWAY_BILL_PROMPT_TEMPLATE,
    "packing_list": PACKING_LIST_PROMPT_TEMPLATE,
    "packing_lists": PACKING_LIST_PROMPT_TEMPLATE,
    "delivery_note": DELIVERY_NOTE_PROMPT_TEMPLATE,
    "delivery_notes": DELIVERY_NOTE_PROMPT_TEMPLATE
}

def get_document_schema(document_type: str) -> dict:
    """
    Get the appropriate JSON schema for a document type.
    
    Args:
        document_type: The document type (e.g., 'freight_invoice', 'commercial_invoice', etc.)
    
    Returns:
        dict: The JSON schema for the document type
    """
    # Normalize document type
    doc_type_key = document_type.lower().replace(' ', '_').replace('-', '_')
    
    # Try exact match first
    if doc_type_key in DOCUMENT_SCHEMAS:
        logger.info(f"Using schema for document type: {doc_type_key}")
        return DOCUMENT_SCHEMAS[doc_type_key]
    
    # Try alternative keys
    alternative_keys = [
        doc_type_key.replace('_invoice', ''),
        doc_type_key.replace('_waybill', ''),
        doc_type_key.replace('bill_of_lading', 'air_waybill'),  # Map bill_of_lading to air_waybill (consistent with template mapping)
        doc_type_key.replace('packing_list', 'packing'),
        doc_type_key.replace('freight', 'invoice'),
        doc_type_key.replace('commercial', 'invoice'),
        'air_waybill',  # Fallback for bill_of_lading and similar document types
        'airway_bill'   # Additional fallback variant
    ]
    
    for alt_key in alternative_keys:
        if alt_key in DOCUMENT_SCHEMAS:
            logger.info(f"Using schema for document type: {doc_type_key} (mapped to: {alt_key})")
            return DOCUMENT_SCHEMAS[alt_key]
    
    # Default to freight invoice schema if no match found
    logger.warning(f"No specific schema found for document type: {document_type}, using freight invoice schema")
    return FREIGHT_INVOICE_SCHEMA

# ---------- CARRIER MATCHING FUNCTIONS ----------
# ---------- EMAIL-CARRIER AUTHORIZATION CHECK ----------
def check_sender_carrier_authorization(sender_email: str, identified_carrier: str) -> tuple[bool, str]:
    """
    Check if the sender email is authorized to send documents for the identified carrier.
    
    Args:
        sender_email: The email address of the sender
        identified_carrier: The carrier identified from the document (KWE, MAGNO, etc.)
        
    Returns:
        tuple: (is_authorized: bool, error_message: str)
    """
    if not sender_email or not identified_carrier:
        return True, ""  # Skip check if email or carrier not provided
    
    sender_email_lower = sender_email.lower().strip()
    carrier_upper = identified_carrier.upper().strip()
    
    # Define carrier-to-email-domain mapping
    carrier_domains = {
        'KWE': ["sadhanand.moorthy@pando.ai","PPearson@ITS.JNJ.com","jeeva@pando.ai"],
        'MAGNO': ["shashank.tiwari@pando.ai", "CString3@ITS.JNJ.com","jeeva@pando.ai"]
        
    }
    
    # Check if sender email matches the carrier authorized emails
    authorized_domains = carrier_domains.get(carrier_upper, [])
    
    if not authorized_domains:
        # Unknown carrier - allow for now (could be other carriers)
        logger.info(f"Unknown carrier '{identified_carrier}' - skipping authorization check")
        return True, ""
    
    # Convert authorized domains to lowercase for case-insensitive comparison
    authorized_domains_lower = [domain.lower().strip() for domain in authorized_domains]
    
    # Check if sender email matches any authorized domain for this carrier
    # Use exact match instead of substring match for better security
    is_authorized = sender_email_lower in authorized_domains_lower
    
    if not is_authorized:
        error_msg = f"Sender {sender_email} is not authorized to send {carrier_upper} documents. Authorized domains: {', '.join(authorized_domains)}"
        logger.warning(error_msg)
        return False, error_msg
    
    logger.info(f"✅ Authorization check passed: {sender_email} is authorized for {carrier_upper}")
    return True, ""

# ---------- CARRIER CLASSIFICATION ----------
class CarrierClassifier:
    def __init__(self, force_refresh=False):
        """Initialize the carrier classifier with Claude model."""
        logger.info("Initializing CarrierClassifier")
        self.carrier_templates = load_prompt_template_from_s3(S3_BUCKET, S3_PREFIX)
        if not self.carrier_templates:
            logger.warning("S3 prompt templates not available, using fallback carrier list")
            # Do NOT fallback to KNOWN_CARRIERS; require S3-derived carriers only
            self.supported_carriers = []
            logger.info("No supported carriers available from S3; classification will be skipped")
        else:
            self.supported_carriers = list(self.carrier_templates.keys())
            logger.info(f"Loaded {len(self.supported_carriers)} supported carriers from S3")
        
        logger.info(f"Initialized carrier classifier with {len(self.supported_carriers)} supported carriers")
        logger.info(f"Supported carriers: {self.supported_carriers}")
        
    def classify_document(self, text: str) -> Optional[str]:
        """Classify a document using Claude to determine the carrier."""
        try:
            # If we have no supported carriers, skip classification
            if not self.supported_carriers:
                logger.warning("No supported carriers available. Skipping carrier classification.")
                return None
            logger.info("Starting document classification with Claude")
            logger.info(f"Full text length: {len(text)} characters")
            # Use first 20000 characters for classification to ensure we capture carrier names
            # Carrier names often appear in headers/footers which might be later in the text
            classification_sample = text[:20000] if len(text) > 20000 else text
            logger.info(f"Classification sample length: {len(classification_sample)} characters")
            logger.info(f"CLASSIFICATION SAMPLE TEXT (first 500 chars): {classification_sample[:500]}")
            # Check if key carrier terms are present in the text
            text_upper = classification_sample.upper()
            if 'KINTETSU' in text_upper or 'KWE' in text_upper:
                logger.info("✅ Found KINTETSU/KWE in text")
            if 'MAGNO' in text_upper:
                logger.info("✅ Found MAGNO in text")
            if 'KINTETSU' not in text_upper and 'KWE' not in text_upper and 'MAGNO' not in text_upper:
                logger.warning("⚠️ No carrier keywords found in classification sample")

            prompt = f"""You are an expert system for identifying freight shipping carriers from invoice documents.

            Your task is to read the provided invoice text and classify the correct carrier.

            Classification Rules:
            1. You must respond ONLY with one of these EXACT carrier names:
            - MAGNO
            - KWE
            - UNKNOWN

            2. Classification criteria (case-insensitive):
            - If the text contains "MAGNO" or "Magno International, LP" or "Magno International", classify as MAGNO.
            - If the text contains "KWE" or "Kintetsu" or "Kintetsu World Express" or "Kintetsu World Express (U.S.A), Inc." or "KINTETSU WORLD EXPRESS", classify as KWE.
            - Note: "Kintetsu" may appear anywhere in the document (headers, footers, company info, etc.)

            3. If none of the above keywords are found, classify as UNKNOWN.

            4. Your output must contain ONLY one of these words:
            - MAGNO
            - KWE
            - UNKNOWN

            Do not include any explanation, punctuation, or additional text.

            Here is the invoice text:
            {classification_sample}
            """
        #     prompt = f"""You are an expert at identifying shipping carriers from invoice documents.
        #     Based on the following text from an invoice, identify which carrier it belongs to.
        #     Only respond with one of these exact carrier names if you are confident, otherwise respond with "GENERIC":
        # {', '.join(self.supported_carriers)}
        #     Here is the invoice text:
        # {classification_sample}
        #     Respond with ONLY the carrier name or "GENERIC".
        #     """
           

            logger.info("Sending classification request to Claude")
            #logger.info(f"Classification prompt prepared (length: {len(prompt)} chars)")
            
            model_id = BEDROCK_MODEL_ID if 'BEDROCK_MODEL_ID' in globals() and BEDROCK_MODEL_ID else os.environ.get('BEDROCK_MODEL_ID')
            #logger.info(f"Using model ID: {model_id}")
            
            start_time = time.time()
            
            # Define the LLM call function for retry wrapper
            def make_classification_call():
                return bedrock.invoke_model(
                    modelId=model_id,
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 10,  # Only need a short response
                        "temperature": 0.1,  # Lower temperature for more deterministic results
                        "messages": [{"role": "user", "content": prompt}]
                    })
                )
            
            # Use retry wrapper for the LLM call
            response = retry_llm_call(make_classification_call)
            
            end_time = time.time()
            #logger.info(f"Classification response received in {end_time - start_time:.2f} seconds")
            
            response_body = json.loads(response['body'].read())
            #logger.info(f"Claude response keys: {list(response_body.keys())}")
            logger.info(f"FULL CLAUDE CLASSIFICATION RESPONSE: {json.dumps(response_body, indent=2)}")
            
            if 'content' in response_body and isinstance(response_body['content'], list):
                carrier = response_body['content'][0]['text'].strip()
                logger.info(f"Claude classification result: '{carrier}'")
                
                if carrier in self.supported_carriers:
                    logger.info(f"Document classified as {carrier}")
                    return carrier
                else:
                    logger.warning(f"Claude returned unknown carrier: '{carrier}'")
                    logger.warning(f"Expected one of (case-sensitive): {self.supported_carriers}")
                    return None
            else:
                logger.error("Unexpected response format from Claude")
                logger.error(f"Response body: {response_body}")
                return None
                
        except Exception as e:
            logger.error(f"Error classifying document with Claude: {str(e)}", exc_info=True)
            raise

def match_carrier_name(llm_identified_carrier: str, supported_carriers: List[str]) -> Optional[str]:
    """
    Match LLM-identified carrier against supported carriers list from S3 using case-sensitive equality.
    
    Args:
        llm_identified_carrier: Carrier name identified by LLM
        supported_carriers: Carrier names supported (as loaded from S3 templates)
        
    Returns:
        Optional[str]: Exact matched carrier name or None if no case-sensitive match
    """
    if not llm_identified_carrier or not supported_carriers:
        return None
    candidate = llm_identified_carrier.strip()
    if candidate in supported_carriers:
        logger.info(f"Case-sensitive carrier match found: {candidate}")
        return candidate
    logger.warning(f"No case-sensitive carrier match for: '{llm_identified_carrier}'. Supported: {supported_carriers}")
    return None

# ---------- PDF PAGE EXTRACTION CLASS (from pdf_page_extractor.py) ----------
class LLMBasedPDFClusterer:
    def __init__(self, pdf_path, output_dir="llm_clusters", max_workers=5):
        """
        Initialize PDF clusterer using LLM for layout and content analysis
        
        Args:
            pdf_path: Path to the PDF file
            output_dir: Directory to save clustered PDFs
            max_workers: Maximum number of threads for parallel processing
        """
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.max_workers = max_workers
        self.doc = fitz.open(pdf_path)
        self.n_pages = len(self.doc)
        
        # AWS clients (one per thread)
        self._thread_local = threading.local()
        
        # Create output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        logger.info(f"Processing PDF with {self.n_pages} pages using LLM analysis")
        
        # Store data
        self.page_images = []  # Store page images
        self.page_analyses = {}  # Store LLM analyses
        self.clusters = {}  # Store final clusters
        self.blank_pages = []  # Track blank pages
        
        # Lock for thread safety
        self.lock = threading.Lock()

    @property
    def bedrock(self):
        """Get thread-local Bedrock client"""
        if not hasattr(self._thread_local, 'bedrock'):
            self._thread_local.bedrock = boto3.client('bedrock-runtime')
        return self._thread_local.bedrock

    def convert_pages_to_images(self, dpi=150):  # Reduced DPI from 200 to 150
        """Convert all PDF pages to images with size limits for Claude API"""
        logger.info(f"Converting {self.n_pages} PDF pages to images...")
        
        for page_num in range(self.n_pages):
            logger.info(f"Converting page {page_num+1}/{self.n_pages}")
            page = self.doc[page_num]
            
            # Start with a lower DPI and adjust if needed
            current_dpi = dpi
            max_attempts = 3
            attempts = 0
            
            while attempts < max_attempts:
                # Convert page to image at specified DPI
                pix = page.get_pixmap(matrix=fitz.Matrix(current_dpi/72, current_dpi/72))
                
                # Check dimensions before proceeding
                if pix.width > 7900 or pix.height > 7900:  # Buffer below 8000 limit
                    current_dpi *= 0.75  # Reduce DPI by 25%
                    attempts += 1
                    logger.info(f"  Reducing DPI to {current_dpi:.1f} (dimensions too large)")
                    continue
                    
                # Convert to PIL Image
                img_data = pix.tobytes()
                img = Image.open(io.BytesIO(img_data))
                
                # Check if page is blank
                is_blank = self._is_blank_page(img)
                if is_blank:
                    logger.info(f"Page {page_num+1} appears to be blank")
                    self.blank_pages.append(page_num)
                
                # Store image and convert to base64 for API
                img_byte_arr = io.BytesIO()
                
                # Use progressive JPEG with quality adjustment to reduce size
                img.save(img_byte_arr, format='JPEG', quality=75, optimize=True, progressive=True)
                img_bytes = img_byte_arr.getvalue()
                
                # Check file size
                if len(img_bytes) > 5 * 1024 * 1024:  # 5MB
                    if attempts < max_attempts - 1:
                        current_dpi *= 0.75  # Reduce DPI further
                        attempts += 1
                        logger.info(f"  Reducing DPI to {current_dpi:.1f} (file too large)")
                        continue
                    else:
                        # On last attempt, reduce quality more aggressively
                        img_byte_arr = io.BytesIO()
                        img.save(img_byte_arr, format='JPEG', quality=60, optimize=True)
                        img_bytes = img_byte_arr.getvalue()
                
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                self.page_images.append({
                    'page_num': page_num,
                    'image': img,
                    'b64_image': img_b64,
                    'is_blank': is_blank
                })
                
                # Successfully processed the image, break out of the loop
                break
                
            if attempts == max_attempts:
                logger.info(f"  Warning: Could not reduce page {page_num+1} to acceptable size after {max_attempts} attempts")
                
                # Create a thumbnail instead as last resort
                img.thumbnail((4000, 4000), Image.LANCZOS)
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='JPEG', quality=60, optimize=True)
                img_bytes = img_byte_arr.getvalue()
                img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                
                self.page_images.append({
                    'page_num': page_num,
                    'image': img,
                    'b64_image': img_b64,
                    'is_blank': is_blank,
                    'resized': True
                })
    
    def _is_blank_page(self, img, threshold=0.99):
        """
        Check if a page is blank by analyzing pixel values
    
    Args:
            img: PIL Image object
            threshold: Threshold for determining blankness (higher = more strict)
        
    Returns:
            Boolean indicating if page is blank
        """
        # Convert to grayscale and get pixel data
        gray_img = img.convert('L')
        pixels = np.array(gray_img)
        
        # Calculate whiteness ratio (pixels close to white)
        white_pixels = np.sum(pixels > 240)
        total_pixels = pixels.size
        whiteness_ratio = white_pixels / total_pixels
        
        return whiteness_ratio > threshold
    
    def analyze_pages_with_llm(self):
        """Analyze all pages with LLM to extract layout and content features"""
        logger.info("Analyzing pages with LLM...")
        
        # Process pages in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all page processing tasks (skip blank pages)
            futures = []
            for page_data in self.page_images:
                if not page_data['is_blank']:
                    futures.append(executor.submit(self._analyze_page, page_data))
            
            # Wait for all tasks to complete
            for future in futures:
                future.result()
    
    def _analyze_page(self, page_data):
        """Analyze a single page with LLM"""
        page_num = page_data['page_num']
        img_b64 = page_data['b64_image']
        
        try:
            logger.info(f"Analyzing page {page_num+1}/{self.n_pages} with LLM")
            
            # Call Claude with image, with retry logic for size issues
            max_attempts = 3
            attempts = 0
            analysis = None
            
            while attempts < max_attempts and analysis is None:
                try:
                    # Create prompt for LLM
                    prompt = self._create_page_analysis_prompt(page_num + 1, img_b64)
                    analysis = self._call_claude_with_image(prompt, img_b64)
                    
                except Exception as e:
                    if "image exceeds 5 MB maximum" in str(e) or "dimensions exceed max allowed size" in str(e):
                        attempts += 1
                        if attempts < max_attempts:
                            # Reduce image quality and try again
                            logger.info(f"  Reducing image quality and retrying ({attempts}/{max_attempts})")
                            img = page_data['image']
                            img_byte_arr = io.BytesIO()
                            quality = 70 - (attempts * 15)  # Reduce quality each attempt
                            
                            # Calculate scale factor to reduce dimensions
                            scale = 0.85 - (attempts * 0.15)  # Reduce size each attempt
                            new_width = int(img.width * scale)
                            new_height = int(img.height * scale)
                            resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                            
                            resized_img.save(img_byte_arr, format='JPEG', quality=quality, optimize=True)
                            img_bytes = img_byte_arr.getvalue()
                            img_b64 = base64.b64encode(img_bytes).decode('utf-8')
                            continue
                        else:
                            # Try OCR-based fallback
                            logger.info(f"  Trying OCR fallback for page {page_num+1}")
                            analysis = self._ocr_fallback(page_data)
                            if not analysis:
                                logger.info(f"  Attempting to split page {page_num+1} for analysis")
                            analysis = self._split_and_analyze_large_page(page_data)
                    if not analysis:
                        raise
            
            if not analysis:
                raise Exception("Failed to analyze page after multiple attempts")
            
            # Parse the results
            parsed_analysis = self._parse_page_analysis(analysis, page_num)
            
            # Store analysis results
            with self.lock:
                self.page_analyses[page_num] = parsed_analysis
                
        except Exception as e:
            logger.error(f"Error analyzing page {page_num+1}: {str(e)}")
            
            # Store basic info for failed pages
            with self.lock:
                self.page_analyses[page_num] = {
                    "page_num": page_num,
                    "document_type": "Unknown Document Type",
                    "error": str(e),
                    "layout_elements": [],
                    "key_identifiers": {},
                    "continuation_markers": [],
                    "header_content": "",
                    "footer_content": "",
                    "document_group": "Error Processing"
                }
    
    def _ocr_fallback(self, page_data):
        """Use OCR as fallback for pages that fail with Claude"""
        try:
            # This is a placeholder - you could implement OCR here if needed
            logger.info(f"OCR fallback not implemented for page {page_data['page_num']+1}")
            return None
        except Exception as e:
            logger.error(f"OCR fallback failed: {str(e)}")
            return None
    
    def _split_and_analyze_large_page(self, page_data):
        """Split large page and analyze parts separately"""
        try:
            # This is a placeholder - you could implement page splitting here if needed
            logger.info(f"Page splitting not implemented for page {page_data['page_num']+1}")
            return None
        except Exception as e:
            logger.error(f"Page splitting failed: {str(e)}")
            return None
    
    def _create_page_analysis_prompt(self, page_num, img_b64):
        """Create prompt for page analysis"""
        return """
You are an expert document analyzer specialized in identifying document types and structural elements.

Task:
Analyze the provided document image carefully and generate a structured analysis.

Focus on:
1. Document type identification (freight invoice, packing list, delivery note, commercial invoice, airway bill, sea way bill, etc.)
2. The most common document types are:
   - Freight Invoice (shipping charges details)
   - Commercial Invoice (product details)
   - Packing List (products being shipped with quantities)
   - Air Waybill (air freight transport document)
   - Sea Waybill (sea freight transport document)
   - Bill of Lading (legal document between shipper and carrier)
   
If there is any other document type present and you can identify it, name it under that type name else put it under "Unknown Document Type".

Note:
 1. Invoice is something related to shipping charges details also it has mention like Original Invoice,Corrected Invoice, etc. whereas Commercial invoice is related to product details.
 2. Be very specific about document types - don't use generic terms like "Shipping Documentation" when you can identify it as "Air Waybill" or "Bill of Lading"
 3. The waybill documents has mention like Air Waybill,Sea Waybill, Bill of Lading, etc. clearly mention the document type.
 4. For packing lists, identify if they are part of the same shipment by looking at reference numbers, dates, and other identifiers
 5. There is differnce between packing list and packing slip. So don't confuse them and combine them together.
 6. There is difference between freight invoice and waybill documents. The waybill documents has mention like Air Waybill,Sea Waybill, Bill of Lading, etc. whereas freight invoice has mention like Original Invoice, Freight Invoice, etc.

2. Layout analysis (headers, footers, tables, form fields, etc.)
3. Key identifiers (invoice number, dates, company names, etc.)
4. Page continuity indicators (page numbers, "continued from", etc.)

For each document page, identify:
- Document type (be specific)
- Key structural elements (headers, tables, form fields)
- Specific identifiers (document numbers, dates, reference numbers, shipment IDs)
- Page sequence indicators (page numbers, continuation markers)
- Header/footer content that might help in identifying related pages

This is page {page_num} of a multi-page PDF. Your analysis will be used to cluster similar pages together.

Return your analysis as a structured JSON object with:
- "document_type": The detected document type (be specific)
   - If any document type related to freight invoice is present, then return the document type as "Freight Invoice".
   - If any document type related to commercial invoice is present, then return the document type as "Commercial Invoice".
   - If any document type related to packing list is present, then return the document type as "Packing List".
   - If any document type related to air waybill is present, then return the document type as "Air Waybill".
   - If any document type related to airway bill is present, then return the document type as "Air Waybill". 
   - If any document type related to bill of lading is present, then return the document type as "Bill of Lading".
   - If any document type related to sea waybill is present, then return the document type as "Sea Waybill".
   - If any other document type is present, then return the document type as "Unknown Document Type".
- "layout_elements": Key layout elements found
- "key_identifiers": Specific identifiers in the document (reference numbers, dates, etc.)
- "continuation_markers": Any indication this is part of a multi-page document
- "header_content": Text found in document headers
- "footer_content": Text found in document footers
- "document_group": A category that identifies which document group this belongs to (e.g., "Commercial Invoice - ABC123", "Packing List - XYZ789")

Note: You should return the document type as the actual document type name given above, not the generic term like "Air Waybill Terms and Conditions" or "Air Waybill pages" or "Commercial Invoice documents" like that.

Your analysis will be used to group similar pages together based on layout and content patterns.
""".format(page_num=page_num)
    
    def _call_claude_with_image(self, prompt, img_b64):
        """Call Claude with image for analysis"""
        try:
            # Configure input for Claude 3 Sonnet
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/jpeg",
                                    "data": img_b64
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt+"After analyzing the document, check once again thoroughly and confirm the document type and return the correct document type in the response."
                            }
                        ]
                    }
                ]
            }
            
            # Make the API call with retry logic
            def make_claude_image_call():
                return self.bedrock.invoke_model(
                    modelId=BEDROCK_MODEL_ID,
                    body=json.dumps(payload)
                )
            
            response = retry_llm_call(make_claude_image_call)
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            return response_body["content"][0]["text"]
        
        except Exception as e:
            logger.error(f"Error calling Claude model: {str(e)}")
            raise
    
    def _call_bedrock_llm(self, prompt):
        """Call Bedrock LLM with text-only prompt"""
        try:
            # Configure input for Claude
            payload = {
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 4096,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
            
            # Make the API call with retry logic
            def make_bedrock_llm_call():
                return self.bedrock.invoke_model(
                    modelId=BEDROCK_MODEL_ID,
                    body=json.dumps(payload)
                )
            
            response = retry_llm_call(make_bedrock_llm_call)
            
            # Parse response
            response_body = json.loads(response.get('body').read())
            return response_body["content"][0]["text"]
        
        except Exception as e:
            logger.error(f"Error calling Bedrock model: {str(e)}")
            return None

    def _parse_page_analysis(self, analysis_text, page_num):
        """Parse LLM analysis into structured data"""
        if not analysis_text:
            return {
                "page_num": page_num,
                "document_type": "Unknown",
                "error": "No analysis returned"
            }
        
        try:
            # Try to find JSON in the response
            json_pattern = re.compile(r'```(?:json)?(.*?)```', re.DOTALL)
            matches = json_pattern.findall(analysis_text)
            
            if matches:
                # Use the first JSON block found
                json_str = matches[0].strip()
                result = json.loads(json_str)
            else:
                # If no code blocks, try to parse the entire response as JSON
                # First, look for anything that looks like a JSON object
                json_obj_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                obj_matches = json_obj_pattern.findall(analysis_text)
                
                if obj_matches:
                    result = json.loads(obj_matches[0])
                else:
                    # Fall back to a structured extraction
                    result = self._extract_structured_info(analysis_text)
            
            # Add page number to result
            result["page_num"] = page_num
            return result
                
        except Exception as e:
            logger.error(f"Error parsing LLM response for page {page_num+1}: {str(e)}")
            # Create a basic structure with the raw response
            return {
                "page_num": page_num,
                "document_type": self._extract_document_type(analysis_text),
                "raw_analysis": analysis_text[:1000] + "..." if len(analysis_text) > 1000 else analysis_text,
                "parsing_error": True
            }
    
    def _extract_document_type(self, text):
        """Extract document type from text response"""
        # Look for common document type patterns
        patterns = [
            r"document[_ ]type[\"':,\s]+([A-Za-z ]+)",
            r"appears to be an? ([A-Za-z ]+)",
            r"This is an? ([A-Za-z ]+)",
            r"identified as an? ([A-Za-z ]+)",
            r"this document is an? ([A-Za-z ]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                doc_type = match.group(1).strip()
                # Clean up common endings
                doc_type = re.sub(r"\..*\\\$", "", doc_type)
                doc_type = re.sub(r"\s+document\\\$", "", doc_type, flags=re.IGNORECASE)
                return doc_type
        
        return "Unknown"
    
    def _extract_structured_info(self, text):
        """Extract structured info from text when JSON parsing fails"""
        result = {
            "document_type": self._extract_document_type(text),
            "layout_elements": [],
            "key_identifiers": {},
            "continuation_markers": [],
            "header_content": "",
            "footer_content": "",
            "document_group": ""
        }
        
        # Extract header content
        header_match = re.search(r"header[_ ]content[\"':,\s]+(.*?)(?=\n\n|\n[A-Za-z])", text, re.IGNORECASE | re.DOTALL)
        if header_match:
            result["header_content"] = header_match.group(1).strip()
            
        # Extract footer content
        footer_match = re.search(r"footer[_ ]content[\"':,\s]+(.*?)(?=\n\n|\n[A-Za-z])", text, re.IGNORECASE | re.DOTALL)
        if footer_match:
            result["footer_content"] = footer_match.group(1).strip()
        
        # Extract layout elements
        layout_section = re.search(r"layout[_ ]elements[\"':,\s]+(.*?)(?=\n\n|\n[A-Za-z])", text, re.IGNORECASE | re.DOTALL)
        if layout_section:
            elements = re.findall(r"[•\-\*]\s*(.*?)(?=\n[•\-\*]|\n\n|\Z)", layout_section.group(1), re.DOTALL)
            result["layout_elements"] = [elem.strip() for elem in elements if elem.strip()]
        
        # Extract key identifiers
        id_patterns = [
            (r"invoice[_ ]number[\"':,\s]+([A-Za-z0-9\-]+)", "invoice_number"),
            (r"order[_ ]number[\"':,\s]+([A-Za-z0-9\-]+)", "order_number"),
            (r"date[\"':,\s]+([A-Za-z0-9\-/]+)", "date"),
            (r"company[_ ]name[\"':,\s]+([A-Za-z0-9\- ]+)", "company_name")
        ]
        
        for pattern, key in id_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result["key_identifiers"][key] = match.group(1).strip()
        
        # Extract document group
        group_match = re.search(r"document[_ ]group[\"':,\s]+([A-Za-z0-9\- ]+)", text, re.IGNORECASE)
        if group_match:
            result["document_group"] = group_match.group(1).strip()
        
        return result
    
    def determine_clusters(self):
        """Determine page clusters based on LLM analyses"""
        logger.info("Determining page clusters based on LLM analyses...")
        
        if not self.page_analyses:
            logger.info("No page analyses available. Run analyze_pages_with_llm() first.")
            return {}
        
        # Create a consolidated prompt with all page analyses
        clustering_prompt = self._create_clustering_prompt()
        
        # Call LLM for clustering decision
        clustering_result = self._call_bedrock_llm(clustering_prompt)
        
        # Parse clustering results
        self.clusters = self._parse_clustering_result(clustering_result)
        
        return self.clusters
    
    def _create_clustering_prompt(self):
        """Create prompt for clustering decision"""
        # Convert page analyses to a consistent format
        page_summaries = []
        
        for page_num in sorted(self.page_analyses.keys()):
            analysis = self.page_analyses[page_num]
            
            # Extract key information for clustering
            doc_type = analysis.get("document_type", "Unknown")
            layout_elements = analysis.get("layout_elements", [])
            key_identifiers = analysis.get("key_identifiers", {})
            continuation_markers = analysis.get("continuation_markers", [])
            header_content = analysis.get("header_content", "")
            footer_content = analysis.get("footer_content", "")
            document_group = analysis.get("document_group", "")
            
            # Create a summary for this page
            summary = {
                "page_number": page_num + 1,
                "document_type": doc_type,
                "key_elements": layout_elements,
                "identifiers": key_identifiers,
                "header": header_content,
                "footer": footer_content,
                "continuation_markers": continuation_markers,
                "document_group": document_group
            }
            
            page_summaries.append(summary)
        
        # Create the prompt
        prompt = """
        You are an expert document clustering system tasked with grouping related pages together.

        I'll provide analyses of individual pages from a multi-page PDF. Your job is to determine which pages should be grouped together as part of the same document or document type.

        IMPORTANT REQUIREMENTS:
        Note: 
        1. Do not combine any pages that are not similar to each other.(Eg: Freight Invoice and Commercial Invoice).
        2. Freight invoice has details of shipping charges and it has mention like Original Invoice, Freight Invoice, etc. whereas Commercial invoice has details of product details.
        3. Do not confuse Freight Invoice and Commercial Invoice.
        4. There is differnce between packing list and packing slip. So don't confuse them and combine them together.
        1. DO NOT split similar document types into multiple clusters. For example, all "Commercial Invoice" pages should be in ONE cluster, all "Packing List" pages should be in ONE cluster.

        2. Group pages by document type AND by specific identifiers (like invoice numbers, reference numbers, etc.)

        3. Consider page continuity markers (page numbers, "continued from", etc.)

        4. Use header/footer content to identify related pages

        5. Return results as JSON with cluster information

        Page Analyses:
        """ + json.dumps(page_summaries, indent=2) + """

        Return a JSON object with this structure:
        {
            "clusters": {
                "cluster_1": {
                    "pages": [1, 2, 3],
                    "document_type": "Commercial Invoice",
                    "reason": "Same invoice number and document type"
                },
                "cluster_2": {
                    "pages": [4, 5],
                    "document_type": "Packing List", 
                    "reason": "Same reference number and document type"
                }
            }
        }

        IMPORTANT: Only group pages that are clearly part of the same document based on document type and identifiers.
        """
        
        return prompt
    
    def _parse_clustering_result(self, clustering_text):
        """Parse clustering result from LLM"""
        try:
            # Try to find JSON in the response
            json_pattern = re.compile(r'```(?:json)?(.*?)```', re.DOTALL)
            matches = json_pattern.findall(clustering_text)
            
            if matches:
                json_str = matches[0].strip()
                result = json.loads(json_str)
            else:
                # Try to parse the entire response as JSON
                json_obj_pattern = re.compile(r'(\{.*\})', re.DOTALL)
                obj_matches = json_obj_pattern.findall(clustering_text)
                
                if obj_matches:
                    result = json.loads(obj_matches[0])
                else:
                    # Fall back to simple document type grouping
                    return self._fallback_clustering()
            
            # Extract clusters from result
            clusters = result.get("clusters", {})
            
            # Convert to our expected format and combine clusters of the same document type
            doc_type_groups = {}
            for cluster_id, cluster_info in clusters.items():
                pages = cluster_info.get("pages", [])
                doc_type = cluster_info.get("document_type", "Unknown")
                reason = cluster_info.get("reason", "")
                
                # Group pages by document type
                if doc_type not in doc_type_groups:
                    doc_type_groups[doc_type] = {
                        "pages": [],
                        "reasons": []
                    }
                
                doc_type_groups[doc_type]["pages"].extend(pages)
                doc_type_groups[doc_type]["reasons"].append(reason)
            
            # Create final clusters with combined pages
            formatted_clusters = {}
            for i, (doc_type, group_info) in enumerate(doc_type_groups.items()):
                # Sort and remove duplicates from pages
                unique_pages = sorted(list(set(group_info["pages"])))
                combined_reason = f"Combined clustering: {len(unique_pages)} pages of type '{doc_type}'"
                
                formatted_clusters[str(i + 1)] = {
                    "pages": unique_pages,
                    "document_type": doc_type,
                    "reason": combined_reason
                }
            
            return formatted_clusters
            
        except Exception as e:
            logger.error(f"Error parsing clustering result: {str(e)}")
            return self._fallback_clustering()
    
    def _fallback_clustering(self):
        """Fallback clustering based on document type only"""
        logger.info("Using fallback clustering based on document type")
        
        # Group pages by document type
        doc_type_groups = {}
        for page_num, analysis in self.page_analyses.items():
            doc_type = analysis.get("document_type", "Unknown")
            if doc_type not in doc_type_groups:
                doc_type_groups[doc_type] = []
            doc_type_groups[doc_type].append(page_num + 1)  # Convert to 1-based
        
        # Create clusters
        clusters = {}
        for i, (doc_type, pages) in enumerate(doc_type_groups.items()):
            clusters[str(i)] = {
                "pages": pages,
                "document_type": doc_type,
                "reason": f"Fallback clustering: grouped {len(pages)} pages of type '{doc_type}'"
            }
        
        return clusters
    
    def save_clustered_pdfs(self, s3_bucket=None, main_pdf_name=None):
        """Save clustered pages as separate PDFs to both local and S3"""
        logger.info("Saving clustered PDFs...")
        
        if not self.clusters:
            logger.info("No clusters available. Run determine_clusters() first.")
            return
        
        saved_clusters = {}
        
        # Save clusters
        for cluster_id, cluster_info in self.clusters.items():
            page_indices = [p - 1 for p in cluster_info.get("pages", [])]  # Convert to 0-based indexing
            doc_type = cluster_info.get("document_type", "Unknown")
            
            if not page_indices:
                continue
                
            # Sort pages by original order
            page_indices.sort()
                
            # Sanitize the document type for filename
            safe_doc_type = re.sub(r'[\\/*?:"<>|]', "_", doc_type)  # Replace invalid filename chars
                
            # Create a descriptive filename
            local_output_path = os.path.join(
                self.output_dir, 
                f"cluster_{cluster_id}_{safe_doc_type}.pdf"
            )
            
            # Create a new PDF for this cluster
            new_doc = fitz.open()
            
            # Add pages from original PDF
            for page_idx in page_indices:
                new_doc.insert_pdf(self.doc, from_page=page_idx, to_page=page_idx)
                
            # Save the clustered PDF locally
            new_doc.save(local_output_path)
            new_doc.close()
                
            # Save to S3 if bucket and main PDF name are provided
            s3_path = None
            if s3_bucket and main_pdf_name:
                try:
                    # Create S3 key for clustered PDF
                    clean_main_name = os.path.splitext(main_pdf_name)[0]
                    clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
                    s3_key = f"clustered_pdfs/{clean_main_name}/cluster_{cluster_id}_{safe_doc_type}.pdf"
                    
                    # Upload to S3
                    with open(local_output_path, 'rb') as pdf_file:
                        s3.put_object(
                            Bucket=s3_bucket,
                            Key=s3_key,
                            Body=pdf_file,
                            ContentType='application/pdf'
                        )
                    
                    s3_path = f"s3://{s3_bucket}/{s3_key}"
                    logger.info(f"Uploaded cluster {cluster_id} to S3: {s3_path}")
                    
                except Exception as e:
                    logger.error(f"Failed to upload cluster {cluster_id} to S3: {str(e)}")
            
            saved_clusters[cluster_id] = {
                'local_path': local_output_path,
                's3_path': s3_path,
                'doc_type': doc_type,
                'pages': page_indices,
                'safe_doc_type': safe_doc_type
            }
                
            logger.info(f"Saved cluster {cluster_id} ({doc_type}) with {len(page_indices)} pages to {local_output_path}")
        
        return saved_clusters
    
    def generate_summary_report(self):
        """Generate a summary report of the clustering results"""
        logger.info("Generating summary report...")
        
        if not self.clusters:
            logger.info("No clusters available. Run determine_clusters() first.")
            return
        
        # Create a summary report
        report = {
            "filename": os.path.basename(self.pdf_path),
            "total_pages": self.n_pages,
            "cluster_count": len(self.clusters),
            "blank_pages": [p + 1 for p in self.blank_pages],  # Convert to 1-based indexing
            "clusters": {}
        }
        
        # Add cluster information
        for cluster_id, cluster_info in self.clusters.items():
            pages = cluster_info.get("pages", [])
            doc_type = cluster_info.get("document_type", "Unknown")
            reason = cluster_info.get("reason", "")
            
            report["clusters"][cluster_id] = {
                "document_type": doc_type,
                "pages": pages,
                "page_count": len(pages),
                "reason": reason
            }
        
        # Save the report
        report_path = os.path.join(self.output_dir, "clustering_report.json")
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Saved clustering report to {report_path}")
        
        return report
    
    def process(self, s3_bucket=None, main_pdf_name=None):
        """Run the full clustering process"""
        try:
            start_time = time.time()
            
            # Convert pages to images
            self.convert_pages_to_images(dpi=150)  # Reduced DPI to avoid size issues
            
            # Analyze pages with LLM
            self.analyze_pages_with_llm()
            
            # Determine clusters
            self.determine_clusters()
            
            # Save clustered PDFs (both locally and to S3)
            saved_clusters = self.save_clustered_pdfs(s3_bucket, main_pdf_name)
            
            # Generate summary report
            report = self.generate_summary_report()
            
            total_time = time.time() - start_time
            logger.info(f"\nProcessing complete! Total time: {total_time:.1f} seconds")
            
            # Print cluster summary
            logger.info("\nClustering Results:")
            for cluster_id, cluster_info in self.clusters.items():
                pages = cluster_info.get("pages", [])
                doc_type = cluster_info.get("document_type", "Unknown")
                logger.info(f"Cluster {cluster_id}: {doc_type} - {len(pages)} pages {pages}")
            
            # Print blank pages
            if self.blank_pages:
                blank_pages_1based = [p + 1 for p in self.blank_pages]
                logger.info(f"\nBlank pages detected (not saved): {blank_pages_1based}")
            
            # Add saved clusters info to report
            report['saved_clusters'] = saved_clusters
            
            return report
            
        except Exception as e:
            logger.error(f"Error during processing: {str(e)}")
            return None
    
    def get_carrier_from_freight_invoices(self):
        """Extract carrier information from freight invoice pages using LLM analysis"""
        freight_clusters = []
        
        # Find clusters that contain freight invoices
        for cluster_id, cluster_info in self.clusters.items():
            doc_type = cluster_info.get("document_type", "").lower()
            if "freight" in doc_type and "invoice" in doc_type:
                freight_clusters.append(cluster_info)
        
        if not freight_clusters:
            logger.warning("No freight invoice clusters found")
            return None
        
        # Get freight invoice pages for Textract processing
        freight_pages = []
        for cluster_info in freight_clusters:
            pages = cluster_info.get("pages", [])
            freight_pages.extend([p - 1 for p in pages])  # Convert to 0-based
        
        if not freight_pages:
            logger.warning("No freight invoice pages found")
            return None
        
        # Process freight invoices with Textract to get detailed text
        logger.info(f"Processing {len(freight_pages)} freight invoice pages for carrier identification")
        
        try:
            # Create a temporary PDF with freight invoice pages
            temp_pdf_path = os.path.join(self.output_dir, "temp_freight_invoice.pdf")
            temp_doc = fitz.open()
            
            for page_idx in freight_pages:
                temp_doc.insert_pdf(self.doc, from_page=page_idx, to_page=page_idx)
            
            temp_doc.save(temp_pdf_path)
            temp_doc.close()
            
            # Extract text using Textract
            freight_text = ""
            try:
                # Upload to S3 temporarily for Textract
                bucket, key = upload_local_pdf_to_s3(temp_pdf_path)
                forms_dict, tables_list, text_lines_list, page_count = extract_text_from_pdf(bucket, key)
                
                # Combine text from forms, tables, and lines - use correct key names
                freight_text = combine_textract_data({
                    'key_value_pairs': forms_dict,
                    'tables': tables_list,
                    'text_lines': text_lines_list,
                    'method': 'textract'
                })
                
                # Clean up S3
                try:
                    s3.delete_object(Bucket=bucket, Key=key)
                except Exception as e:
                    logger.warning(f"Could not clean up S3 object: {str(e)}")

                logger.info(f"Freight text extracted: {len(freight_text)} characters")
                logger.info(f"Freight text preview (first 500 chars): {freight_text[:500]}")
            except Exception as e:
                logger.warning(f"Textract failed, using fallback: {str(e)}")
                # Fallback to PyPDF2/pdfplumber
                freight_text, _ = extract_text_from_local_pdf_fallback(temp_pdf_path)
                logger.info(f"Fallback method extracted (freight_text): {freight_text}")
            
            # Clean up temporary file
            try:
                os.remove(temp_pdf_path)
            except Exception as e:
                logger.warning(f"Could not remove temp file: {str(e)}")
            
            if not freight_text.strip():
                logger.warning("No text extracted from freight invoices")
                return None
            
            # Use CarrierClassifier to identify carrier from freight invoice text
            try:
                logger.info("Using CarrierClassifier to identify carrier from freight invoice text")
                
                # Initialize carrier classifier
                carrier_classifier = CarrierClassifier()
                
                # Classify the document
                logger.info(f"Freight text: {freight_text}")
                identified_carrier = carrier_classifier.classify_document(freight_text)
                logger.info(f"CarrierClassifier identified carrier: '{identified_carrier}'")

                if identified_carrier:
                    logger.info(f"CarrierClassifier identified carrier: '{identified_carrier}'")
                    
                    # Match against known carriers (additional validation)
                    matched_carrier = match_carrier_name(identified_carrier, carrier_classifier.supported_carriers)
                    if matched_carrier:
                        logger.info(f"Carrier matched: {matched_carrier}")
                        
                        # Store Textract data for reuse in individual document processing
                        textract_data = {
                            'key_value_pairs': forms_dict,
                            'tables': tables_list,
                            'text_lines': text_lines_list,
                            'page_count': len(freight_pages),
                            'method': 'textract'
                        }
                        
                        # Store in the page_extractor for later reuse
                        self.freight_textract_data = textract_data
                        
                        # Return both carrier and Textract data
                        return {
                            'carrier': matched_carrier,
                            'textract_data': textract_data
                        }
                    else:
                        logger.warning(f"CarrierClassifier identified carrier '{identified_carrier}' not in known carriers list")
                        return None
                else:
                    logger.warning("CarrierClassifier could not identify carrier")
                    return None
                    
            except Exception as e:
                logger.error(f"Error using CarrierClassifier for carrier identification: {str(e)}")
                return None
                
        except Exception as e:
            logger.error(f"Error processing freight invoices for carrier identification: {str(e)}")
            return None
    
    def get_document_clusters_for_processing(self):
        """Convert clusters to the format expected by the processing function"""
        clusters = {
            'freight_invoices': [],
            'commercial_invoices': [],
            'bills_of_lading': [],
            'air_waybills': [],
            'other_documents': []
        }
        
        for cluster_id, cluster_info in self.clusters.items():
            pages = cluster_info.get("pages", [])
            doc_type = cluster_info.get("document_type", "Unknown").lower()
            
            # Convert page numbers to 0-based indices
            page_indices = [p - 1 for p in pages]
            
            if "freight" in doc_type and "invoice" in doc_type:
                clusters['freight_invoices'].extend(page_indices)
            elif "commercial" in doc_type and "invoice" in doc_type:
                clusters['commercial_invoices'].extend(page_indices)
            elif "bill of lading" in doc_type or "bol" in doc_type:
                clusters['bills_of_lading'].extend(page_indices)
            elif "air waybill" in doc_type or "airwaybill" in doc_type:
                clusters['air_waybills'].extend(page_indices)
            else:
                clusters['other_documents'].extend(page_indices)
        
        return clusters

# ---------- LOCAL PDF PROCESSING FUNCTIONS ----------
def upload_local_pdf_to_s3(file_path: str, bucket: str = None) -> tuple:
    """
    Upload a local PDF file to S3 temporarily for Textract processing.
    Returns: (bucket, key) for the uploaded file
    """
    try:
        if not bucket:
            bucket = OUTPUT_BUCKET
        
        # Generate a unique key for the temporary file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = os.path.basename(file_path)
        temp_key = f"temp_textract/{timestamp}_{file_name}"
        
        logger.info(f"Uploading local PDF to S3: s3://{bucket}/{temp_key}")
        
        # Upload the file
        with open(file_path, 'rb') as file:
            s3.put_object(
                Bucket=bucket,
                Key=temp_key,
                Body=file,
                ContentType='application/pdf'
            )
        
        logger.info(f"Successfully uploaded to S3: s3://{bucket}/{temp_key}")
        return bucket, temp_key
        
    except Exception as e:
        logger.error(f"Error uploading local PDF to S3: {str(e)}")
        raise

def cleanup_temp_s3_file(bucket: str, key: str):
    """
    Delete the temporary file from S3 after processing.
    """
    try:
        s3.delete_object(Bucket=bucket, Key=key)
        logger.info(f"Cleaned up temporary S3 file: s3://{bucket}/{key}")
    except Exception as e:
        logger.warning(f"Failed to cleanup temporary S3 file: {str(e)}")

def extract_text_from_local_pdf_with_textract(file_path: str) -> tuple:
    """
    Extract text from a local PDF file using Textract (via temporary S3 upload).
    Returns: (forms_dict, tables_list, text_lines_list, page_count)
    """
    temp_bucket = None
    temp_key = None
    
    try:
        logger.info(f"Extracting text from local PDF using Textract: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        # Upload to S3 temporarily
        temp_bucket, temp_key = upload_local_pdf_to_s3(file_path)
        
        # Use existing Textract function
        kvs, tables, lines, page_count = extract_text_from_pdf(temp_bucket, temp_key)
        
        logger.info(f"Textract extracted {len(kvs)} key-value pairs, {len(tables)} tables, {len(lines)} text lines from {page_count} pages")
        
        return kvs, tables, lines, page_count
        
    except Exception as e:
        logger.error(f"Error extracting text from local PDF with Textract: {str(e)}")
        raise
    finally:
        # Clean up temporary S3 file
        if temp_bucket and temp_key:
            cleanup_temp_s3_file(temp_bucket, temp_key)

def create_single_page_pdf(original_pdf_path: str, page_numbers: list, output_path: str):
    """
    Create a new PDF with only the specified pages from the original PDF.
    """
    try:
        if not PDF_EXTRACTION_AVAILABLE:
            logger.warning("PyMuPDF not available, cannot create single page PDFs")
            return False
            
        doc = fitz.open(original_pdf_path)
        new_doc = fitz.open()
        
        for page_num in page_numbers:
            if page_num < len(doc):
                new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
        
        new_doc.save(output_path)
        new_doc.close()
        doc.close()
        
        logger.info(f"Created single page PDF: {output_path} with pages {page_numbers}")
        return True
        
    except Exception as e:
        logger.error(f"Error creating single page PDF: {str(e)}")
        return False

def extract_text_from_local_pdf_fallback(file_path: str) -> tuple:
    """
    Fallback method: Extract text from a local PDF file using PyPDF2 or pdfplumber.
    Returns: (text_content, page_count)
    """
    try:
        logger.info(f"Extracting text from local PDF using fallback method: {file_path}")
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")
        
        text_content = ""
        page_count = 0
        
        if PDF_LIBRARY == "PyPDF2":
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                page_count = len(pdf_reader.pages)
                
                for page_num in range(page_count):
                    page = pdf_reader.pages[page_num]
                    text_content += page.extract_text() + "\n"
                    
        elif PDF_LIBRARY == "pdfplumber":
            with pdfplumber.open(file_path) as pdf:
                page_count = len(pdf.pages)
                
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_content += page_text + "\n"
        
        logger.info(f"Fallback method extracted {len(text_content)} characters from {page_count} pages")
        return text_content.strip(), page_count
        
    except Exception as e:
        logger.error(f"Error extracting text from local PDF with fallback: {str(e)}")
        raise

def save_results_locally(data: dict, input_file: str, textract_data: dict = None) -> str:
    """
    Save extraction results to a local JSON file.
    Returns: path to the saved file
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_extracted_{timestamp}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Prepare the data to save
        result_data = {
            "extraction_timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "extracted_data": data,
            "textract_data": textract_data if textract_data else None
        }
        
        # Save to local file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved locally to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving results locally: {str(e)}")
        raise

# ---------- TEXTRACT FUNCTIONS ----------
def extract_text_from_pdf(bucket: str, key: str, max_retries=3, retry_delay=2):
    """
    Extracts forms, tables, and text lines from a PDF in S3 using Textract, with retries.
    Returns: (forms_dict, tables_list, text_lines_list, page_count)
    """
    retry_count = 0
    last_exception = None
    
    while retry_count < max_retries:
        try:
            return _do_extract_text_from_pdf(bucket, key)
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            # Only retry for specific errors that might be transient
            if error_code in ['ProvisionedThroughputExceededException', 'ThrottlingException', 'ServiceUnavailable']:
                retry_count += 1
                wait_time = retry_delay * (2 ** retry_count)  # Exponential backoff
                logger.warning(f"Transient error ({error_code}), retrying in {wait_time}s (attempt {retry_count}/{max_retries})")
                time.sleep(wait_time)
                last_exception = e
            else:
                # Don't retry for other client errors
                raise
        except Exception as e:
            # For other exceptions, increment retry but don't wait as long
            retry_count += 1
            logger.warning(f"Error in extraction, retrying (attempt {retry_count}/{max_retries}): {str(e)}")
            time.sleep(retry_delay)
            last_exception = e
    
    # If we've exhausted retries, raise the last exception
    logger.error(f"Failed after {max_retries} attempts")
    if last_exception:
        raise last_exception
    else:
        raise Exception(f"Failed after {max_retries} attempts")

def _do_extract_text_from_pdf(bucket: str, key: str):
    """
    Actual implementation of text extraction using Textract.
    """
    def start_document_analysis(bucket, key, features=['FORMS', 'TABLES']):
        try:
            # Define the Textract call function for retry wrapper
            def make_start_analysis_call():
                return textract.start_document_analysis(
                    DocumentLocation={'S3Object': {'Bucket': bucket, 'Name': key}},
                    FeatureTypes=features
                )
            
            # Use retry wrapper for the Textract call
            response = retry_textract_call(make_start_analysis_call)
            return response['JobId']
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'InvalidS3ObjectException':
                logger.error(f"Textract could not access the S3 object: s3://{bucket}/{key}")
                raise ValueError(f"Textract could not access the S3 object. Check if the file exists and permissions are correct.")
            elif error_code == 'BadDocumentException':
                logger.error(f"Textract could not process the document: s3://{bucket}/{key}")
                raise ValueError(f"Textract could not process the document. Check if it's a valid PDF.")
            else:
                raise
   
    def wait_for_job(job_id, max_attempts=42, delay=10):
        for attempt in range(max_attempts):
            # Define the Textract call function for retry wrapper
            def make_get_analysis_call():
                return textract.get_document_analysis(JobId=job_id)
            
            # Use retry wrapper for the Textract call
            response = retry_textract_call(make_get_analysis_call)
            status = response['JobStatus']
            logger.info(f"Attempt {attempt + 1}: Job {job_id} status = {status}")
            if status == 'SUCCEEDED':
                return True
            elif status == 'FAILED':
                raise Exception(f"Textract job failed: {response.get('StatusMessage')}")
            time.sleep(delay)
        raise TimeoutError("Textract job timed out")

    def get_all_blocks(job_id):
        blocks = []
        next_token = None
        while True:
            kwargs = {'JobId': job_id}
            if next_token:
                kwargs['NextToken'] = next_token
            
            # Define the Textract call function for retry wrapper
            def make_get_analysis_call():
                return textract.get_document_analysis(**kwargs)
            
            # Use retry wrapper for the Textract call
            response = retry_textract_call(make_get_analysis_call)
            blocks.extend(response['Blocks'])
            next_token = response.get('NextToken')
            if not next_token:
                break
        return blocks

    def get_text_for_block(block, block_map):
        text = ''
        for rel in block.get('Relationships', []):
            if rel['Type'] == 'CHILD':
                for cid in rel['Ids']:
                    word = block_map.get(cid)
                    if word and word['BlockType'] == 'WORD':
                        text += word['Text'] + ' '
                    elif word and word['BlockType'] == 'SELECTION_ELEMENT' and word.get('SelectionStatus') == 'SELECTED':
                        text += '☑ '
        return text.strip()

    def extract_kv_pairs(blocks, block_map):
        key_map = {}
        value_map = {}
        for block in blocks:
            if block['BlockType'] == 'KEY_VALUE_SET':
                if 'KEY' in block.get('EntityTypes', []):
                    key_map[block['Id']] = block
                else:
                    value_map[block['Id']] = block
        kvs = {}
        for key_id, key_block in key_map.items():
            key_text = get_text_for_block(key_block, block_map)
            value_block = None
            for rel in key_block.get('Relationships', []):
                if rel['Type'] == 'VALUE':
                    for value_id in rel['Ids']:
                        value_block = value_map.get(value_id)
            value_text = get_text_for_block(value_block, block_map) if value_block else ''
            if key_text and value_text:  # Only include if both key and value are non-empty
                kvs[key_text] = value_text
        return kvs

    def extract_tables(blocks, block_map):
        tables = []
        for block in blocks:
            if block['BlockType'] == 'TABLE':
                table = {}
                # Get IDs of child CELL blocks
                table_cells = []
                if 'Relationships' in block:
                    for rel in block['Relationships']:
                        if rel['Type'] == 'CHILD':
                            table_cells.extend(rel['Ids'])

                for cell_id in table_cells:
                    cell = block_map.get(cell_id)
                    if not cell or cell['BlockType'] != 'CELL':
                        continue
                    row = cell['RowIndex']
                    col = cell['ColumnIndex']
                    text = get_text_for_block(cell, block_map)
                    table.setdefault(row, {})[col] = text

                if not table:
                    continue
                    
                max_row = max(table.keys())
                max_col = max(max(row.keys()) for row in table.values())
                grid = [['' for _ in range(max_col)] for _ in range(max_row)]
                for r in table:
                    for c in table[r]:
                        grid[r - 1][c - 1] = table[r][c]

                tables.append(grid)
        return tables

    def extract_text_lines(blocks):
        return [block['Text'] for block in blocks if block['BlockType'] == 'LINE']

    try:
        logger.info(f"Starting Textract analysis for s3://{bucket}/{key}")
        
        # Start textract job
        analysis_job_id = start_document_analysis(bucket, key)
        wait_for_job(analysis_job_id)
        analysis_blocks = get_all_blocks(analysis_job_id)
        block_map = {b['Id']: b for b in analysis_blocks}

        # Extract content
        kvs = extract_kv_pairs(analysis_blocks, block_map)
        tables = extract_tables(analysis_blocks, block_map)
        lines = extract_text_lines(analysis_blocks)
        
        # Count pages
        page_count = sum(1 for block in analysis_blocks if block['BlockType'] == 'PAGE')
        
        logger.info(f"Extracted {len(lines)} lines, {len(kvs)} key-value pairs, {len(tables)} tables, {page_count} pages.")
        return kvs, tables, lines, page_count
    except Exception as e:
        logger.error(f"Error extracting text from PDF: {str(e)}")
        raise

def extract_json_from_text(content: str) -> dict:
    """
    Extract valid JSON from text content using multiple strategies.
    """
    # Strategy 1: Direct JSON parsing if the entire content is valid JSON
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass
    
    # Strategy 2: Look for JSON in markdown code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', content, re.DOTALL)
    if code_block_match:
        try:
            json_str = code_block_match.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 3: Find first JSON object using regex
    json_pattern = re.search(r'(\{(?:[^{}]|(?:\{[^{}]*\}))*\})', content, re.DOTALL)
    if json_pattern:
        try:
            json_str = json_pattern.group(1)
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass
    
    # Strategy 4: Find the outermost JSON object (most reliable but can be slow on large content)
    try:
        # Find the first opening brace
        json_start = content.find('{')
        if json_start == -1:
            raise ValueError("No JSON object found")
        
        # Track nested braces to find matching closing brace
        open_count = 0
        in_string = False
        escaped = False
        for i in range(json_start, len(content)):
            char = content[i]
            
            # Handle string boundaries
            if char == '"' and not escaped:
                in_string = not in_string
            
            # Skip processing inside strings except for escaping
            if in_string:
                escaped = char == '\\' and not escaped
                continue
                
            # Reset escaped flag outside strings
            escaped = False
            
            # Count braces
            if char == '{':
                open_count += 1
            elif char == '}':
                open_count -= 1
                if open_count == 0:  # Found matching closing brace
                    json_str = content[json_start:i+1]
                    return json.loads(json_str)
    except (ValueError, json.JSONDecodeError):
        pass
    
    # If we get here, we couldn't find valid JSON
    raise ValueError("Could not extract valid JSON from content")

def filter_empty_kvs(kvs: dict) -> dict:
    """
    Filter out empty key-value pairs
    """
    return {k: v for k, v in kvs.items() if v.strip()}

def add_missing_kvs(extracted_info: dict, kvs: dict) -> dict:
    """
    Add any key-value pairs that aren't already present in the extraction results.
    Returns the updated extracted_info.
    """
    # Make sure additional_info exists
    if 'additional_info' not in extracted_info:
        extracted_info['additional_info'] = []
    
    # Get all keys currently in additional_info
    existing_keys = []
    for item in extracted_info['additional_info']:
        existing_keys.extend(list(item.keys()))
    
    # Add missing key-value pairs
    added_count = 0
    for key, value in kvs.items():
        # Skip empty values
        if not value.strip():
            continue
            
        # Check if this key or a similar key is already in additional_info
        key_found = False
        key_lower = key.lower().strip()
        for existing_key in existing_keys:
            existing_key_lower = existing_key.lower().strip()
            if (key_lower in existing_key_lower or 
                existing_key_lower in key_lower or 
                existing_key_lower == key_lower):
                key_found = True
                break
        
        # If not found, add it
        if not key_found:
            extracted_info['additional_info'].append({
                key: {
                    "value": value,
                    "confidence": 0.9,
                    "explanation": f"Extracted from document key-value pairs"
                }
            })
            existing_keys.append(key)
            added_count += 1
    
    if added_count > 0:
        logger.info(f"Added {added_count} additional key-value pairs to additional_info")
    
    return extracted_info

# ---------- CITY VALIDATION FROM EXCEL ----------
# Cache for city lists loaded from Excel
_city_lists_cache = {
    'origin_cities': None,
    'destination_cities': None,
    'last_loaded': None
}

def load_city_lists_from_excel(bucket: str = "pando-j-and-j-invoice", 
                                excel_key: str = "templates/Excel_mapping/UNLOCODE cities.xlsx",
                                force_reload: bool = False) -> Tuple[List[str], List[str]]:
    """
    Load origin and destination city lists from Excel file in S3.
    
    Args:
        bucket: S3 bucket name (default: "pando-j-and-j-invoice")
        excel_key: S3 key for the Excel file (default: "templates/Excel_mapping/UNLOCODE cities.xlsx")
        force_reload: If True, reload even if cached
    
    Returns:
        tuple: (origin_cities_list, destination_cities_list)
    """
    global _city_lists_cache
    
    # Return cached data if available and not forcing reload
    if not force_reload and _city_lists_cache['origin_cities'] is not None:
        logger.info("Using cached city lists from Excel")
        return _city_lists_cache['origin_cities'], _city_lists_cache['destination_cities']
    
    try:
        logger.info(f"Loading city lists from s3://{bucket}/{excel_key}")
        response = s3.get_object(Bucket=bucket, Key=excel_key)
        excel_content = response['Body'].read()
        
        # Read Excel file - assuming first sheet contains the data
        df = pd.read_excel(BytesIO(excel_content), sheet_name=0)
        
        # Extract column A (Origin Cities) - first column (index 0)
        origin_cities = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()
        # Remove empty strings
        origin_cities = [city for city in origin_cities if city and city.lower() != 'nan']
        
        # Extract column B (Destination Cities) - second column (index 1)
        destination_cities = df.iloc[:, 1].dropna().astype(str).str.strip().tolist()
        # Remove empty strings
        destination_cities = [city for city in destination_cities if city and city.lower() != 'nan']
        
        logger.info(f"Loaded {len(origin_cities)} origin cities and {len(destination_cities)} destination cities from Excel")
        
        # Cache the results
        _city_lists_cache['origin_cities'] = origin_cities
        _city_lists_cache['destination_cities'] = destination_cities
        _city_lists_cache['last_loaded'] = time.time()
        
        return origin_cities, destination_cities
        
    except Exception as e:
        logger.error(f"Error loading city lists from Excel: {e}")
        # If loading fails, return empty lists (validation will fail)
        if _city_lists_cache['origin_cities'] is None:
            logger.warning("No cached city lists available, returning empty lists")
            return [], []
        else:
            logger.warning("Using cached city lists due to load error")
            return _city_lists_cache['origin_cities'], _city_lists_cache['destination_cities']

def find_closest_city(city_name: str, valid_cities: list = None) -> tuple:
    """
    Find the closest matching city from the valid cities list.
    
    Args:
        city_name: The city name to match
        valid_cities: List of valid cities (required, no default)
    
    Returns:
        tuple: (matched_city, match_type) where match_type is 'exact', 'close', or 'none'
    """
    if valid_cities is None or len(valid_cities) == 0:
        logger.warning("No valid cities list provided to find_closest_city")
        return None, 'none'
    
    if not city_name or not isinstance(city_name, str):
        return None, 'none'
    
    city_name_clean = city_name.strip()
    if not city_name_clean:
        return None, 'none'
    
    # Try exact match (case-insensitive)
    for valid_city in valid_cities:
        if valid_city.lower() == city_name_clean.lower():
            logger.info(f"Exact city match found: '{city_name_clean}' -> '{valid_city}'")
            return valid_city, 'exact'
    
    # Try partial match - check if the city name is contained in any valid city
    # or if any valid city is contained in the city name
    for valid_city in valid_cities:
        valid_lower = valid_city.lower()
        city_lower = city_name_clean.lower()
        
        # Check if city name is part of valid city (e.g., "Chennai" in "Chennai (ex Madras)")
        if city_lower in valid_lower:
            logger.info(f"Partial city match found (city in valid): '{city_name_clean}' -> '{valid_city}'")
            return valid_city, 'close'
        
        # Check if valid city is part of city name (e.g., "Chennai (ex Madras)" contains "Chennai")
        # Extract base city name (remove parentheticals from valid city)
        base_valid = valid_lower.split('(')[0].strip()
        if base_valid and base_valid in city_lower:
            logger.info(f"Partial city match found (valid in city): '{city_name_clean}' -> '{valid_city}'")
            return valid_city, 'close'
        
        # Check if base city names match (remove parentheticals from both)
        base_city = city_lower.split('(')[0].strip()
        if base_valid and base_city == base_valid:
            logger.info(f"Base city match found: '{city_name_clean}' -> '{valid_city}'")
            return valid_city, 'close'
    
    # Try fuzzy matching using simple string similarity
    # Calculate similarity based on common characters and length
    best_match = None
    best_score = 0
    
    city_lower = city_name_clean.lower()
    base_city = city_lower.split('(')[0].strip()
    
    for valid_city in valid_cities:
        valid_lower = valid_city.lower()
        base_valid = valid_lower.split('(')[0].strip()
        
        # Calculate similarity score
        # Check if base names are similar (at least 70% of shorter name matches)
        shorter_len = min(len(base_city), len(base_valid))
        longer_len = max(len(base_city), len(base_valid))
        
        if shorter_len == 0:
            continue
        
        # Count matching characters in order
        matches = 0
        for i in range(min(len(base_city), len(base_valid))):
            if base_city[i] == base_valid[i]:
                matches += 1
        
        similarity = matches / longer_len if longer_len > 0 else 0
        
        # Also check if one starts with the other
        if base_city.startswith(base_valid) or base_valid.startswith(base_city):
            similarity = max(similarity, 0.8)
        
        if similarity > best_score and similarity >= 0.7:
            best_score = similarity
            best_match = valid_city
    
    if best_match:
        logger.info(f"Fuzzy city match found (score: {best_score:.2f}): '{city_name_clean}' -> '{best_match}'")
        return best_match, 'close'
    
    logger.warning(f"No city match found for: '{city_name_clean}'")
    return None, 'none'

def validate_and_correct_cities(extracted_info: dict, document_type: str = None) -> tuple:
    """
    Validate and correct source and destination city names in the extracted info.
    Uses Excel file from S3: pando-j-and-j-invoice/templates/Excel_mapping/UNLOCODE cities.xlsx
    - Source cities are validated against column A (Origin Cities)
    - Destination cities are validated against column B (Destination Cities)
    
    Args:
        extracted_info: The extracted data structure
        document_type: Type of document being processed
    
    Returns:
        tuple: (corrected_info, validation_passed) where validation_passed is True if cities are valid
    """
    # Skip city validation for non-freight documents
    if not document_type or 'freight' not in document_type.lower():
        logger.info("Skipping city validation for non-freight document")
        return extracted_info, True
    
    # Load city lists from Excel
    origin_cities, destination_cities = load_city_lists_from_excel()
    
    if not origin_cities or not destination_cities:
        logger.error("Failed to load city lists from Excel. City validation cannot proceed.")
        return extracted_info, False
    
    corrected_info = json.loads(json.dumps(extracted_info))  # Deep copy
    validation_passed = True
    cities_corrected = []
    
    logger.info("=== VALIDATING AND CORRECTING CITY NAMES ===")
    logger.info(f"Using {len(origin_cities)} origin cities and {len(destination_cities)} destination cities from Excel")
    
    # Handle different document structures
    if document_type and 'freight' in document_type.lower():
        # Freight invoice structure: data[].shipments[].source_city and destination_city
        if 'data' in corrected_info and isinstance(corrected_info['data'], list):
            for data_item in corrected_info['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        # Validate source_city against Origin Cities (column A)
                        if 'source_city' in shipment:
                            city_field = shipment['source_city']
                            if isinstance(city_field, dict) and 'value' in city_field:
                                original_city = city_field.get('value', '')
                                # Handle None, empty string, or non-string values
                                if original_city is None:
                                    original_city = ''
                                original_city = str(original_city).strip()
                                
                                if original_city:
                                    matched_city, match_type = find_closest_city(original_city, origin_cities)
                                    if matched_city:
                                        # Update value even for exact matches to ensure standardized city name is in JSON
                                        if match_type == 'exact':
                                            # Update to standardized name even if it's an exact match (e.g., "MEMPHIS" -> "Memphis")
                                            if original_city != matched_city:
                                                logger.info(f"Standardizing source_city: '{original_city}' -> '{matched_city}'")
                                                city_field['value'] = matched_city
                                                cities_corrected.append(('source_city', original_city, matched_city))
                                        elif match_type == 'close':
                                            logger.info(f"Correcting source_city: '{original_city}' -> '{matched_city}'")
                                            city_field['value'] = matched_city
                                            city_field['explanation'] = f"Corrected from '{original_city}' to '{matched_city}' (close match)"
                                            cities_corrected.append(('source_city', original_city, matched_city))
                                    else:
                                        logger.warning(f"❌ Source city '{original_city}' not found in origin cities list (column A)")
                                        validation_passed = False
                                else:
                                    logger.warning(f"❌ Source city is empty or missing (mandatory field)")
                                    validation_passed = False
                        
                        # Validate destination_city against Destination Cities (column B)
                        if 'destination_city' in shipment:
                            city_field = shipment['destination_city']
                            if isinstance(city_field, dict) and 'value' in city_field:
                                original_city = city_field.get('value', '')
                                # Handle None, empty string, or non-string values
                                if original_city is None:
                                    original_city = ''
                                original_city = str(original_city).strip()
                                
                                if original_city:
                                    matched_city, match_type = find_closest_city(original_city, destination_cities)
                                    if matched_city:
                                        # Update value even for exact matches to ensure standardized city name is in JSON
                                        if match_type == 'exact':
                                            # Update to standardized name even if it's an exact match (e.g., "MEMPHIS" -> "Memphis")
                                            if original_city != matched_city:
                                                logger.info(f"Standardizing destination_city: '{original_city}' -> '{matched_city}'")
                                                city_field['value'] = matched_city
                                                cities_corrected.append(('destination_city', original_city, matched_city))
                                        elif match_type == 'close':
                                            logger.info(f"Correcting destination_city: '{original_city}' -> '{matched_city}'")
                                            city_field['value'] = matched_city
                                            city_field['explanation'] = f"Corrected from '{original_city}' to '{matched_city}' (close match)"
                                            cities_corrected.append(('destination_city', original_city, matched_city))
                                    else:
                                        logger.warning(f"❌ Destination city '{original_city}' not found in destination cities list (column B)")
                                        validation_passed = False
                                else:
                                    logger.warning(f"❌ Destination city is empty or missing (mandatory field)")
                                    validation_passed = False
    
    # Log results
    if cities_corrected:
        logger.info(f"✅ Corrected {len(cities_corrected)} city names:")
        for field, original, corrected in cities_corrected:
            logger.info(f"  - {field}: '{original}' -> '{corrected}'")
    
    if validation_passed:
        logger.info("✅ City validation passed - all cities match valid list")
    else:
        logger.warning("❌ City validation failed - some cities don't match valid list")
    
    return corrected_info, validation_passed

# ---------- JSON STRUCTURE VALIDATION FUNCTIONS ----------
def validate_field_structure(field_value, field_name=""):
    """
    Validate that a field has the correct structure (value, confidence, explanation).
    
    Args:
        field_value: The field value to validate
        field_name: Name of the field (for error messages)
    
    Returns:
        tuple: (is_valid: bool, error_message: str)
    """
    if not isinstance(field_value, dict):
        return False, f"Field '{field_name}' is not a dict, got {type(field_value).__name__}"
    
    # Check for required keys
    required_keys = ['value', 'confidence', 'explanation']
    missing_keys = [key for key in required_keys if key not in field_value]
    
    if missing_keys:
        return False, f"Field '{field_name}' is missing required keys: {missing_keys}"
    
    # Validate confidence is a number between 0 and 1
    confidence = field_value.get('confidence')
    if not isinstance(confidence, (int, float)):
        return False, f"Field '{field_name}' has invalid confidence type: {type(confidence).__name__}, expected number"
    
    if confidence < 0 or confidence > 1:
        return False, f"Field '{field_name}' has invalid confidence value: {confidence}, expected 0-1"
    
    # Validate explanation is a string
    explanation = field_value.get('explanation')
    if not isinstance(explanation, str):
        return False, f"Field '{field_name}' has invalid explanation type: {type(explanation).__name__}, expected string"
    
    # Validate value is string, number, or null
    value = field_value.get('value')
    if value is not None and not isinstance(value, (str, int, float)):
        return False, f"Field '{field_name}' has invalid value type: {type(value).__name__}, expected string, number, or null"
    
    return True, ""


def validate_json_structure(data, path="root"):
    """
    Recursively validate that all fields in the JSON structure follow the schema format
    (value, confidence, explanation structure).
    
    Args:
        data: The data structure to validate
        path: Current path in the structure (for error messages)
    
    Returns:
        tuple: (is_valid: bool, errors: list)
    """
    errors = []
    
    if isinstance(data, dict):
        # Check if this is a base schema field (has value, confidence, explanation)
        if all(key in data for key in ['value', 'confidence', 'explanation']):
            is_valid, error_msg = validate_field_structure(data, path)
            if not is_valid:
                errors.append(error_msg)
            return len(errors) == 0, errors
        
        # Otherwise, recursively check nested structures
        for key, value in data.items():
            new_path = f"{path}.{key}" if path != "root" else key
            
            # Handle arrays
            if isinstance(value, list):
                for i, item in enumerate(value):
                    item_path = f"{new_path}[{i}]"
                    is_valid, item_errors = validate_json_structure(item, item_path)
                    if not is_valid:
                        errors.extend(item_errors)
            # Handle nested objects
            elif isinstance(value, dict):
                is_valid, nested_errors = validate_json_structure(value, new_path)
                if not is_valid:
                    errors.extend(nested_errors)
            # Skip primitive values at root level (like simple strings, numbers)
            # These are valid for non-schema fields
    
    elif isinstance(data, list):
        for i, item in enumerate(data):
            item_path = f"{path}[{i}]"
            is_valid, item_errors = validate_json_structure(item, item_path)
            if not is_valid:
                errors.extend(item_errors)
    
    return len(errors) == 0, errors


def get_required_fields_from_schema(schema):
    """
    Extract required fields from a JSON schema recursively.
    
    Args:
        schema: The JSON schema dictionary
    
    Returns:
        set: Set of required field paths (e.g., {'data', 'data.invoice_number', 'data.shipments.shipment_number'})
    """
    required_fields = set()
    
    def extract_required(schema_obj, prefix=""):
        if not isinstance(schema_obj, dict):
            return
        
        # Get required fields at current level
        current_required = schema_obj.get("required", [])
        for field in current_required:
            field_path = f"{prefix}.{field}" if prefix else field
            required_fields.add(field_path)
        
        # Recursively process properties
        properties = schema_obj.get("properties", {})
        for prop_name, prop_schema in properties.items():
            new_prefix = f"{prefix}.{prop_name}" if prefix else prop_name
            extract_required(prop_schema, new_prefix)
        
        # Handle array items
        items = schema_obj.get("items", {})
        if isinstance(items, dict):
            extract_required(items, prefix)
    
    extract_required(schema)
    return required_fields


def check_missing_required_fields(data, schema, path="root", max_depth=50):
    """
    Check if all required fields from the schema are present in the data.
    
    Args:
        data: The data structure to check
        schema: The JSON schema with required fields
        path: Current path in the structure (for error messages)
        max_depth: Maximum recursion depth to prevent infinite loops
    
    Returns:
        tuple: (is_valid: bool, missing_fields: list)
    """
    missing_fields = []
    visited_paths = set()  # Track visited paths to prevent infinite loops
    
    def check_field(data_obj, schema_obj, current_path="", depth=0):
        # Prevent infinite recursion
        if depth > max_depth:
            logger.warning(f"Maximum recursion depth ({max_depth}) reached at path: {current_path}")
            return
        
        # Track visited paths
        if current_path in visited_paths:
            return  # Already visited this path
        visited_paths.add(current_path)
        
        if not isinstance(schema_obj, dict):
            return
        
        # Get required fields at current level
        current_required = schema_obj.get("required", [])
        properties = schema_obj.get("properties", {})
        
        # Check required fields at this level
        if isinstance(data_obj, dict):
            for field in current_required:
                field_path = f"{current_path}.{field}" if current_path else field
                if field not in data_obj:
                    missing_fields.append(field_path)
        
        # Recursively check nested structures
        if isinstance(data_obj, dict) and depth < max_depth:
            for prop_name, prop_schema in properties.items():
                if prop_name in data_obj:
                    new_path = f"{current_path}.{prop_name}" if current_path else prop_name
                    
                    # Skip if we've already visited this path
                    if new_path in visited_paths:
                        continue
                    
                    # Handle array items
                    if prop_schema.get("type") == "array" and isinstance(data_obj[prop_name], list):
                        items_schema = prop_schema.get("items", {})
                        # Limit array checking to first 10 items to prevent excessive recursion
                        for i, item in enumerate(data_obj[prop_name][:10]):
                            item_path = f"{new_path}[{i}]"
                            check_field(item, items_schema, item_path, depth + 1)
                    # Handle nested objects
                    elif isinstance(prop_schema, dict) and isinstance(data_obj[prop_name], dict):
                        # Only recurse if not a base schema field (value, confidence, explanation)
                        prop_value = data_obj[prop_name]
                        if isinstance(prop_value, dict) and not all(key in prop_value for key in ['value', 'confidence', 'explanation']):
                            check_field(data_obj[prop_name], prop_schema, new_path, depth + 1)
    
    try:
        check_field(data, schema, path, 0)
    except RecursionError as e:
        logger.error(f"Recursion error in check_missing_required_fields: {e}")
        logger.warning("Returning validation as failed due to recursion error")
        # Return as invalid if recursion error occurs
        return False, ["Recursion error during validation"]
    
    return len(missing_fields) == 0, missing_fields


def validate_llm_output(structured_output, schema, document_type=None):
    """
    Comprehensive validation of LLM output:
    1. Validate JSON structure (value, confidence, explanation format)
    2. Check for missing required fields
    
    Args:
        structured_output: The extracted data from LLM
        schema: The JSON schema for validation
        document_type: Document type (for logging)
    
    Returns:
        tuple: (is_valid: bool, validation_errors: list)
    """
    validation_errors = []
    
    logger.info(f"=== VALIDATING LLM OUTPUT STRUCTURE ===")
    
    # Step 1: Validate JSON structure (value, confidence, explanation format)
    try:
        structure_valid, structure_errors = validate_json_structure(structured_output)
        if not structure_valid:
            logger.warning(f"❌ Structure validation failed: {len(structure_errors)} errors")
            for error in structure_errors[:10]:  # Log first 10 errors
                logger.warning(f"  - {error}")
            if len(structure_errors) > 10:
                logger.warning(f"  ... and {len(structure_errors) - 10} more errors")
            validation_errors.extend([f"Structure error: {e}" for e in structure_errors])
        else:
            logger.info("✅ Structure validation passed (all fields have value, confidence, explanation)")
    except Exception as e:
        logger.error(f"Exception during structure validation: {e}")
        logger.warning("Skipping structure validation due to error - will not retry")
        # Don't add to validation_errors - this means we skip validation, not fail it
        # This prevents retries due to validation code errors
    
    # Step 2: Check for missing required fields (only if structure validation passed or was skipped)
    try:
        fields_valid, missing_fields = check_missing_required_fields(structured_output, schema)
        if not fields_valid:
            # Only fail if it's a real validation failure, not a recursion error
            if missing_fields and not (len(missing_fields) == 1 and "Recursion error" in missing_fields[0]):
                logger.warning(f"❌ Required fields validation failed: {len(missing_fields)} missing fields")
                for field in missing_fields[:10]:  # Log first 10 missing fields
                    logger.warning(f"  - Missing: {field}")
                if len(missing_fields) > 10:
                    logger.warning(f"  ... and {len(missing_fields) - 10} more missing fields")
                validation_errors.extend([f"Missing required field: {field}" for field in missing_fields])
            else:
                logger.warning("⚠️ Required fields validation skipped due to recursion error")
                # Don't add to validation_errors - skip validation instead of failing
        else:
            logger.info("✅ Required fields validation passed (all required fields present)")
    except Exception as e:
        logger.error(f"Exception during required fields validation: {e}")
        logger.warning("Skipping required fields validation due to error - will not retry")
        # Don't add to validation_errors - this prevents retries due to validation code errors
    
    is_valid = len(validation_errors) == 0
    
    if is_valid:
        logger.info(f"✅ LLM output validation PASSED for {document_type or 'document'}")
    else:
        logger.error(f"❌ LLM output validation FAILED for {document_type or 'document'}: {len(validation_errors)} issues")
    
    return is_valid, validation_errors


def extract_information_with_claude(text: str, textract_kvs: dict, carrier_name: str, document_type: str = None, main_pdf_name: str = None, max_retries=2) -> dict:
    """
    Extract structured information using Claude with schema validation and retries.
    Uses S3-based prompt templates for carrier-specific and document-specific extraction.
    """
    # Filter out empty key-value pairs
    filtered_kvs = filter_empty_kvs(textract_kvs)
    
    logger.info(f"Using {len(filtered_kvs)} non-empty key-value pairs from total {len(textract_kvs)}")
    
    # Get document-specific schema
    schema = get_document_schema(document_type) if document_type else FREIGHT_INVOICE_SCHEMA
    logger.info(f"Using schema for document type: {document_type or 'freight_invoice'}")
    
    for attempt in range(max_retries + 1):
        try:
            logger.info(f"Starting Claude extraction (attempt {attempt + 1}/{max_retries + 1})")
            
            # Get carrier-specific and document-specific prompt template from S3
            if not carrier_name:
                logger.error("Carrier name is required for extraction")
                raise ValueError("Carrier name is required for extraction")
            
            prompt_template = get_prompt_template(carrier_name, document_type)
            if not prompt_template:
                logger.error(f"No prompt template found for carrier: {carrier_name}, document_type: {document_type}")
                raise ValueError(f"No prompt template found for carrier: {carrier_name}, document_type: {document_type}")
            
            logger.info(f"Using prompt template for: {carrier_name} - {document_type or 'default'}")
            
            # Handle template formatting with multiple possible placeholders
            prompt = None
            
            # Try freight template format first (new structure)
            if document_type and 'freight' in document_type.lower():
                try:
                    prompt = prompt_template.format(
                        freight_invoice_text=text,
                        air_way_bill_text=text,
                        commercial_invoices_text=text
                    )
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"Freight template formatting failed: {e}. Trying fallback.")
                    prompt = None
            
            # If freight formatting failed or not freight document, try other formats
            if prompt is None:
                # Try new format for non-freight documents
                try:
                    prompt = prompt_template.format(
                        freight_invoice_text=text,
                        air_way_bill_text=text,
                        commercial_invoices_text=text
                    )
                except (KeyError, ValueError, IndexError) as e:
                    logger.warning(f"New template format failed: {e}. Trying old format.")
                    # Fallback to old format
                    try:
                        prompt = prompt_template.format(pdf_text=text)
                    except (KeyError, ValueError, IndexError) as e:
                        logger.warning(f"Old template format failed: {e}. Using string replacement.")
                        # Ultimate fallback - replace common placeholders
                        prompt = prompt_template.replace('{pdf_text}', text)
                        prompt = prompt.replace('{freight_invoice_text}', text)
                        prompt = prompt.replace('{air_way_bill_text}', text)
                        prompt = prompt.replace('{commercial_invoices_text}', text)
            
            # Save the extracted text files for debugging
            save_extracted_text_files(text, document_type, carrier_name, main_pdf_name)
            
            model_id = BEDROCK_MODEL_ID
            
            # Use converse model with document-specific tool schema for structured output
            # Create safe tool name by removing spaces and special characters
            safe_doc_type = document_type.lower().replace(' ', '_').replace('-', '_') if document_type else "invoice"
            tool_name = f"validate_{safe_doc_type}_data"
            tool_description = f"Extract and validate {document_type or 'invoice'} fields."
            
            response = bedrock.converse(
                modelId=model_id,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                toolConfig={
                    "tools": [{
                        "toolSpec": {
                            "name": tool_name,
                            "description": tool_description,
                            "inputSchema": {"json": schema}
                        }
                    }],
                },
                additionalModelRequestFields={
                    "reasoning_config": {
                        "type": "enabled",
                        "budget_tokens": 2000
                    },
                    "max_tokens": 100000, 
                }
            )
            
            # Parse response from converse model
            logger.info(f"Claude response received (size: {len(json.dumps(response))} bytes)")
            
            # Cost Tracking
            try:
                usage = response.get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                total_cost = input_tokens * 0.000003 + output_tokens * 0.000015
                logger.info(f"Claude tokens used: input={input_tokens}, output={output_tokens}, cost=${total_cost:.6f}")
            except Exception as e:
                logger.warning(f"Failed to extract cost info: {e}")

            # Extract structured_output from tool use
            structured_output = None
            for msg in response.get("content", []):
                tool_data = msg.get("toolUse", {})
                if "response" in tool_data:
                    structured_output = tool_data["response"]
                    break
                # Alternative: check if the tool use data is directly in input
                elif "input" in tool_data:
                    structured_output = tool_data["input"]
                    logger.info("Using alternative tool response extraction from input field")
                    break

            if not structured_output:
                try:
                    # Fallback output structure
                    content_list = response.get("output", {}).get("message", {}).get("content", [])
                    for msg in content_list:
                        if "toolUse" in msg and "input" in msg["toolUse"]:
                            structured_output = msg["toolUse"]["input"]
                            break
                except Exception as fallback_err:
                    logger.warning(f"Error in fallback structure: {fallback_err}")

            if not structured_output:
                logger.error("No structured output received from Claude")
                raise ValueError("No structured output received from Claude")
            
            logger.info(f"Structured output extracted successfully")
            
            # Use the structured output directly from the tool
            extracted_info = structured_output
            
            # Validate the LLM output structure and required fields
            try:
                validation_valid, validation_errors = validate_llm_output(extracted_info, schema, document_type)
                
                if not validation_valid:
                    error_summary = f"Validation failed with {len(validation_errors)} errors. "
                    error_summary += f"First 3 errors: {', '.join(validation_errors[:3])}"
                    logger.warning(f"❌ LLM output validation failed: {error_summary}")
                    
                    # Retry if we have attempts left - ONLY retry for actual validation failures
                    if attempt < max_retries:
                        logger.warning(f"Retrying extraction due to validation failure (attempt {attempt + 1}/{max_retries + 1})")
                        time.sleep(1)
                        continue
                    else:
                        logger.error(f"❌ Validation failed after {max_retries + 1} attempts. Returning data anyway but with validation errors.")
                        # Log all validation errors for debugging
                        for i, error in enumerate(validation_errors[:20], 1):  # Log first 20 errors
                            logger.error(f"  Validation Error {i}: {error}")
                        if len(validation_errors) > 20:
                            logger.error(f"  ... and {len(validation_errors) - 20} more validation errors")
                else:
                    logger.info(f"✅ LLM output validation passed successfully")
                    # Validation passed - proceed normally, no retry needed
            except RecursionError as rec_err:
                logger.error(f"Recursion error during validation (not a validation failure): {rec_err}")
                logger.info("Validation code error occurred - skipping validation and proceeding (no retry)")
                # Don't retry for validation code errors - just proceed with the data
            except Exception as val_err:
                logger.error(f"Exception during validation (not a validation failure): {val_err}")
                logger.info("Validation code error occurred - skipping validation and proceeding (no retry)")
                # Don't retry for validation code errors - just proceed with the data
            
            # Add any missing KVs directly to the result
            extracted_info = add_missing_kvs(extracted_info, filtered_kvs)
            
            # Validate and correct city names
            extracted_info, city_validation_passed = validate_and_correct_cities(extracted_info, document_type)
            
            # Retry if city validation failed and we have attempts left
            if not city_validation_passed:
                if attempt < max_retries:
                    logger.warning(f"City validation failed. Retrying extraction (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"❌ City validation failed after {max_retries + 1} attempts. Returning data anyway but with invalid cities.")
            
            logger.info(f"Successfully extracted structured information from invoice")
            return extracted_info
            
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(1)
                continue
            else:
                logger.error(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
                raise ValueError(f"Failed to parse JSON from Claude response: {e}")
                
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(1)
                continue
            else:
                logger.error(f"Error in Claude processing after {max_retries + 1} attempts: {e}")
            raise

    # Should never reach here but just in case
    raise ValueError("Failed to extract information after multiple attempts")

def simplify_extracted_info(complex_info):
    """
    Convert the complex structure with confidence and explanation to a simple key-value structure.
    Example: Convert {'invoiceNumber': {'value': '123', 'confidence': 0.9, 'explanation': 'Found on top'}}
    to {'invoiceNumber': '123'}
    """
    simplified = {}
    
    # Process regular fields
    for key, value in complex_info.items():
        if key == 'charges':
            # Handle charges array specially
            simplified_charges = []
            if isinstance(value, list):
                for charge in value:
                    simple_charge = {}
                    for charge_key, charge_value in charge.items():
                        if isinstance(charge_value, dict) and 'value' in charge_value:
                            simple_charge[charge_key] = charge_value['value']
                    simplified_charges.append(simple_charge)
            simplified['charges'] = simplified_charges
        elif key == 'additional_info':
            # Handle additional info array specially
            simplified_additional = []
            if isinstance(value, list):
                for info in value:
                    simple_info = {}
                    for info_key, info_value in info.items():
                        if isinstance(info_value, dict) and 'value' in info_value:
                            simple_info[info_key] = info_value['value']
                    if simple_info:  # Only add if we have values
                        simplified_additional.append(simple_info)
            simplified['additional_info'] = simplified_additional
        # Handle carrier_info and customer objects specially
        elif key in ['carrier_info', 'customer'] and isinstance(value, dict):
            simplified_entity = {}
            for entity_key, entity_value in value.items():
                if isinstance(entity_value, dict) and 'value' in entity_value:
                    simplified_entity[entity_key] = entity_value['value']
            simplified[key] = simplified_entity
        # Handle regular fields with value structure
        elif isinstance(value, dict) and 'value' in value:
            simplified[key] = value['value']
        else:
            simplified[key] = value  # Pass through other values as is
            
    return simplified

def save_results_to_s3(data: dict, input_key: str, textract_data: dict = None) -> str:
    """
    Save extracted information to S3 output bucket
    Returns: S3 URI of the saved file
    """
    carrier_name = None  # Will be auto-detected from freight invoices  # Using default carrier for all documents
    
    if not OUTPUT_BUCKET:
        raise ValueError("OUTPUT_BUCKET environment variable is not set")
    
    try:
        # Create output path based on input path - handle special characters
        file_name = os.path.basename(input_key)
        # Remove problematic characters from file name for output
        clean_file_base = os.path.splitext(file_name)[0]
        clean_file_base = ''.join(c if c.isalnum() or c in '-_.' else '_' for c in clean_file_base)
        
        timestamp = time.strftime('%Y%m%d-%H%M%S')
        output_key = f"extractions/{carrier_name}/{clean_file_base}-{timestamp}.json"
        
        # Create metadata with processing info
        metadata = {
            "original_file": input_key,
            "carrier": carrier_name,
            "processed_time": time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            "extraction_service": "textract-claude"
        }
        
        # Save both complex and simplified versions
        simplified_data = simplify_extracted_info(data)
        
        # Combine data and metadata
        output_data = {
            "metadata": metadata,
            "extracted_data": simplified_data,
            "extracted_data_full": data,  # Include full data with confidence scores for reference
            "textract_data": textract_data  # Include raw Textract data
        }
        
        # Save to S3
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=output_key,
            Body=json.dumps(output_data, indent=2),
            ContentType="application/json"
        )
        
        logger.info(f"Saved extraction results to s3://{OUTPUT_BUCKET}/{output_key}")
        return f"s3://{OUTPUT_BUCKET}/{output_key}"
        
    except Exception as e:
        logger.error(f"Error saving results to S3: {str(e)}")
        raise

# ---------- MAIN PROCESSING FUNCTION ----------
def process_invoice(input_bucket, input_key):
    """Main function to process an invoice using Textract and Claude."""
    try:
        # Step 1: Extract text from PDF using Textract
        logger.info(f"Processing invoice: s3://{input_bucket}/{input_key}")
        kvs, tables, lines, page_count = extract_text_from_pdf(input_bucket, input_key)
            
        # Check if extraction was successful
        if not lines and not tables and not kvs:
            logger.warning(f"Textract could not extract any content from: s3://{input_bucket}/{input_key}")
            return {
                'success': False,
                'error': 'No content could be extracted from the document'
            }
        
        # Create a complete textract_data object to save
        textract_data = {
            "key_value_pairs": kvs,
            "tables": tables,
            "text_lines": lines,
            "page_count": page_count
        }
        
        # Log key-value pairs from Textract (if there aren't too many)
        if len(kvs) < 30:
            logger.info(f"Textract Key-Value Pairs: {json.dumps(kvs, indent=2)}")
        else:
            logger.info(f"Textract extracted {len(kvs)} key-value pairs (too many to log)")
        
        # Log sample of tables data
        logger.info(f"Textract Tables Count: {len(tables)}")
        if tables:
            for i, table in enumerate(tables[:2]):  # Log only first 2 tables
                if table and len(table) > 0:
                    sample_rows = table[:min(3, len(table))]
                    logger.info(f"Table {i+1} (first {len(sample_rows)} rows): {json.dumps(sample_rows)}")
        
        # Log sample of text lines (first and last few)
        if lines:
            first_lines = lines[:min(5, len(lines))]
            last_lines = lines[max(0, len(lines)-5):] if len(lines) > 5 else []
            logger.info(f"First few OCR lines: {json.dumps(first_lines)}")
            if last_lines:
                logger.info(f"Last few OCR lines: {json.dumps(last_lines)}")
            logger.info(f"Total OCR lines: {len(lines)}")
        
        # Format extracted data for the prompt
        def table_to_markdown(table):
            if not table or not table[0]:
                return ""
            col_count = len(table[0])
            cleaned_table = []
            for row in table:
                cleaned_row = [(cell if cell is not None else "").strip() for cell in row]
                if len(cleaned_row) < col_count:
                    cleaned_row += [""] * (col_count - len(cleaned_row))
                cleaned_table.append(cleaned_row)
            header = "| " + " | ".join(cleaned_table[0]) + " |"
            separator = "| " + " | ".join(["---"] * col_count) + " |"
            rows = ["| " + " | ".join(row) + " |" for row in cleaned_table[1:]]
            return "\n".join([header, separator] + rows)
        
        # Prepare formatted text but limit it to a reasonable size
        formatted_kvs = "\n".join(f"{key}: {value}" for key, value in kvs.items() if value.strip())
        
        # Limit number of tables to prevent prompt size issues
        tables_to_format = tables[:min(5, len(tables))]  # At most 5 tables
        formatted_tables = "\n\n".join(table_to_markdown(tbl) for tbl in tables_to_format)
        
        # Limit number of text lines to prevent prompt size issues
        max_lines = 500  # Limit to 500 lines
        if len(lines) > max_lines:
            selected_lines = lines[:max_lines//2] + lines[-max_lines//2:]
            formatted_lines = "\n".join(selected_lines)
            logger.info(f"Limited OCR text to {max_lines} lines (out of {len(lines)} total)")
        else:
            formatted_lines = "\n".join(lines)
        
        raw_text = "\n\n".join(filter(None, [
            "Key-Value Pairs:\n" + formatted_kvs if formatted_kvs else None,
            "Tables:\n" + formatted_tables if formatted_tables else None,
            "Text Lines:\n" + formatted_lines if formatted_lines else None
        ]))
        
        # Step 2: Extract structured information using Claude with schema validation
        logger.info("Using default carrier for extraction since no carrier classification is available in this function")
        # Default to a neutral placeholder when classifier isn't used here
        default_carrier = "GENERIC"
        # Extract main PDF name from the input key
        main_pdf_name = os.path.splitext(os.path.basename(input_key))[0]
        extracted_info = extract_information_with_claude(raw_text, kvs, default_carrier, "freight_invoice", main_pdf_name)
        
        # Step 3: Save results to output S3 bucket, including the textract data
        output_location = save_results_to_s3(extracted_info, input_key, textract_data)
        
        # Return simplified data in the response
        simplified_data = simplify_extracted_info(extracted_info)
        
        return {
            'success': True,
            'carrier': None,  # Will be auto-detected
            'extracted_info': simplified_data,
            'page_count': page_count,
            'output_location': output_location
        }
        
    except Exception as e:
        logger.error(f"Error processing invoice: {str(e)}", exc_info=True)  # Include stack trace
        return {
            'success': False,
            'error': str(e)
        }

# ---------- NEW S3-BASED PROCESSING FUNCTION ----------
def process_invoice_with_s3_clustering(input_file: str, carrier_name: str = None, use_textract: bool = True, s3_bucket: str = None, original_filename: str = None, sender_email: str = None) -> dict:
    """
    Process an invoice with S3-based clustering and separate document processing.
    
    Args:
        input_file: Path to the local PDF file
        carrier_name: Optional carrier name (will be auto-detected from freight invoices)
        use_textract: Whether to use Textract (True) or fallback method (False)
        s3_bucket: S3 bucket for storing clustered PDFs and results
        original_filename: Original filename for S3 storage
        sender_email: Email address of the sender (for authorization check)
    
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing invoice with S3 clustering: {input_file}")
        
        # Use OUTPUT_BUCKET if s3_bucket not provided
        if not s3_bucket:
            s3_bucket = OUTPUT_BUCKET
        
        # Use original filename if provided, otherwise use the input file basename
        main_pdf_name = original_filename if original_filename else os.path.basename(input_file)
        
        # Step 1: Extract and analyze pages to identify document types
        logger.info("Step 1: Extracting and analyzing pages...")
        page_extractor = LLMBasedPDFClusterer(input_file, OUTPUT_DIR)
        
        # Run the full clustering process with S3 storage
        clustering_report = page_extractor.process(s3_bucket, main_pdf_name)
        
        if not clustering_report or not clustering_report.get('saved_clusters'):
            logger.error("Failed to create clusters or save to S3")
            return {
                'success': False,
                'error': 'Failed to create clusters or save to S3'
            }
        
        saved_clusters = clustering_report['saved_clusters']
        
        # Step 2: Identify carrier from freight invoices using LLM
        logger.info("Step 2: Identifying carrier from freight invoices using LLM...")
        carrier_result = page_extractor.get_carrier_from_freight_invoices()
        if carrier_result and isinstance(carrier_result, dict):
            identified_carrier = carrier_result.get('carrier')
            logger.info(f"Carrier identified as: {identified_carrier}")
        elif carrier_result:
            identified_carrier = carrier_result
            logger.info(f"Carrier identified as: {identified_carrier}")
        else:
            logger.error("Failed to identify carrier from freight invoices")
            return {
                'success': False,
                'error': 'Could not identify carrier from freight invoices',
                'carrier': None
            }
        
        carrier_name = identified_carrier
        
        # Step 2.5: Check sender authorization for the identified carrier
        if sender_email:
            logger.info(f"Step 2.5: Checking sender authorization for carrier {identified_carrier}...")
            is_authorized, auth_error = check_sender_carrier_authorization(sender_email, identified_carrier)
            if not is_authorized:
                logger.error(f"❌ Authorization check failed: {auth_error}")
                logger.warning("🛑 STOPPING PROCESSING - Carrier and sender email do not match. No further processing will occur.")
                return {
                    'success': False,
                    'error': auth_error,
                    'error_code': 'AUTHORIZATION_FAILED',
                    'carrier': identified_carrier,
                    'sender_email': sender_email
                }
        
        # Step 3: Process documents - use separated approach for freight, individual for others
        logger.info("Step 3: Processing documents with appropriate method...")
        individual_results = {}
        all_successful = True
        
        # Get existing Textract data for freight invoices (if available from carrier identification)
        freight_textract_data = None
        if hasattr(page_extractor, 'freight_textract_data'):
            freight_textract_data = page_extractor.freight_textract_data
        
        # Check if we have freight documents that need separated processing
        freight_clusters = []
        other_clusters = []
        
        for cluster_id, cluster_data in saved_clusters.items():
            doc_type = cluster_data.get('doc_type', '').lower()
            if 'freight' in doc_type and 'invoice' in doc_type:
                freight_clusters.append((cluster_id, cluster_data))
            else:
                other_clusters.append((cluster_id, cluster_data))
        
        # Process freight documents using separated approach if we have them
        if freight_clusters:
            logger.info(f"Found {len(freight_clusters)} freight document(s) - using separated processing approach")
            print(f"=== PROCESSING FREIGHT DOCUMENTS WITH SEPARATED APPROACH ===")
            
            # Collect all textract results for separated processing
            all_textract_results = {}
            
            # Add freight invoice data
            if freight_textract_data:
                all_textract_results['freight_invoices'] = freight_textract_data
                logger.info("Added freight invoice textract data to separated processing")
                print("✅ Added freight invoice textract data")
            
            # Process other document types and add their textract data
            for cluster_id, cluster_data in other_clusters:
                doc_type = cluster_data.get('doc_type', '').lower()
                doc_type_normalized = doc_type.replace(' ', '_').replace('-', '_')
                
                # Skip unknown document types
                known_doc_types = [
                    'commercial_invoice', 'commercial_invoices', 
                    'air_waybill', 'air_waybills','airway_bill', 'airway_bills',
                    'bill_of_lading', 'bills_of_lading',
                    'packing_list', 'packing_lists',
                    'import_declaration', 'import_declarations',
                    'delivery_note', 'delivery_notes',
                    'receipt', 'receipts'
                ]
                
                if doc_type_normalized not in known_doc_types:
                    logger.info(f"Skipping unknown document type: {doc_type} (cluster {cluster_id})")
                    continue
                
                # Extract textract data for this document type
                s3_path = cluster_data.get('s3_path')
                if s3_path and s3_path.startswith('s3://'):
                    s3_path_clean = s3_path.replace('s3://', '')
                    parts = s3_path_clean.split('/', 1)
                    bucket_part = parts[0]
                    key_part = parts[1] if len(parts) > 1 else ''
                    
                    logger.info(f"Extracting textract data for {doc_type} from {s3_path}")
                    
                    try:
                        kvs, tables, lines, page_count = extract_text_from_pdf(bucket_part, key_part)
                        textract_data = {
                            "key_value_pairs": kvs,
                            "tables": tables,
                            "text_lines": lines,
                            "page_count": page_count,
                            "method": "textract"
                        }
                        
                        # Map to the correct key for separated processing
                        if 'air' in doc_type and 'waybill' in doc_type:
                            all_textract_results['air_waybills'] = textract_data
                            logger.info(f"Added air waybill textract data ({len(lines)} lines, {len(kvs)} kvs)")
                            print(f"✅ Added air waybill textract data")
                        elif 'commercial' in doc_type and 'invoice' in doc_type:
                            all_textract_results['commercial_invoices'] = textract_data
                            logger.info(f"Added commercial invoice textract data ({len(lines)} lines, {len(kvs)} kvs)")
                            print(f"✅ Added commercial invoice textract data")
                        
                    except Exception as e:
                        logger.error(f"Failed to extract textract data for {doc_type}: {e}")
                        continue
            
            # Use separated processing for freight documents
            if all_textract_results:
                print(f"All textract results: {all_textract_results}")
                logger.info(f"Processing freight documents with {len(all_textract_results)} document types")
                print(f"Processing freight documents with {list(all_textract_results.keys())}")
                
                try:
                    # Use the separated extraction function
                    extracted_info = extract_information_with_claude_separated("", all_textract_results, carrier_name, main_pdf_name)
                    
                    # Create result for freight processing and save to S3
                    for cluster_id, cluster_data in freight_clusters:
                        # Simplify the extracted data
                        simplified_data = simplify_extracted_info(extracted_info)
                        
                        # Save results to S3 with proper naming (same as individual processing)
                        cluster_info = {
                            'cluster_id': cluster_id,
                            'doc_type': cluster_data.get('doc_type'),
                            'pages': cluster_data.get('pages'),
                            'safe_doc_type': cluster_data.get('safe_doc_type')
                        }
                        
                        # Create textract data for saving (use freight textract data)
                        textract_data = all_textract_results.get('freight_invoices', {})
                        
                        # Save document results to S3
                        output_location = save_document_results_to_s3(
                            simplified_data, 
                            cluster_info, 
                            main_pdf_name, 
                            textract_data
                        )
                        
                        individual_results[cluster_id] = {
                            'success': True,
                            'cluster_id': cluster_id,
                            'doc_type': cluster_data.get('doc_type'),
                            'extracted_info': simplified_data,
                            'textract_data': textract_data,
                            'output_location': output_location,
                            'processing_method': 'separated_freight_processing'
                        }
                        logger.info(f"Successfully processed freight cluster {cluster_id} using separated approach")
                        logger.info(f"Output path: {output_location}")
                        print(f"✅ Successfully processed freight cluster {cluster_id}")
                        print(f"✅ Freight JSON saved to: {output_location}")
                    
                except Exception as e:
                    logger.error(f"Failed to process freight documents with separated approach: {e}")
                    print(f"❌ Failed to process freight documents: {e}")
                    all_successful = False
                    
                    # Mark freight clusters as failed
                    for cluster_id, cluster_data in freight_clusters:
                        individual_results[cluster_id] = {
                            'success': False,
                            'error': str(e),
                            'doc_type': cluster_data.get('doc_type')
                        }
        
        # Process other document types individually
        if other_clusters:
            logger.info(f"Processing {len(other_clusters)} non-freight documents individually")
            print(f"=== PROCESSING {len(other_clusters)} NON-FREIGHT DOCUMENTS INDIVIDUALLY ===")
            
            for cluster_id, cluster_data in other_clusters:
                s3_path = cluster_data.get('s3_path')
                if not s3_path:
                    logger.warning(f"No S3 path found for cluster {cluster_id}, skipping")
                    continue
                
                # Extract S3 bucket and key from s3_path
                # s3_path format: s3://bucket-name/key/path
                s3_path_clean = s3_path.replace('s3://', '')
                parts = s3_path_clean.split('/', 1)
                bucket_part = parts[0]  # First part is the bucket name
                key_part = parts[1] if len(parts) > 1 else ''  # Rest is the key
                
                logger.info(f"Parsed S3 path - Bucket: {bucket_part}, Key: {key_part}")
                logger.info(f"Full S3 path: {s3_path}")
                
                # Prepare cluster info for processing
                cluster_info = {
                    'cluster_id': cluster_id,
                    'doc_type': cluster_data.get('doc_type'),
                    'pages': cluster_data.get('pages'),
                    'safe_doc_type': cluster_data.get('safe_doc_type')
                }
                
                logger.info(f"Processing cluster {cluster_id}: {cluster_data.get('doc_type')}")
                
                # Check if this is a known document type that we can process
                doc_type = cluster_data.get('doc_type', '').lower()
                
                # Define known document types that we can process
                known_doc_types = [
                    'commercial_invoice', 'commercial_invoices', 
                    'air_waybill', 'air_waybills',
                    'bill_of_lading', 'bills_of_lading',
                    'packing_list', 'packing_lists',
                    'import_declaration', 'import_declarations',
                    'delivery_note', 'delivery_notes',
                    'receipt', 'receipts'
                ]
                
                # Check if document type is known
                doc_type_normalized = doc_type.replace(' ', '_').replace('-', '_')
                if doc_type_normalized not in known_doc_types:
                    logger.info(f"Skipping unknown document type: {doc_type} (cluster {cluster_id})")
                    individual_results[cluster_id] = {
                        'success': False,
                        'skipped': True,
                        'reason': f'Unknown document type: {doc_type}',
                        'doc_type': doc_type
                    }
                    continue
                
                # Process individual clustered document
                result = process_individual_clustered_document(
                    bucket_part, 
                    key_part, 
                    cluster_info, 
                    carrier_name, 
                    main_pdf_name,
                    None  # No existing data for non-freight documents
                )
                
                individual_results[cluster_id] = result
                
                if not result.get('success'):
                    all_successful = False
                    logger.error(f"Failed to process cluster {cluster_id}: {result.get('error')}")
                else:
                    logger.info(f"Successfully processed cluster {cluster_id}")
        
        # Step 4: Create summary report
        summary_report = {
            'main_pdf': main_pdf_name,
            'carrier': carrier_name,
            'total_clusters': len(saved_clusters),
            'successful_clusters': sum(1 for r in individual_results.values() if r.get('success')),
            'failed_clusters': sum(1 for r in individual_results.values() if not r.get('success') and not r.get('skipped')),
            'skipped_clusters': sum(1 for r in individual_results.values() if r.get('skipped')),
            'clustering_report': clustering_report,
            'individual_results': individual_results,
            'processing_method': 's3_clustering_with_separate_processing'
        }
        
        # Save summary report to S3
        summary_location = save_summary_report_to_s3(summary_report, main_pdf_name, s3_bucket)
        
        logger.info(f"S3-based processing completed. Summary saved to: {summary_location}")
        
        return {
            'success': all_successful,
            'carrier': carrier_name,
            'main_pdf': main_pdf_name,
            'total_clusters': len(saved_clusters),
            'successful_clusters': sum(1 for r in individual_results.values() if r.get('success')),
            'failed_clusters': sum(1 for r in individual_results.values() if not r.get('success') and not r.get('skipped')),
            'skipped_clusters': sum(1 for r in individual_results.values() if r.get('skipped')),
            'clustering_report': clustering_report,
            'individual_results': individual_results,
            'summary_location': summary_location,
            'processing_method': 's3_clustering_with_separate_processing'
        }
        
    except Exception as e:
        logger.error(f"Error processing invoice with S3 clustering: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def save_summary_report_to_s3(summary_report: dict, main_pdf_name: str, s3_bucket: str) -> str:
    """
    Save summary report to S3.
    
    Args:
        summary_report: Summary report data
        main_pdf_name: Main PDF name
        s3_bucket: S3 bucket
    
    Returns:
        str: S3 URI of the saved summary report
    """
    try:
        # Create clean names for S3 keys
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
        # Create output path for summary
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_key = f"summary_reports/{clean_main_name}_summary_{timestamp}.json"
        
        # Save to S3
        s3.put_object(
            Bucket=s3_bucket,
            Key=output_key,
            Body=json.dumps(summary_report, indent=2),
            ContentType="application/json"
        )
        
        logger.info(f"Saved summary report to s3://{s3_bucket}/{output_key}")
        return f"s3://{s3_bucket}/{output_key}"
        
    except Exception as e:
        logger.error(f"Error saving summary report to S3: {str(e)}")
        raise

# ---------- LOCAL PROCESSING FUNCTION ----------
def process_invoice_local(input_file: str, carrier_name: str = None, use_textract: bool = True, original_filename: str = None, sender_email: str = None) -> dict:
    """
    Process an invoice from a local PDF file using page extraction and separate Textract processing.
    
    Args:
        input_file: Path to the local PDF file
        carrier_name: Optional carrier name (will be auto-detected from freight invoices)
        use_textract: Whether to use Textract (True) or fallback method (False)
        original_filename: Original filename for processing
        sender_email: Email address of the sender (for authorization check)
    
    Returns:
        dict: Processing results
    """
    try:
        logger.info(f"Processing local invoice with page extraction: {input_file}")
        
        # Use original filename if provided, otherwise use the input file basename
        main_pdf_name = original_filename if original_filename else os.path.basename(input_file)
        
        # Step 1: Extract and analyze pages to identify document types
        logger.info("Step 1: Extracting and analyzing pages...")
        page_extractor = LLMBasedPDFClusterer(input_file, OUTPUT_DIR)
        
        # Run the full clustering process
        clustering_report = page_extractor.process()
        
        # Get document clusters in the format expected by processing
        document_clusters = page_extractor.get_document_clusters_for_processing()
        
        # Step 2: Identify carrier from freight invoices using LLM and get their Textract results
        logger.info("Step 2: Identifying carrier from freight invoices using LLM...")
        carrier_result = page_extractor.get_carrier_from_freight_invoices()
        if carrier_result and isinstance(carrier_result, dict):
            identified_carrier = carrier_result.get('carrier')
            freight_textract_data = carrier_result.get('textract_data')
            logger.info(f"Carrier identified as: {identified_carrier}")
            logger.info(f"Freight invoice Textract data already extracted: {freight_textract_data is not None}")
        elif carrier_result:
            identified_carrier = carrier_result
            freight_textract_data = None
            logger.info(f"Carrier identified as: {identified_carrier}")
        else:
            logger.error("Failed to identify carrier from freight invoices")
            return {
                'success': False,
                'error': 'Could not identify carrier from freight invoices',
                'carrier': None
            }
        
        carrier_name = identified_carrier
        
        # Step 2.5: Check sender authorization for the identified carrier
        if sender_email:
            logger.info(f"Step 2.5: Checking sender authorization for carrier {identified_carrier}...")
            is_authorized, auth_error = check_sender_carrier_authorization(sender_email, identified_carrier)
            if not is_authorized:
                logger.error(f"❌ Authorization check failed: {auth_error}")
                logger.warning("🛑 STOPPING PROCESSING - Carrier and sender email do not match. No further processing will occur.")
                return {
                    'success': False,
                    'error': auth_error,
                    'error_code': 'AUTHORIZATION_FAILED',
                    'carrier': identified_carrier,
                    'sender_email': sender_email
                }
        
        # Step 3: Process documents based on carrier and document type
        logger.info("Step 3: Processing documents based on type and carrier...")
        all_textract_results = {}
        all_text_content = {}
        
        # Use already extracted freight invoice data or process if not available
        if freight_textract_data:
            logger.info("Reusing freight invoice Textract data from carrier identification")
            all_textract_results['freight_invoices'] = freight_textract_data
            all_text_content['freight_invoices'] = combine_textract_data(freight_textract_data)
        else:
            # Fallback: process freight invoices if not already extracted
            freight_pages = document_clusters.get('freight_invoices', [])
            if freight_pages:
                logger.info(f"Processing {len(freight_pages)} freight invoice pages (fallback)")
                freight_textract_data = process_document_pages(input_file, freight_pages, "freight_invoice", use_textract)
                all_textract_results['freight_invoices'] = freight_textract_data
                all_text_content['freight_invoices'] = combine_textract_data(freight_textract_data)
        
        # Process additional documents based on carrier
        if carrier_name and ('MAGNO INTERNATIONAL' in carrier_name.upper() or 'KWEI' in carrier_name.upper() or 'AAMRO' in carrier_name.upper()):
            logger.info(f"Carrier is {carrier_name}, processing air waybills and commercial invoices")
            
            # Process air waybills
            air_waybill_pages = document_clusters.get('air_waybills', [])
            if air_waybill_pages:
                logger.info(f"Processing {len(air_waybill_pages)} air waybill pages")
                air_waybill_textract_data = process_document_pages(input_file, air_waybill_pages, "air_waybill", use_textract)
                all_textract_results['air_waybills'] = air_waybill_textract_data
                all_text_content['air_waybills'] = combine_textract_data(air_waybill_textract_data)
            
            # Process commercial invoices
            commercial_pages = document_clusters.get('commercial_invoices', [])
            if commercial_pages:
                logger.info(f"Processing {len(commercial_pages)} commercial invoice pages")
                commercial_textract_data = process_document_pages(input_file, commercial_pages, "commercial_invoice", use_textract)
                all_textract_results['commercial_invoices'] = commercial_textract_data
                all_text_content['commercial_invoices'] = combine_textract_data(commercial_textract_data)
                
        elif carrier_name and 'MGIO' in carrier_name.upper():
            logger.info(f"Carrier is MGIO ({carrier_name}), processing commercial invoices and bills of lading")
            
            # Process commercial invoices
            commercial_pages = document_clusters.get('commercial_invoices', [])
            if commercial_pages:
                logger.info(f"Processing {len(commercial_pages)} commercial invoice pages")
                commercial_textract_data = process_document_pages(input_file, commercial_pages, "commercial_invoice", use_textract)
                all_textract_results['commercial_invoices'] = commercial_textract_data
                all_text_content['commercial_invoices'] = combine_textract_data(commercial_textract_data)
            
            # Process bills of lading
            bol_pages = document_clusters.get('bills_of_lading', [])
            if bol_pages:
                logger.info(f"Processing {len(bol_pages)} bill of lading pages")
                bol_textract_data = process_document_pages(input_file, bol_pages, "bill_of_lading", use_textract)
                all_textract_results['bills_of_lading'] = bol_textract_data
                all_text_content['bills_of_lading'] = combine_textract_data(bol_textract_data)
        else:
            logger.info(f"Carrier is not in known list ({carrier_name}), skipping additional document processing")
        
        # Step 4: Combine all text content for Claude processing
        logger.info("Step 4: Combining text content for Claude analysis...")
        combined_text = ""
        for doc_type, text_content in all_text_content.items():
            combined_text += f"\n=== {doc_type.upper().replace('_', ' ')} ===\n"
            combined_text += text_content + "\n"
        
        if not combined_text.strip():
            logger.warning("No text content extracted from any documents")
            return {
                'success': False,
                'error': 'No text could be extracted from any documents'
            }
        
        # Step 5: Extract information using Claude with separate document context
        logger.info("Step 5: Extracting structured information using Claude...")
        extracted_info = extract_information_with_claude_separated("", all_textract_results, carrier_name)
        
        # Step 6: Simplify the extracted data
        simplified_data = simplify_extracted_info(extracted_info)
        
        # Step 7: Save results locally with separate document data
        output_location = save_results_locally_separated(simplified_data, input_file, all_textract_results, carrier_name)
        
        logger.info(f"Local processing completed successfully. Results saved to: {output_location}")
        
        return {
            'success': True,
            'carrier': carrier_name,
            'extracted_info': simplified_data,
            'document_clusters': document_clusters,
            'clustering_report': clustering_report,
            'textract_results_by_type': all_textract_results,
            'output_location': output_location,
            'processing_method': 'textract_with_page_extraction' if use_textract else 'fallback_with_page_extraction'
        }
        
    except Exception as e:
        logger.error(f"Error processing local invoice: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def process_document_pages(input_file: str, page_numbers: list, doc_type: str, use_textract: bool) -> dict:
    """
    Process specific pages of a document type using Textract or fallback method.
    """
    try:
        # Create temporary single-page PDF for this document type
        temp_pdf_path = f"/tmp/{doc_type}_{int(time.time())}.pdf"
        
        if not create_single_page_pdf(input_file, page_numbers, temp_pdf_path):
            logger.error(f"Failed to create single page PDF for {doc_type}")
            return {'error': 'Failed to create single page PDF'}
        
        try:
            if use_textract:
                # Use Textract for processing
                kvs, tables, lines, page_count = extract_text_from_local_pdf_with_textract(temp_pdf_path)
                return {
                    'key_value_pairs': kvs,
                    'tables': tables,
                    'text_lines': lines,
                    'page_count': page_count,
                    'method': 'textract',
                    'pages': page_numbers
                }
            else:
                # Use fallback method
                text_content, page_count = extract_text_from_local_pdf_fallback(temp_pdf_path)
                return {
                    'text_content': text_content,
                    'page_count': page_count,
                    'method': 'fallback',
                    'pages': page_numbers
                }
        finally:
            # Clean up temporary file
            if os.path.exists(temp_pdf_path):
                os.remove(temp_pdf_path)
                
    except Exception as e:
        logger.error(f"Error processing {doc_type} pages: {str(e)}")
        return {'error': str(e), 'pages': page_numbers}

def combine_textract_data(textract_data: dict) -> str:
    """Combine Textract data into a single text string"""
    if 'error' in textract_data:
        return f"Error: {textract_data['error']}"
    
    combined_text = ""
    
    if textract_data.get('method') == 'textract':
        # Priority 1: Add text lines FIRST (most important - contains all visible text including headers/footers)
        # This is critical for carrier identification as it includes company names, logos, etc.
        if textract_data.get('text_lines'):
            lines_text = "\n".join(textract_data['text_lines'])
            if lines_text.strip():
                combined_text += lines_text + "\n"
                logger.info(f"Added {len(textract_data['text_lines'])} text lines to combined text")
        
        # Priority 2: Add key-value pairs
        if textract_data.get('key_value_pairs'):
            kv_text = "\n".join([f"{k}: {v}" for k, v in textract_data['key_value_pairs'].items()])
            if kv_text:
                combined_text += kv_text + "\n"
                logger.info(f"Added {len(textract_data['key_value_pairs'])} key-value pairs to combined text")
        
        # Priority 3: Add tables as readable text (not as string representation of arrays)
        if textract_data.get('tables'):
            table_count = 0
            for table in textract_data['tables']:
                if table and isinstance(table, list):
                    # Convert table rows to readable text
                    table_text = ""
                    for row in table:
                        if row and isinstance(row, list):
                            # Filter out empty cells and join with spaces/tabs
                            row_text = " | ".join([str(cell).strip() if cell else "" for cell in row if cell and str(cell).strip()])
                            if row_text.strip():
                                table_text += row_text + "\n"
                    if table_text.strip():
                        combined_text += "\n" + table_text
                        table_count += 1
            if table_count > 0:
                logger.info(f"Added {table_count} tables to combined text")
    
    elif textract_data.get('method') == 'fallback':
        combined_text = textract_data.get('text_content', '')
    
    result = combined_text.strip()
    logger.info(f"Combined textract data: {len(result)} total characters")
    return result

def extract_information_with_claude_separated(combined_text: str, textract_results_by_type: dict, carrier_name: str = None, main_pdf_name: str = None, max_retries=2) -> dict:
    """
    Extract structured information using Claude with awareness of separate document types.
    Uses S3-based prompt templates for carrier-specific extraction.
    For freight documents, passes correct text to each placeholder (freight_invoice_text, air_way_bill_text, commercial_invoices_text).
    For other document types, uses hardcoded prompts with only their own text.
    """
    logger.info("Starting Claude extraction with separated document context")
    print("=== STARTING CLAUDE EXTRACTION WITH SEPARATED DOCUMENT CONTEXT ===")
    
    # Log input parameters
    logger.info(f"Carrier name: {carrier_name}")
    logger.info(f"Number of document types to process: {len(textract_results_by_type)}")
    logger.info(f"Document types available: {list(textract_results_by_type.keys())}")
    
    print(f"Carrier: {carrier_name}")
    print(f"Document types to process: {list(textract_results_by_type.keys())}")
    
    # Extract text content for each document type
    freight_invoice_text = ""
    air_way_bill_text = ""
    commercial_invoices_text = ""
    
    # Process each document type and extract the appropriate text
    logger.info("=== DOCUMENT TEXT EXTRACTION DEBUG ===")
    logger.info(f"Processing {len(textract_results_by_type)} document types from textract_results_by_type")
    
    for doc_type, text_content in textract_results_by_type.items():
        logger.info(f"Processing document type: '{doc_type}'")
        
        if text_content and 'error' not in text_content:
            # Combine textract data to get the actual text content
            if isinstance(text_content, dict):
                # Extract text from textract results
                combined_text_content = combine_textract_data(text_content)
                logger.info(f"  - Textract data type: dict with keys: {list(text_content.keys())}")
            else:
                combined_text_content = str(text_content)
                logger.info(f"  - Textract data type: {type(text_content)}")
            
            # Log text preview for debugging
            text_preview = combined_text_content[:200] + "..." if len(combined_text_content) > 200 else combined_text_content
            logger.info(f"  - Text preview: {text_preview}")
            
            # Map to the correct placeholder based on document type
            if 'freight' in doc_type.lower() and 'invoice' in doc_type.lower():
                freight_invoice_text = combined_text_content
                logger.info(f"✅ MAPPED TO FREIGHT_INVOICE_TEXT: {len(combined_text_content)} characters")
                print(f"✅ FREIGHT INVOICE TEXT EXTRACTED: {len(combined_text_content)} chars")
            elif 'air' in doc_type.lower() and 'waybill' in doc_type.lower():
                air_way_bill_text = combined_text_content
                logger.info(f"✅ MAPPED TO AIR_WAY_BILL_TEXT: {len(combined_text_content)} characters")
                print(f"✅ AIR WAYBILL TEXT EXTRACTED: {len(combined_text_content)} chars")
            elif 'commercial' in doc_type.lower() and 'invoice' in doc_type.lower():
                commercial_invoices_text = combined_text_content
                logger.info(f"✅ MAPPED TO COMMERCIAL_INVOICES_TEXT: {len(combined_text_content)} characters")
                print(f"✅ COMMERCIAL INVOICE TEXT EXTRACTED: {len(combined_text_content)} chars")
            else:
                logger.warning(f"⚠️  UNMAPPED DOCUMENT TYPE: '{doc_type}' - text not assigned to any placeholder")
                print(f"⚠️  UNMAPPED DOCUMENT TYPE: '{doc_type}' - {len(combined_text_content)} chars")
        else:
            logger.warning(f"❌ No valid text content for document type: '{doc_type}'")
            print(f"❌ No valid text content for document type: '{doc_type}'")
    
    # Log final text assignments
    logger.info("=== FINAL TEXT ASSIGNMENTS ===")
    logger.info(f"freight_invoice_text: {len(freight_invoice_text)} characters")
    logger.info(f"air_way_bill_text: {len(air_way_bill_text)} characters") 
    logger.info(f"commercial_invoices_text: {len(commercial_invoices_text)} characters")
    
    print("=== FINAL TEXT ASSIGNMENTS ===")
    print(f"freight_invoice_text: {len(freight_invoice_text)} characters")
    print(f"air_way_bill_text: {len(air_way_bill_text)} characters")
    print(f"commercial_invoices_text: {len(commercial_invoices_text)} characters")
    
    # Get carrier-specific prompt template from S3
    if not carrier_name:
        logger.error("Carrier name is required for extraction")
        raise ValueError("Carrier name is required for extraction")
    
    # For separated processing, we'll use the freight invoice template and schema as default
    # since this function processes multiple document types together
    prompt_template = get_prompt_template(carrier_name, "freight_invoice")
    if not prompt_template:
        logger.error(f"No prompt template found for carrier: {carrier_name}")
        raise ValueError(f"No prompt template found for carrier: {carrier_name}")
    
    logger.info(f"Using carrier-specific prompt template for: {carrier_name} (freight_invoice)")
    
    # Handle template formatting with error handling
    logger.info("=== PROMPT TEMPLATE FORMATTING ===")
    logger.info(f"Template length: {len(prompt_template)} characters")
    
    try:
        # Try new freight template structure with multiple text placeholders
        # Pass the correct text to each placeholder
        logger.info("Formatting prompt with separate text for each document type...")
        logger.info(f"  - freight_invoice_text placeholder: {len(freight_invoice_text)} chars")
        logger.info(f"  - air_way_bill_text placeholder: {len(air_way_bill_text)} chars")
        logger.info(f"  - commercial_invoices_text placeholder: {len(commercial_invoices_text)} chars")
        
        prompt = prompt_template.format(
            freight_invoice_text=freight_invoice_text,
            air_way_bill_text=air_way_bill_text,
            commercial_invoices_text=commercial_invoices_text
        )
        
        logger.info(f"✅ Successfully formatted prompt with separate text for each document type")
        print("✅ PROMPT FORMATTED SUCCESSFULLY with separate text for each placeholder")
        
        # Log a preview of the formatted prompt to verify text placement
        logger.info("=== FORMATTED PROMPT PREVIEW ===")
        prompt_preview = prompt[:500] + "..." if len(prompt) > 500 else prompt
        logger.info(f"Formatted prompt preview: {prompt_preview}")
        
    except KeyError as e:
        logger.warning(f"❌ Template placeholder error: {e}. Trying fallback formatting.")
        print(f"❌ TEMPLATE PLACEMENT ERROR: {e} - Using fallback formatting")
        
        # Ultimate fallback - replace common placeholders
        prompt = prompt_template.replace('{pdf_text}', freight_invoice_text)
        prompt = prompt.replace('{freight_invoice_text}', freight_invoice_text)
        prompt = prompt.replace('{air_way_bill_text}', air_way_bill_text)
        prompt = prompt.replace('{commercial_invoices_text}', commercial_invoices_text)
        
        logger.info("✅ Fallback formatting completed")
        print("✅ FALLBACK FORMATTING COMPLETED")
    
    # Verify text placement in the final prompt
    logger.info("=== FINAL PROMPT VERIFICATION ===")
    logger.info(f"Final prompt length: {len(prompt)} characters")
    
    # Check if the text sections are properly placed
    if "Invoice Text:" in prompt and freight_invoice_text:
        logger.info("✅ Freight invoice text section found in prompt")
        print("✅ FREIGHT INVOICE TEXT SECTION FOUND in prompt")
    else:
        logger.warning("⚠️  Freight invoice text section not found or empty in prompt")
        print("⚠️  FREIGHT INVOICE TEXT SECTION NOT FOUND or empty in prompt")
    
    if "Air Way Bill (AWB) Text:" in prompt and air_way_bill_text:
        logger.info("✅ Air waybill text section found in prompt")
        print("✅ AIR WAYBILL TEXT SECTION FOUND in prompt")
    else:
        logger.warning("⚠️  Air waybill text section not found or empty in prompt")
        print("⚠️  AIR WAYBILL TEXT SECTION NOT FOUND or empty in prompt")
    
    if "Commercial Invoice Text:" in prompt and commercial_invoices_text:
        logger.info("✅ Commercial invoice text section found in prompt")
        print("✅ COMMERCIAL INVOICE TEXT SECTION FOUND in prompt")
    else:
        logger.warning("⚠️  Commercial invoice text section not found or empty in prompt")
        print("⚠️  COMMERCIAL INVOICE TEXT SECTION NOT FOUND or empty in prompt")
    
    # Save the extracted text files for debugging (for separated processing)
    # Note: This saves the combined text for all document types
    combined_debug_text = f"Freight Invoice:\n{freight_invoice_text}\n\nAir Way Bill:\n{air_way_bill_text}\n\nCommercial Invoice:\n{commercial_invoices_text}"
    # Use the same naming convention as individual processing
    save_extracted_text_files(combined_debug_text, "freight_invoice", carrier_name, main_pdf_name or "separated_documents")
    
    logger.info("=== ATTACHMENT TEXT EXTRACTION COMPLETE ===")
    print("=== ATTACHMENT TEXT EXTRACTION COMPLETE ===")
    
    # Use freight invoice schema for combined processing
    schema = get_document_schema("freight_invoice")
    
    for attempt in range(max_retries + 1):
        try:
            if attempt > 0:
                logger.info(f"Retry attempt {attempt + 1}/{max_retries + 1} for separated extraction")
                time.sleep(1)
            
            # Use converse model with tool schema for structured output
            response = bedrock.converse(
                modelId=BEDROCK_MODEL_ID,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                toolConfig={
                    "tools": [{
                        "toolSpec": {
                            "name": "validate_freight_invoice_data",
                            "description": "Extract and validate freight invoice fields.",
                            "inputSchema": {"json": schema}
                        }
                    }],
                },
                additionalModelRequestFields={
                    "reasoning_config": {
                        "type": "enabled",
                        "budget_tokens": 2000
                    },
                    "max_tokens": 100000, 
                }
            )
            
            # Parse response from converse model
            logger.info(f"Claude response received (size: {len(json.dumps(response))} bytes)")
            
            # Cost Tracking
            try:
                usage = response.get('usage', {})
                input_tokens = usage.get('inputTokens', 0)
                output_tokens = usage.get('outputTokens', 0)
                total_cost = input_tokens * 0.000003 + output_tokens * 0.000015
                logger.info(f"Claude tokens used: input={input_tokens}, output={output_tokens}, cost=${total_cost:.6f}")
            except Exception as e:
                logger.warning(f"Failed to extract cost info: {e}")

            # Extract structured_output from tool use
            structured_output = None
            for msg in response.get("content", []):
                tool_data = msg.get("toolUse", {})
                if "response" in tool_data:
                    structured_output = tool_data["response"]
                    break
                # Alternative: check if the tool use data is directly in input
                elif "input" in tool_data:
                    structured_output = tool_data["input"]
                    logger.info("Using alternative tool response extraction from input field")
                    break

            if not structured_output:
                try:
                    # Fallback output structure
                    content_list = response.get("output", {}).get("message", {}).get("content", [])
                    for msg in content_list:
                        if "toolUse" in msg and "input" in msg["toolUse"]:
                            structured_output = msg["toolUse"]["input"]
                            break
                except Exception as fallback_err:
                    logger.warning(f"Error in fallback structure: {fallback_err}")

            if not structured_output:
                logger.error("No structured output received from Claude")
                raise ValueError("No structured output received from Claude")
            
            # Validate the LLM output structure and required fields
            try:
                validation_valid, validation_errors = validate_llm_output(structured_output, schema, "freight_invoice")
                
                if not validation_valid:
                    error_summary = f"Validation failed with {len(validation_errors)} errors. "
                    error_summary += f"First 3 errors: {', '.join(validation_errors[:3])}"
                    logger.warning(f"❌ LLM output validation failed: {error_summary}")
                    
                    # Retry if we have attempts left - ONLY retry for actual validation failures
                    if attempt < max_retries:
                        logger.warning(f"Retrying extraction due to validation failure (attempt {attempt + 1}/{max_retries + 1})")
                        continue
                    else:
                        logger.error(f"❌ Validation failed after {max_retries + 1} attempts. Returning data anyway but with validation errors.")
                        # Log all validation errors for debugging
                        for i, error in enumerate(validation_errors[:20], 1):  # Log first 20 errors
                            logger.error(f"  Validation Error {i}: {error}")
                        if len(validation_errors) > 20:
                            logger.error(f"  ... and {len(validation_errors) - 20} more validation errors")
                else:
                    logger.info(f"✅ LLM output validation passed successfully")
                    # Validation passed - proceed normally, no retry needed
            except RecursionError as rec_err:
                logger.error(f"Recursion error during validation (not a validation failure): {rec_err}")
                logger.info("Validation code error occurred - skipping validation and proceeding (no retry)")
                # Don't retry for validation code errors - just proceed with the data
            except Exception as val_err:
                logger.error(f"Exception during validation (not a validation failure): {val_err}")
                logger.info("Validation code error occurred - skipping validation and proceeding (no retry)")
                # Don't retry for validation code errors - just proceed with the data
            
            # Validate and correct city names
            structured_output, city_validation_passed = validate_and_correct_cities(structured_output, "freight_invoice")
            
            # Retry if city validation failed and we have attempts left
            if not city_validation_passed:
                if attempt < max_retries:
                    logger.warning(f"City validation failed. Retrying extraction (attempt {attempt + 1}/{max_retries + 1})")
                    time.sleep(1)
                    continue
                else:
                    logger.error(f"❌ City validation failed after {max_retries + 1} attempts. Returning data anyway but with invalid cities.")
            
            logger.info("Successfully extracted structured information from separated documents")
            return structured_output
            
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                logger.warning(f"JSON parsing error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(1)
                continue
            else:
                logger.error(f"Failed to parse JSON after {max_retries + 1} attempts: {e}")
                raise ValueError(f"Failed to parse JSON from Claude response: {e}")
        
        except Exception as e:
            if attempt < max_retries:
                logger.warning(f"Error on attempt {attempt + 1}: {e}. Retrying...")
                time.sleep(1)
                continue
            else:
                logger.error(f"Error in Claude processing after {max_retries + 1} attempts: {e}")
                raise
    
    # Should never reach here but just in case
    raise ValueError("Failed to extract information after multiple attempts")

def process_individual_clustered_document(s3_bucket: str, s3_key: str, cluster_info: dict, carrier_name: str, main_pdf_name: str, existing_textract_data: dict = None) -> dict:
    """
    Process an individual clustered document with separate LLM converse call.
    
    Args:
        s3_bucket: S3 bucket containing the clustered PDF
        s3_key: S3 key of the clustered PDF
        cluster_info: Information about the cluster
        carrier_name: Carrier name for processing
        main_pdf_name: Main PDF name for organizing results
        existing_textract_data: Optional existing Textract data to reuse (for freight invoices)
    
    Returns:
        dict: Processing results for this document
    """
    try:
        logger.info(f"Processing individual clustered document: {s3_key}")
        # Skip unknown document types
        doc_type_check = (cluster_info.get('doc_type') or '').lower()
        if 'unknown' in doc_type_check:
            logger.info("Skipping unknown document type")
            return {
                'success': False,
                'skipped': True,
                'reason': 'Unknown document type',
                'doc_type': cluster_info.get('doc_type'),
                's3_path': f"s3://{s3_bucket}/{s3_key}"
            }
        
        # Define bucket and key variables at the start
        bucket_name = s3_bucket
        key_name = s3_key
        
        # Check if we have existing Textract data to reuse (for freight invoices)
        if existing_textract_data and existing_textract_data.get('method') == 'textract':
            logger.info("Reusing existing Textract data for freight invoice processing")
            kvs = existing_textract_data.get('key_value_pairs', {})
            tables = existing_textract_data.get('tables', [])
            lines = existing_textract_data.get('text_lines', [])
            page_count = existing_textract_data.get('page_count', 0)
        else:
            # Extract text from the clustered PDF using Textract
            
            logger.info(f"Extracting text from clustered PDF: s3://{bucket_name}/{key_name}")
            kvs, tables, lines, page_count = extract_text_from_pdf(bucket_name, key_name)
        
        # Check if extraction was successful
        if not lines and not tables and not kvs:
            logger.warning(f"Textract could not extract any content from clustered PDF: s3://{bucket_name}/{key_name}")
            return {
                'success': False,
                'error': 'No content could be extracted from the clustered document',
                's3_path': f"s3://{bucket_name}/{key_name}"
            }
        
        # Create textract data object
        textract_data = {
            "key_value_pairs": kvs,
            "tables": tables,
            "text_lines": lines,
            "page_count": page_count,
            "method": "textract"
        }
        
        # Format extracted data for the prompt
        def table_to_markdown(table):
            if not table or not table[0]:
                return ""
            col_count = len(table[0])
            cleaned_table = []
            for row in table:
                cleaned_row = [(cell if cell is not None else "").strip() for cell in row]
                if len(cleaned_row) < col_count:
                    cleaned_row += [""] * (col_count - len(cleaned_row))
                cleaned_table.append(cleaned_row)
            header = "| " + " | ".join(cleaned_table[0]) + " |"
            separator = "| " + " | ".join(["---"] * col_count) + " |"
            rows = ["| " + " | ".join(row) + " |" for row in cleaned_table[1:]]
            return "\n".join([header, separator] + rows)
        
        # Prepare formatted text but limit it to a reasonable size
        formatted_kvs = "\n".join(f"{key}: {value}" for key, value in kvs.items() if value.strip())
        
        # Limit number of tables to prevent prompt size issues
        tables_to_format = tables[:min(5, len(tables))]  # At most 5 tables
        formatted_tables = "\n\n".join(table_to_markdown(tbl) for tbl in tables_to_format)
        
        # Limit number of text lines to prevent prompt size issues
        max_lines = 500  # Limit to 500 lines
        if len(lines) > max_lines:
            selected_lines = lines[:max_lines//2] + lines[-max_lines//2:]
            formatted_lines = "\n".join(selected_lines)
            logger.info(f"Limited OCR text to {max_lines} lines (out of {len(lines)} total)")
        else:
            formatted_lines = "\n".join(lines)
        
        raw_text = "\n\n".join(filter(None, [
            "Key-Value Pairs:\n" + formatted_kvs if formatted_kvs else None,
            "Tables:\n" + formatted_tables if formatted_tables else None,
            "Text Lines:\n" + formatted_lines if formatted_lines else None
        ]))
        
        # Extract structured information using Claude with schema validation
        doc_type = cluster_info.get('doc_type', 'Unknown')
        logger.info(f"Extracting information from {doc_type} document using Claude")
        extracted_info = extract_information_with_claude(raw_text, kvs, carrier_name, doc_type, main_pdf_name)
        
        # Simplify the extracted data
        simplified_data = simplify_extracted_info(extracted_info)
        
        # Save results to S3 with document-specific naming
        output_location = save_document_results_to_s3(
            simplified_data, 
            cluster_info, 
            main_pdf_name, 
            textract_data
        )
        
        logger.info(f"Successfully processed clustered document: {cluster_info.get('doc_type', 'Unknown')}")
        
        return {
            'success': True,
            'cluster_id': cluster_info.get('cluster_id'),
            'doc_type': cluster_info.get('doc_type'),
            'extracted_info': simplified_data,
            'textract_data': textract_data,
            'page_count': page_count,
            's3_path': f"s3://{bucket_name}/{key_name}",
            'output_location': output_location
        }
        
    except Exception as e:
        logger.error(f"Error processing individual clustered document: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e),
            's3_path': f"s3://{bucket_name}/{key_name}" if 'bucket_name' in locals() and 'key_name' in locals() else s3_key
        }

def save_extracted_text_files(text: str, document_type: str, carrier_name: str, main_pdf_name: str) -> None:
    """
    Save the extracted text files that are fed into prompt templates for debugging.
    
    Args:
        text: The extracted text content
        document_type: Type of document (e.g., 'freight_invoice', 'air_waybill')
        carrier_name: Name of the carrier (e.g., 'KWE', 'MAGNO')
        main_pdf_name: Name of the main PDF file
    """
    try:
        # Create clean names for S3 keys
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
        # Determine the text file name based on document type
        doc_type_key = document_type.lower().replace(' ', '_') if document_type else 'unknown'
        text_filename_map = {
            'freight_invoice': 'freight_invoice_text.txt',
            'freight_invoices': 'freight_invoice_text.txt',
            'air_waybill': 'air_way_bill_text.txt',
            'air_waybills': 'air_way_bill_text.txt',
            'commercial_invoice': 'commercial_invoices_text.txt',
            'commercial_invoices': 'commercial_invoices_text.txt',
        }
        
        text_filename = text_filename_map.get(doc_type_key)
        if not text_filename:
            # Fallback: use document type for filename
            fallback_name = re.sub(r'[^a-zA-Z0-9]', '_', doc_type_key)
            text_filename = f"{fallback_name}_text.txt"
        
        # Create the S3 key for the text file
        text_key = f"Output/{clean_main_name}/AWS_extracted_text/{text_filename}"
        
        # Save the text to S3
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=text_key,
            Body=text.encode('utf-8'),
            ContentType="text/plain"
        )
        
        logger.info(f"Saved extracted text to s3://{OUTPUT_BUCKET}/{text_key}")
        
    except Exception as e:
        logger.error(f"Error saving extracted text files: {str(e)}")
        # Don't raise the exception as this is just for debugging

def save_document_results_to_s3(data: dict, cluster_info: dict, main_pdf_name: str, textract_data: dict = None) -> str:
    """
    Save extracted information for a specific document to S3 with document-specific naming.
    
    Args:
        data: Extracted data
        cluster_info: Cluster information
        main_pdf_name: Main PDF name
        textract_data: Textract data
    
    Returns:
        str: S3 URI of the saved file
    """

    try:
        # Create clean names for S3 keys
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
        doc_type = cluster_info.get('doc_type', 'Unknown')
        
        # Standardize document type to get proper filename
        doc_type_key = doc_type.lower().replace(' ', '_').replace('-', '_') if isinstance(doc_type, str) else 'unknown'
        
        # Map document types to filenames
        filename_map = {
            'freight_invoice': 'freight.json',
            'freight_invoices': 'freight.json',
            'air_waybill': 'airwaybill.json',
            'air_waybills': 'airwaybill.json',
            'airway_bill': 'airwaybill.json',
            'airway_bills': 'airwaybill.json',
            'sea_waybill': 'airwaybill.json',
            'sea_waybills': 'airwaybill.json',
            'commercial_invoice': 'commercial.json',
            'commercial_invoices': 'commercial.json',
            'packing_list': 'packing.json',
            'packing_lists': 'packing.json',
            'delivery_note': 'delivery.json',
            'delivery_notes': 'delivery.json'
        }
        
        file_name = filename_map.get(doc_type_key)
        
        if not file_name:
            # Fallback: remove non-alphanumerics from doc type for filename
            fallback_name = re.sub(r'[^a-zA-Z0-9]', '', doc_type.lower()) if isinstance(doc_type, str) else 'document'
            file_name = f"{fallback_name}.json"
            logger.info(f"Using fallback filename: {file_name} for document type: {doc_type}")

        output_key = f"Output/{clean_main_name}/Output_json/{file_name}"
        logger.info(f"Output path: {output_key}")
        
        # Create metadata with processing info
        metadata = {
            "original_file": main_pdf_name,
            "document_type": doc_type,
            "cluster_id": cluster_info.get('cluster_id'),
            "processed_time": datetime.now().isoformat(),
            "extraction_service": "textract-claude-individual"
        }
        
        # Create a simplified output structure that only includes the schema-compliant data
        simplified_output = {}
        
        # Extract only the document-specific data by looking for standard document type keys
        standard_doc_keys = [
            'packing_list', 'commercial_invoice', 'air_waybill', 'airway_bill', 
            'freight_invoice', 'delivery_note', 'bill_of_lading'
        ]
        
        # Find the document key in the data
        found_key = None
        for key in standard_doc_keys:
            if key in data:
                found_key = key
                simplified_output[key] = data[key]
                break
        
        # If no standard key found, try a case-insensitive match
        if not found_key:
            for key in data:
                key_lower = key.lower() if isinstance(key, str) else ''
                for std_key in standard_doc_keys:
                    if std_key in key_lower:
                        simplified_output[key] = data[key]
                        found_key = key
                        break
                if found_key:
                    break
        
        # If still nothing found, use the entire data object but remove additional_info
        if not simplified_output and isinstance(data, dict):
            simplified_output = data.copy()
            if 'additional_info' in simplified_output:
                del simplified_output['additional_info']
            
        logger.info(f"Created simplified output with only schema-compliant data")
        
        # Combine metadata and simplified data
        output_data = {
            "metadata": metadata,
            "extracted_data": simplified_output
        }
        
        # Save to S3
        s3.put_object(
            Bucket=OUTPUT_BUCKET,
            Key=output_key,
            Body=json.dumps(output_data, indent=2),
            ContentType="application/json"
        )
        
        logger.info(f"Saved document results to s3://{OUTPUT_BUCKET}/{output_key}")
        return f"s3://{OUTPUT_BUCKET}/{output_key}"
        
    except Exception as e:
        logger.error(f"Error saving document results to S3: {str(e)}")
        raise

    # try:
    #     # Create clean names for S3 keys
    #     clean_main_name = os.path.splitext(main_pdf_name)[0]
    #     clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
    #     doc_type = cluster_info.get('doc_type', 'Unknown')
    #     safe_doc_type = re.sub(r'[\\/*?:"<>|]', "_", doc_type)
        
    #     # Create output path per requirement:
    #     # pando-j-and-j-output/extraction/{pdf_base}/{document}.json
    #     # Where document names are normalized as:
    #     #  - freight_invoice -> freight.json
    #     #  - air_waybill -> airwaybill.json
    #     #  - commercial_invoice -> commercial.json
    #     #  - others -> normalized doc type without spaces/underscores + .json
    #     doc_type_key = doc_type.lower().replace(' ', '_') if isinstance(doc_type, str) else 'unknown'
    #     filename_map = {
    #         'freight_invoice': 'freight.json',
    #         'freight_invoices': 'freight.json',
    #         'air_waybill': 'airwaybill.json',
    #         'air_waybills': 'airwaybill.json',
    #         'airway_bill': 'airwaybill.json',  # Add this line 
    #         'airway_bills': 'airwaybill.json', # Add this line
    #         'sea_waybills':'airwaybill.json',
    #         'commercial_invoice': 'commercial.json',
    #         'commercial_invoices': 'commercial.json',
    #         'delivery_note':'delivery.json',
    #         'delivery_note':'delivery.json'
    #     }
    #     file_name = filename_map.get(doc_type_key)
    #     if not file_name:
    #         # Fallback: remove non-alphanumerics from doc type for filename
    #         fallback_name = re.sub(r'[^a-zA-Z0-9]', '', doc_type.lower()) if isinstance(doc_type, str) else 'document'
    #         file_name = f"{fallback_name}.json"

    #     output_key = f"extraction/{clean_main_name}/{file_name}"
        
    #     # Create metadata with processing info
    #     metadata = {
    #         "original_file": main_pdf_name,
    #         "document_type": doc_type,
    #         "cluster_id": cluster_info.get('cluster_id'),
    #         "processed_time": datetime.now().isoformat(),
    #         "extraction_service": "textract-claude-individual"
    #     }
        
    #     # Combine data and metadata
    #     output_data = {
    #         "metadata": metadata,
    #         "extracted_data": data,
    #         "extracted_data_full": data,  # Include full data for reference
    #         "textract_data": textract_data,  # Include raw Textract data
    #         "cluster_info": cluster_info
    #     }
        
    #     # Save to S3
    #     s3.put_object(
    #         Bucket=OUTPUT_BUCKET,
    #         Key=output_key,
    #         Body=json.dumps(output_data, indent=2),
    #         ContentType="application/json"
    #     )
        
    #     logger.info(f"Saved document results to s3://{OUTPUT_BUCKET}/{output_key}")
    #     return f"s3://{OUTPUT_BUCKET}/{output_key}"
        
    # except Exception as e:
    #     logger.error(f"Error saving document results to S3: {str(e)}")
    #     raise

def save_results_locally_separated(data: dict, input_file: str, textract_results_by_type: dict, carrier_name: str) -> str:
    """
    Save extraction results to a local JSON file with separated document data.
    """
    try:
        # Create output directory if it doesn't exist
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{base_name}_extracted_separated_{timestamp}.json"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Prepare the data to save
        result_data = {
            "extraction_timestamp": datetime.now().isoformat(),
            "input_file": input_file,
            "carrier": carrier_name,
            "extracted_data": data,
            "textract_results_by_document_type": textract_results_by_type,
            "processing_method": "separated_document_processing"
        }
        
        # Save to local file
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Results saved locally to: {output_path}")
        return output_path
        
    except Exception as e:
        logger.error(f"Error saving results locally: {str(e)}")
        raise

# ---------- EVENT PROCESSING FUNCTION ----------
def process_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an event (either S3 event or custom event format) and return the result.
    This function can be called when importing J_J.py as a module.
    
    Args:
        event: Event dictionary containing either S3 event structure or custom parameters
        
    Returns:
        Dictionary containing the processing result
    """
    try:
        # Log the event type
        logger.info(f"Processing event type: {type(event)}")
        logger.info(f"Event content: {json.dumps(event, default=str)}")
        
        # Check if this is an S3 event or custom event format
        if 'Records' in event and len(event['Records']) > 0:
            # Handle S3 event format
            logger.info("Processing S3 event format")
            record = event['Records'][0]
            s3_info = record['s3']
            input_bucket = s3_info['bucket']['name']
            
            # Properly decode URL-encoded key
            input_key = unquote_plus(s3_info["object"]["key"])
            logger.info(f"Processing file: {input_key} from bucket: {input_bucket}")
            
            # Check if the file is in the target folder (if specified)
            if TARGET_FOLDER and not input_key.startswith(TARGET_FOLDER):
                logger.info(f"Skipping file in non-target folder: {input_key}")
                return {
                    'success': False,
                    'message': f'File not in target folder {TARGET_FOLDER}, skipping processing'
                }
            
            # Log file metadata
            try:
                file_metadata = s3.head_object(Bucket=input_bucket, Key=input_key)
                logger.info(f"File metadata: Size={file_metadata['ContentLength']} bytes, "
                           f"Type={file_metadata.get('ContentType', 'unknown')}, "
                           f"Last Modified={file_metadata.get('LastModified', 'unknown')}")
            except Exception as meta_error:
                logger.warning(f"Could not retrieve file metadata: {str(meta_error)}")
            
            # Process the invoice
            result = process_invoice(input_bucket, input_key)
            
        else:
            # Handle custom event format with provided parameters
            logger.info("Processing custom event format with provided parameters")
            try:
                input_bucket = event['input_bucket']
                input_key = event['input_key']
                output_bucket = event.get('output_bucket', input_bucket)  # Default to input_bucket if not provided
                output_prefix = event.get('output_prefix', 'invoice/output/')
                email_details = event.get('email_details', {})
                job_id = event.get('job_id', 'unknown')
                attachment_id = event.get('attachment_id', 'unknown')
                
                logger.info(f"Using provided parameters: bucket={input_bucket}, key={input_key}")
                logger.info(f"Output bucket: {output_bucket}, output prefix: {output_prefix}")
                logger.info(f"Email details: to={email_details.get('to')}, subject={email_details.get('subject')}")
                logger.info(f"Job ID: {job_id}, Attachment ID: {attachment_id}")
                
                # Log file metadata
                try:
                    file_metadata = s3.head_object(Bucket=input_bucket, Key=input_key)
                    logger.info(f"File metadata: Size={file_metadata['ContentLength']} bytes, "
                               f"Type={file_metadata.get('ContentType', 'unknown')}, "
                               f"Last Modified={file_metadata.get('LastModified', 'unknown')}")
                except Exception as meta_error:
                    logger.warning(f"Could not retrieve file metadata: {str(meta_error)}")
                
                # Process the invoice
                result = process_invoice(input_bucket, input_key)
                
            except (KeyError, IndexError) as e:
                # If not from S3 event, use provided parameters
                logger.info("Not an S3 event, using provided parameters")
                input_bucket = event['input_bucket']
                input_key = event['input_key']
                output_bucket = event['output_bucket']
                output_prefix = event['output_prefix']
                email_details = event['email_details']
                email_id = event['job_id']
                attachment_id = event['attachment_id']
                logger.info(f"Using provided parameters: bucket={input_bucket}, key={input_key}")
                logger.info(f"Email details: to={email_details.get('to')}, subject={email_details.get('subject')}")
                logger.info(f"Job ID: {email_id}, Attachment ID: {attachment_id}")
                
                # Log file metadata
                try:
                    file_metadata = s3.head_object(Bucket=input_bucket, Key=input_key)
                    logger.info(f"File metadata: Size={file_metadata['ContentLength']} bytes, "
                               f"Type={file_metadata.get('ContentType', 'unknown')}, "
                               f"Last Modified={file_metadata.get('LastModified', 'unknown')}")
                except Exception as meta_error:
                    logger.warning(f"Could not retrieve file metadata: {str(meta_error)}")
                
                # Process the invoice
                result = process_invoice(input_bucket, input_key)
        
        return result
        
    except KeyError as e:
        # Handle missing keys in event structure
        logger.error(f"KeyError in event structure: {str(e)}")
        return {
            'success': False,
            'error': f"Invalid event structure: {str(e)}"
        }
    except Exception as e:
        logger.error(f"Error in event processing: {str(e)}", exc_info=True)  # Include stack trace
        return {
            'success': False,
            'error': str(e)
        }

# ---------- LAMBDA HANDLER ----------
def lambda_handler(event: Dict[str, Any], context: Any):
    """AWS Lambda handler function that can handle both S3 events and custom event format."""
    try:
        # Use the process_event function to handle the event
        result = process_event(event)
        
        return {
            'statusCode': 200 if result['success'] else 400,
            'body': json.dumps(result)
        }
        
    except Exception as e:
        logger.error(f"Error in lambda handler: {str(e)}", exc_info=True)  # Include stack trace
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

# ---------- MAIN EXECUTION FOR LOCAL RUNNING ----------
if __name__ == "__main__":
    """
    Main execution section for running the script locally.
    
    Usage:
    1. Run with hardcoded values: python J_J.py
    2. Override file path: python J_J.py <file_path>
    3. Override file path and carrier: python J_J.py <file_path> <carrier>
    4. Process event: python J_J.py --event <event_json_file>
    """
    
    # ===== HARDCODED VALUES - UPDATE THESE =====
    INPUT_FILE = DEFAULT_INPUT_FILE  # Default local PDF file to process
    CARRIER_NAME = None  # Will be auto-detected from freight invoices using LLM
    USE_TEXTRACT = True  # Use Textract (True) or fallback method (False)
    USE_S3_CLUSTERING = True  # Use S3-based clustering and separate processing
    CLUSTERING_S3_BUCKET = OUTPUT_BUCKET  # S3 bucket for storing clustered PDFs and results
    # ===========================================
    
    # Check if processing an event file
    if len(sys.argv) >= 3 and sys.argv[1] == '--event':
        event_file = sys.argv[2]
        print(f"Processing event from file: {event_file}")
        
        try:
            with open(event_file, 'r') as f:
                event = json.load(f)
            
            print("Event loaded successfully:")
            print(json.dumps(event, indent=2))
            print("=" * 50)
            
            # Process the event
            result = process_event(event)
            
            print("\nProcessing Result:")
            print("=" * 50)
            print(json.dumps(result, indent=2))
            
            if result['success']:
                print(f"\n✅ Processing completed successfully!")
            else:
                print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
                
        except FileNotFoundError:
            print(f"Error: Event file not found: {event_file}")
            sys.exit(1)
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in event file: {str(e)}")
            sys.exit(1)
        except Exception as e:
            print(f"❌ Fatal error: {str(e)}")
            sys.exit(1)
        
        # Exit after processing event
        sys.exit(0)
    
    # Parse command line arguments if provided
    if len(sys.argv) >= 2:
        INPUT_FILE = sys.argv[1]
        print(f"Using command line file: {INPUT_FILE}")
    else:
        print(f"Using hardcoded file: {INPUT_FILE}")
        print("Note: You can override with: python J_J.py <file_path> [carrier] [s3_bucket]")
        print("Or process an event with: python J_J.py --event <event_json_file>")
    
    if len(sys.argv) >= 3:
        CARRIER_NAME = sys.argv[2]
        print(f"Using command line carrier: {CARRIER_NAME}")
    else:
        print("Carrier will be auto-detected from freight invoices")
    
    if len(sys.argv) >= 4:
        CLUSTERING_S3_BUCKET = sys.argv[3]
        print(f"Using command line S3 bucket: {CLUSTERING_S3_BUCKET}")
    else:
        print(f"Using hardcoded S3 bucket: {CLUSTERING_S3_BUCKET}")
    
    # Check if file exists
    if not os.path.exists(INPUT_FILE):
        print(f"Error: File not found: {INPUT_FILE}")
        print("Please provide a valid PDF file path.")
        sys.exit(1)
    
    # Process the invoice
    print(f"\nProcessing invoice: {INPUT_FILE}")
    if USE_S3_CLUSTERING:
        print(f"Processing method: S3-based clustering with separate document processing")
        print("Features: Document clustering, S3 storage, separate LLM calls per document")
    else:
        print(f"Processing method: {'Textract with Page Extraction (preferred)' if USE_TEXTRACT else 'Fallback with Page Extraction'}")
        print("Features: Document type identification, carrier detection, separate processing")
    print("=" * 50)
    
    try:
        if USE_S3_CLUSTERING:
            result = process_invoice_with_s3_clustering(INPUT_FILE, CARRIER_NAME, USE_TEXTRACT, CLUSTERING_S3_BUCKET)
        else:
            result = process_invoice_local(INPUT_FILE, CARRIER_NAME, USE_TEXTRACT)
        
        print("\nProcessing Result:")
        print("=" * 50)
        print(json.dumps(result, indent=2))
        
        if result['success']:
            print(f"\n✅ Processing completed successfully!")
            print(f"🏢 Carrier: {result.get('carrier', 'Unknown')}")
            print(f"🔧 Processing method: {result.get('processing_method', 'Unknown')}")
            
            if USE_S3_CLUSTERING:
                print(f"📊 Total clusters: {result.get('total_clusters', 0)}")
                print(f"✅ Successful clusters: {result.get('successful_clusters', 0)}")
                print(f"❌ Failed clusters: {result.get('failed_clusters', 0)}")
                print(f"⏭️  Skipped clusters: {result.get('skipped_clusters', 0)}")
                print(f"📁 Summary report: {result.get('summary_location', 'Unknown')}")
                
                # Show individual results
                individual_results = result.get('individual_results', {})
                if individual_results:
                    print(f"\n📄 Individual Document Results:")
                    for cluster_id, cluster_result in individual_results.items():
                        if cluster_result.get('success'):
                            doc_type = cluster_result.get('doc_type', 'Unknown')
                            output_location = cluster_result.get('output_location', 'Unknown')
                            print(f"  ✅ Cluster {cluster_id} ({doc_type}): {output_location}")
                        elif cluster_result.get('skipped'):
                            doc_type = cluster_result.get('doc_type', 'Unknown')
                            reason = cluster_result.get('reason', 'Unknown reason')
                            print(f"  ⏭️  Cluster {cluster_id} ({doc_type}): {reason}")
                        else:
                            doc_type = cluster_result.get('doc_type', 'Unknown')
                            error = cluster_result.get('error', 'Unknown error')
                            print(f"  ❌ Cluster {cluster_id} ({doc_type}): {error}")
            else:
                print(f"📁 Results saved to: {result.get('output_location', 'Unknown')}")
            
            # Show clustering report
            clustering_report = result.get('clustering_report', {})
            if clustering_report:
                print(f"\n📋 Document Clustering Analysis:")
                clusters_info = clustering_report.get('clusters', {})
                for cluster_id, cluster_info in clusters_info.items():
                    doc_type = cluster_info.get('document_type', 'Unknown')
                    pages = cluster_info.get('pages', [])
                    print(f"  • Cluster {cluster_id}: {doc_type} - pages {pages}")
            
            # Show document clusters
            clusters = result.get('document_clusters', {})
            if clusters:
                print(f"\n📄 Document Processing Groups:")
                for doc_type, pages in clusters.items():
                    if pages:
                        print(f"  • {doc_type.replace('_', ' ').title()}: pages {[p+1 for p in pages]}")
            
            # Show textract results by type
            textract_results = result.get('textract_results_by_type', {})
            if textract_results:
                print(f"\n🔍 Textract Processing:")
                for doc_type, data in textract_results.items():
                    if 'error' not in data:
                        method = data.get('method', 'unknown')
                        pages = data.get('pages', [])
                        print(f"  • {doc_type.replace('_', ' ').title()}: {method} (pages {[p+1 for p in pages]})")
        else:
            print(f"\n❌ Processing failed: {result.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ Fatal error: {str(e)}")
        sys.exit(1)