import json
import logging
import boto3
from typing import Dict, Any, List, Optional
import os
import re
import pandas as pd
import time
import smtplib
import ssl
from io import BytesIO
from datetime import datetime
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from botocore.config import Config
from J_J import process_invoice_with_s3_clustering, process_invoice_local
from botocore.exceptions import EndpointConnectionError, ClientError, ReadTimeoutError

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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
REGION = os.environ.get('AWS_REGION', 'us-east-1')
DYNAMODB_TABLE = os.environ.get('DYNAMODB_TABLE', 'invoice-processing-table')
SMTP_SERVER = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
SMTP_PORT = int(os.environ.get('SMTP_PORT', '587'))
FROM_EMAIL = os.environ.get('FROM_EMAIL', 'noreply@example.com')
SMTP_SECRET_NAME = os.environ.get('SMTP_SECRET_NAME', 'smtp-credentials')
API_ENDPOINT = os.environ.get('API_ENDPOINT', '')
INTERNAL_TOKEN = os.environ.get('INTERNAL_TOKEN', '')
AUTHORIZATION_TOKEN = os.environ.get('AUTHORIZATION_TOKEN', '')

# Pando team email recipients for internal field error alerts
PANDO_TEAM_EMAILS = [
    "jeeva@pando.ai",
    "Shashank.tiwari@pando.ai",
    "sadhanand.moorthy@pando.ai"
]

# Initialize AWS clients
s3 = boto3.client('s3')
dynamodb_client = boto3.client('dynamodb')
lambda_client = boto3.client('lambda', region_name=REGION)
custom_config = Config(
    connect_timeout=30,
    read_timeout=400
)

# API Payload Schema
API_PAYLOAD_SCHEMA = {
    "invoice_number": "",
    "invoice_date": "",
    "payment_due_date": "",
    "payment_terms": "",
    "vendor_reference_id": "",
    "currency": "",
    "total_invoice_value": 0,
    "total_tax_amount": 0,
    "bill_of_lading_number": "",
    "bill_to_name": "",
    "bill_to_gst": "",
    "bill_to_address": "",
    "bill_to_phone_number": "",
    "bill_to_email": "",
    "cost_center": "",
    "billing_entity_name": "",
    "documents_attachment": [
        {
            "file_path": "",
            "bucket_name": "",
            "file_name": "",
            "file_extension": "",
            "type": ""
        }
    ],
    "document_extraction": {
        "file_path": "",
        "bucket_name": ""
    },
    "shipments": [
        {
            "shipment_number": "",
            "mode": "",
            "pro_number": "",
            "source_name": "",
            "destination_name": "",
            "source_code": "",
            "source_city": "",
            "source_state": "",
            "source_country": "",
            "source_zone": "",
            "source_zip_code": "",
            "source_address": "",
            "destination_code": "",
            "destination_city": "",
            "destination_state": "",
            "destination_country": "",
            "destination_zone": "",
            "destination_zip_code": "",
            "destination_address": "",
            "shipment_weight": 0,
            "shipment_weight_uom": "",
            "shipment_volume": 0,
            "shipment_volume_uom": "",
            "shipment_distance": 0,
            "shipment_distance_uom": "",
            "shipment_total_value": 0,
            "shipment_tax_value": 0,
            "shipment_creation_date": "",
            "charges": [
                {
                    "charge_code": "",
                    "charge_name": "",
                    "charge_gross_amount": 0,
                    "charge_tax_amount": 0,
                    "currency": ""
                }
            ],
            "port_of_loading": "",
            "origin_service_type": "",
            "destination_service_type": "",
            "port_of_discharge": "",
            "custom": {
                "service_code": "",
                "container_type": "",
                "special_handling": "",
                "unnumber_count": "",
                "data_loggercount": "",
                "container_count": "",
                "origin_terminal_handling_days_count": "",
                "destination_terminal_handling_days_count": "",
                "thermal_blanket": "",
                "uld_extra_lease_day": "",
                "hazardous_material": "",
                "actual_weight": "",
                "actual_weight_uom": "",
                "total_package": "",
                "cargo": "",
                "temperature_control": "",
                "lane_id": ""
            }
        }
    ],
    "taxes": [],
    "custom_charges": [],
    "custom": {
        "reference_number": "",
        "special_instructions": "",
        "priority": "",
        "pay_as_present": ""
    },
    "shipment_identifiers": {
        "booking_number": "",
        "container_numbers": [""]
    }
}

def remove_metadata_fields(data):
    """
    Recursively check and remove any remaining metadata fields like 'confidence', 'value', 'explanation'.
    Handles complex nested structures including arrays of objects, regardless of field order.
    
    Args:
        data: Any JSON structure (dict, list, or primitive)
        
    Returns:
        The same structure with metadata fields removed
    """
    logger.debug(f"Removing metadata fields from data of type: {type(data)}")
    
    # Base case: not a dict or list
    if not isinstance(data, (dict, list)):
        return data
        
    # Handle list case
    if isinstance(data, list):
        #logger.debug(f"Processing list with {len(data)} items")
        return [remove_metadata_fields(item) for item in data]
        
    # Handle dictionary case
    if isinstance(data, dict):
        #logger.debug(f"Processing dictionary with keys: {list(data.keys())}")
        
        # Check if this is a value container (has value and confidence/explanation)
        is_value_container = ('value' in data and 
                             ('confidence' in data or 'explanation' in data))
        
        if is_value_container:
            logger.debug("Found value container with metadata")
            # If it's a simple value container, just return the value
            value = data.get('value')
            if isinstance(value, (str, int, float, bool)) or value is None:
                return value
            else:
                # If the value itself is complex (dict/list), process it recursively
                logger.debug("Value is complex, processing recursively")
                return remove_metadata_fields(value)
        
        # Process regular dictionary
        result = {}
        for key, value in data.items():
            # Skip metadata keys
            if key in ['confidence', 'explanation']:
                logger.debug(f"Skipping metadata key: {key}")
                continue
                
            # Skip additional_info entirely
            if key == 'additional_info':
                logger.debug("Skipping additional_info field")
                continue
                
            # Special handling for charges array
            if key == 'charges' and isinstance(value, list):
                #logger.debug(f"Processing charges array with {len(value)} items")
                charges_result = []
                for i, charge in enumerate(value):
                    if isinstance(charge, dict):
                        #logger.debug(f"Processing charge item {i} with keys: {list(charge.keys())}")
                        # Process each charge item
                        flat_charge = {}
                        for charge_key, charge_value in charge.items():
                            # Skip metadata fields in charges
                            if charge_key in ['confidence', 'explanation']:
                                #logger.debug(f"Skipping metadata key in charge: {charge_key}")
                                continue
                                
                            # Skip the 'value' field if it's duplicating another field
                            if charge_key == 'value':
                                # Check if it's duplicating charge_name or charge_code
                                if (('charge_name' in charge and charge_value == charge.get('charge_name')) or
                                    ('charge_code' in charge and charge_value == charge.get('charge_code'))):
                                    logger.debug("Skipping duplicate 'value' field in charge")
                                    continue
                            
                            # Check if this is a nested value container
                            if isinstance(charge_value, dict) and 'value' in charge_value:
                                #logger.debug(f"Extracting nested value for charge key: {charge_key}")
                                flat_charge[charge_key] = charge_value.get('value')
                            else:
                                # Process recursively
                                #logger.debug(f"Processing charge key recursively: {charge_key}")
                                flat_charge[charge_key] = remove_metadata_fields(charge_value)
                            
                        charges_result.append(flat_charge)
                    else:
                        # If it's not a dict, just add it as is
                        logger.debug(f"Adding non-dict charge item: {type(charge)}")
                        charges_result.append(charge)
                        
                result[key] = charges_result
            else:
                # For all other fields, process recursively
                logger.debug(f"Processing key recursively: {key}")
                result[key] = remove_metadata_fields(value)
                
        return result
        
    # Fallback case (shouldn't reach here)
    return data


def flatten_structured_output(data):
    """
    Flatten a nested structured output into a simple key-value dictionary.
    Only keeps field names and their values, removing confidence, explanation, and other metadata.
    Works efficiently regardless of field arrangement.
    """
    logger.info(f"Starting to flatten structured output of type: {type(data)}")
    if not isinstance(data, dict):
        #logger.warning(f"Input is not a dictionary, returning as is: {data}")
        return data
        
    result = {}
    #logger.info(f"Input data has {len(data)} top-level keys: {list(data.keys())}")
    logger.info(f"Starting data flattening process")
    
    # First pass: extract all simple fields and handle special cases
    for key, value in data.items():
        logger.debug(f"Processing key: {key}, value type: {type(value)}")
        
        # Skip additional_info field
        if key == "additional_info":
            logger.debug(f"Skipping additional_info field")
            continue
            
        # Handle dict with "value" and metadata
        if isinstance(value, dict) and "value" in value:
            logger.debug(f"Found value container for {key}: {value}")
            result[key] = value["value"]
            continue
            
        # Special handling for charges when it's a string
        if key == "charges" and isinstance(value, str):
            logger.info(f"Found charges as string: {value[:100]}...")
            try:
                # Try to parse the string as JSON
                clean_json_str = value.strip()
                if clean_json_str.endswith(','):
                    clean_json_str = clean_json_str[:-1]
                    logger.debug("Removed trailing comma from charges JSON string")
                
                if ',]' in clean_json_str:
                    clean_json_str = clean_json_str.replace(',]', ']')
                    logger.debug("Fixed invalid ',]' in charges JSON string")
                
                logger.debug(f"Attempting to parse charges JSON string: {clean_json_str[:100]}...")
                charges_list = json.loads(clean_json_str)
                logger.info(f"Successfully parsed charges JSON string into list of {len(charges_list)} items")
                
                result[key] = []
                
                for i, charge_item in enumerate(charges_list):
                    logger.debug(f"Processing charge item {i}: {charge_item}")
                    if isinstance(charge_item, dict):
                        flattened_charge = {}
                        for charge_key, charge_value in charge_item.items():
                            # Skip metadata fields
                            if charge_key in ['confidence', 'explanation']:
                                continue
                                
                            # Handle nested value objects
                            if isinstance(charge_value, dict) and "value" in charge_value:
                                flattened_charge[charge_key] = charge_value["value"]
                                logger.debug(f"Extracted nested value for {charge_key}: {charge_value['value']}")
                            # Skip the 'value' field if it's duplicating another field
                            elif charge_key == 'value':
                                logger.debug(f"Skipping 'value' field in charge item {i}")
                                continue
                            else:
                                flattened_charge[charge_key] = charge_value
                        result[key].append(flattened_charge)
                
                logger.info(f"Successfully parsed charges from JSON string: {len(result[key])} charges found")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse charges string as JSON: {e}")
                logger.error(f"JSON error at position {e.pos}: {e.msg}")
                logger.error(f"Problematic JSON string: {clean_json_str[:100]}...")
                result[key] = []
                logger.warning("Setting charges to empty array due to parsing error")
            continue
                
        # Normal handling for charges as a list
        if key == "charges" and isinstance(value, list):
            logger.info(f"Processing charges array with {len(value)} items")
            result[key] = []
            for i, charge_item in enumerate(value):
                if isinstance(charge_item, dict):
                    logger.debug(f"Processing charge item {i}: {charge_item}")
                    flattened_charge = {}
                    for charge_key, charge_value in charge_item.items():
                        # Skip metadata fields
                        if charge_key in ['confidence', 'explanation']:
                            continue
                            
                        # Handle nested value objects
                        if isinstance(charge_value, dict) and "value" in charge_value:
                            flattened_charge[charge_key] = charge_value["value"]
                            logger.debug(f"Extracted nested value for {charge_key}: {charge_value['value']}")
                        # Skip the 'value' field if it's duplicating another field
                        elif charge_key == 'value':
                            logger.debug(f"Skipping 'value' field in charge item {i}")
                            continue
                        else:
                            flattened_charge[charge_key] = charge_value
                    result[key].append(flattened_charge)
            continue
            
        # Handle other lists
        if isinstance(value, list):
            logger.debug(f"Processing list for key {key} with {len(value)} items")
            result[key] = [flatten_structured_output(item) for item in value]
            continue
            
        # Handle nested dictionaries
        if isinstance(value, dict):
            logger.debug(f"Processing nested dict for key {key} with {len(value)} items")
            result[key] = flatten_structured_output(value)
            continue
            
        # Simple values pass through
        logger.debug(f"Passing through simple value for key {key}: {value}")
        result[key] = value
        
    logger.info(f"Partially flattened structured output with {len(result)} keys: {list(result.keys())}")
    
    # Final pass to ensure no metadata fields remain
    final_result = remove_metadata_fields(result)
    logger.info(f"Final flattened output has {len(final_result)} keys: {list(final_result.keys())}")
    logger.info(f"Data flattening completed successfully")
    
    return final_result

def flatten_all_data(data):
    """
    Flatten all sections and remove all metadata.
    """
    logger.info(f"Starting to flatten all data and remove metadata")
    if not isinstance(data, dict):
        return data
        
    result = {}
    
    for key, value in data.items():
        # Skip metadata section entirely
        if key == "metadata":
            logger.info(f"Skipping metadata section")
            continue
            
        # Flatten all sections
        if isinstance(value, dict):
            logger.info(f"Flattening section: {key}")
            result[key] = flatten_structured_output(value)
        else:
            result[key] = value
    
    return result

def load_json_from_s3(bucket: str, key: str) -> Dict[str, Any]:
    """
    Load JSON data from S3.
    """
    try:
        logger.info(f"Loading JSON from s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        content = response['Body'].read().decode('utf-8')
        return json.loads(content)
    except Exception as e:
        logger.error(f"Error loading JSON from S3: {e}")
        raise

def save_json_to_s3(data: Dict[str, Any], bucket: str, key: str) -> str:
    """
    Save JSON data to S3.
    """
    try:
        logger.info(f"Saving flattened JSON to s3://{bucket}/{key}")
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(data, indent=2, ensure_ascii=False),
            ContentType="application/json"
        )
        return f"s3://{bucket}/{key}"
    except Exception as e:
        logger.error(f"Error saving JSON to S3: {e}")
        raise

def consolidate_duplicate_charges(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Consolidate duplicate charges in the payload by combining charges with the same charge_name.
    For duplicate charges:
    - Sum charge_gross_amount
    - Sum charge_tax_amount
    - Keep charge_name and charge_code from the first occurrence
    - Remove duplicates
    
    Args:
        payload: The API payload dictionary
        
    Returns:
        The payload with consolidated charges (no duplicates)
    """
    logger.info("=== CONSOLIDATING DUPLICATE CHARGES ===")
    
    if not payload or not isinstance(payload, dict):
        logger.warning("Invalid payload structure for charge consolidation")
        return payload
    
    # Process the payload data array
    if "data" in payload and isinstance(payload["data"], list):
        for data_item in payload["data"]:
            if not isinstance(data_item, dict):
                continue
                
            # Process shipments
            if "shipments" in data_item and isinstance(data_item["shipments"], list):
                for shipment in data_item["shipments"]:
                    if not isinstance(shipment, dict):
                        continue
                    
                    # Process charges in each shipment
                    if "charges" in shipment and isinstance(shipment["charges"], list):
                        charges = shipment["charges"]
                        logger.info(f"Processing {len(charges)} charges in shipment")
                        
                        # Dictionary to store consolidated charges by charge_name
                        consolidated_charges = {}
                        
                        for charge in charges:
                            if not isinstance(charge, dict):
                                continue
                            
                            # Get charge_name as the key for consolidation
                            charge_name = charge.get("charge_name", "").strip()
                            
                            # Skip charges without a name
                            if not charge_name:
                                logger.warning("Found charge without charge_name, skipping consolidation")
                                continue
                            
                            # Get amounts (handle None, empty strings, or numeric values)
                            try:
                                gross_amount = float(charge.get("charge_gross_amount", 0) or 0)
                            except (ValueError, TypeError):
                                gross_amount = 0
                            
                            try:
                                tax_amount = float(charge.get("charge_tax_amount", 0) or 0)
                            except (ValueError, TypeError):
                                tax_amount = 0
                            
                            # If charge_name already exists, consolidate
                            if charge_name in consolidated_charges:
                                logger.info(f"Found duplicate charge: '{charge_name}' - consolidating amounts")
                                # Sum the amounts
                                consolidated_charges[charge_name]["charge_gross_amount"] += gross_amount
                                consolidated_charges[charge_name]["charge_tax_amount"] += tax_amount
                                logger.info(f"  Added: gross={gross_amount}, tax={tax_amount}")
                                logger.info(f"  Total: gross={consolidated_charges[charge_name]['charge_gross_amount']}, tax={consolidated_charges[charge_name]['charge_tax_amount']}")
                            else:
                                # First occurrence - keep all fields
                                consolidated_charges[charge_name] = {
                                    "charge_code": charge.get("charge_code", ""),
                                    "charge_name": charge_name,
                                    "charge_gross_amount": gross_amount,
                                    "charge_tax_amount": tax_amount,
                                    "currency": charge.get("currency", "")
                                }
                                logger.info(f"Added new charge: '{charge_name}' (gross={gross_amount}, tax={tax_amount})")
                        
                        # Replace charges list with consolidated charges
                        shipment["charges"] = list(consolidated_charges.values())
                        logger.info(f"✅ Consolidated {len(charges)} charges to {len(shipment['charges'])} unique charges in shipment")
    
    logger.info("=== COMPLETED CHARGE CONSOLIDATION ===")
    return payload

def load_excel_from_s3(bucket: str, key: str) -> Dict[str, pd.DataFrame]:
    """
    Load Excel file from S3 and return dictionary of sheets.
    """
    try:
        logger.info(f"Loading Excel file from s3://{bucket}/{key}")
        response = s3.get_object(Bucket=bucket, Key=key)
        excel_content = response['Body'].read()
        
        # Read Excel file with all sheets
        excel_file = pd.ExcelFile(BytesIO(excel_content))
        sheets = {}
        
        for sheet_name in excel_file.sheet_names:
            logger.info(f"Loading sheet: {sheet_name}")
            sheets[sheet_name] = pd.read_excel(BytesIO(excel_content), sheet_name=sheet_name)
            logger.info(f"Sheet {sheet_name} loaded with {len(sheets[sheet_name])} rows")
        
        return sheets
    except Exception as e:
        logger.error(f"Error loading Excel file from S3: {e}")
        raise

def map_charges_with_excel(freight_data: Dict[str, Any], excel_sheets: Dict[str, pd.DataFrame], vendor_reference_id: str) -> Dict[str, Any]:
    """
    Map charges in freight.json using Excel mapping based on vendor_reference_id.
    """
    try:
        logger.info(f"Starting charge mapping for vendor: {vendor_reference_id}")
        
        # Determine which sheet to use based on vendor
        sheet_name = None
        if vendor_reference_id.upper() == "KWE":
            sheet_name = "KWE_charges"
        elif vendor_reference_id.upper() == "MAGNO":
            sheet_name = "MAGNO_charges"
        else:
            logger.warning(f"Unknown vendor: {vendor_reference_id}, skipping charge mapping")
            return freight_data
        
        if sheet_name not in excel_sheets:
            logger.error(f"Sheet {sheet_name} not found in Excel file")
            return freight_data
        
        mapping_df = excel_sheets[sheet_name]
        logger.info(f"Using mapping sheet: {sheet_name} with {len(mapping_df)} rows")
        
        # Create mapping dictionary
        charge_mapping = {}
        for _, row in mapping_df.iterrows():
            invoice_charge_name = str(row.iloc[0]).strip()  # Charge Name on Invoice
            charge_code = str(row.iloc[1]).strip()          # Charge code on Pando Payload
            charge_name = str(row.iloc[2]).strip()          # Charge Name on Pando Payload
            carrier_type = str(row.iloc[3]).strip()         # Carrier Type (PC/OC)
            
            if invoice_charge_name and invoice_charge_name != 'nan':
                charge_mapping[invoice_charge_name] = {
                    'charge_code': charge_code,
                    'charge_name': charge_name,
                    'carrier_type': carrier_type
                }
                logger.info(f"📋 Excel Mapping: '{invoice_charge_name}' -> code: '{charge_code}', name: '{charge_name}', carrier_type: '{carrier_type}'")
        
        logger.info(f"📊 Created charge mapping with {len(charge_mapping)} entries from Excel")
        logger.info(f"📋 Available mappings: {list(charge_mapping.keys())}")
        
        # Apply mapping to freight data
        if 'data' in freight_data and isinstance(freight_data['data'], list):
            for data_item in freight_data['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        if 'charges' in shipment and isinstance(shipment['charges'], list):
                            logger.info(f"🔍 Processing {len(shipment['charges'])} charges in shipment")
                            for i, charge in enumerate(shipment['charges']):
                                if 'charge_name' in charge:
                                    original_charge_name = charge['charge_name']
                                    original_charge_code = charge.get('charge_code', 'N/A')
                                    
                                    logger.info(f"📝 Charge {i+1}: '{original_charge_name}' (code: '{original_charge_code}')")
                                    
                                    # Check if we have a mapping for this charge based on charge_name
                                    if original_charge_name in charge_mapping:
                                        mapping = charge_mapping[original_charge_name]
                                        old_code = charge.get('charge_code', 'N/A')
                                        old_name = charge['charge_name']
                                        charge['charge_code'] = mapping['charge_code']
                                        charge['charge_name'] = mapping['charge_name']
                                        charge['carrier_type'] = mapping['carrier_type']  # Store carrier type
                                        logger.info(f"✅ MAPPED: '{old_name}' -> '{mapping['charge_name']}' | '{old_code}' -> '{mapping['charge_code']}' | carrier_type: '{mapping['carrier_type']}'")
                                    else:
                                        logger.warning(f"❌ NO MAPPING: '{original_charge_name}' (not found in Excel)")
        
        logger.info("Charge mapping completed successfully")
        
        # Apply business rules after mapping
        freight_data = apply_business_rules(freight_data)
        
        return freight_data
        
    except Exception as e:
        logger.error(f"Error in charge mapping: {e}")
        return freight_data

def map_airports_and_locations_with_excel(freight_data: Dict[str, Any], excel_sheets: Dict[str, pd.DataFrame], vendor_reference_id: str) -> Dict[str, Any]:
    """
    Map airports, states, and countries using Excel mapping based on vendor_reference_id.
    """
    try:
        logger.info(f"Starting airport and location mapping for vendor: {vendor_reference_id}")
        
        # Apply mapping to freight data
        if 'data' in freight_data and isinstance(freight_data['data'], list):
            for data_item in freight_data['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        logger.info(f"🔍 Processing shipment for airport/location mapping")
                        
                        # 1. AIRPORT MAPPING (Same logic for both MAGNO and KWE)
                        if vendor_reference_id.upper() in ["MAGNO", "KWE"]:
                            # For both MAGNO and KWE: Search in UNLocode sheet, column 1 (Name of Airport)
                            if 'UNLocode' in excel_sheets:
                                unlocode_df = excel_sheets['UNLocode']
                                
                                # Map port_of_loading
                                port_of_loading = shipment.get('port_of_loading', '')
                                if port_of_loading:
                                    # First check if the value is already a valid 5-character airport code in column 2
                                    existing_airport_code = unlocode_df[unlocode_df.iloc[:, 1].str.upper() == port_of_loading.upper()]
                                    if not existing_airport_code.empty:
                                        logger.info(f"✅ SKIPPED port_of_loading mapping: '{port_of_loading}' is already a valid 5-character airport code")
                                    else:
                                        # Try multiple matching strategies
                                        loading_match = None
                                        
                                        # Check if it's a 3-character code (IATA code)
                                        is_3_char_code = len(port_of_loading.strip()) == 3 and port_of_loading.isalpha()
                                        
                                        if is_3_char_code:
                                            logger.info(f"🔍 Detected 3-character code for port_of_loading: '{port_of_loading}' - searching in column 3")
                                            # Strategy 1: Search in column 3 (3-character Airport Code)
                                            if unlocode_df.shape[1] >= 3:  # Check if column 3 exists
                                                code_match = unlocode_df[unlocode_df.iloc[:, 2].str.upper() == port_of_loading.upper()]
                                                if not code_match.empty:
                                                    loading_match = code_match
                                                    logger.info(f"🔍 Found 3-character code match for port_of_loading: '{port_of_loading}'")
                                            else:
                                                logger.warning(f"❌ Column 3 not found in UNLocode sheet for 3-character code mapping")
                                        else:
                                            logger.info(f"🔍 Detected airport name for port_of_loading: '{port_of_loading}' - searching in column 1")
                                            # Strategy 1: Exact match (airport name)
                                            exact_match = unlocode_df[unlocode_df.iloc[:, 0].str.lower() == port_of_loading.lower()]
                                            if not exact_match.empty:
                                                loading_match = exact_match
                                                logger.info(f"🔍 Found exact match for port_of_loading: '{port_of_loading}'")
                                            
                                            # Strategy 2: Contains match (if no exact match)
                                            if loading_match is None or loading_match.empty:
                                                loading_match = unlocode_df[unlocode_df.iloc[:, 0].str.contains(port_of_loading, case=False, na=False, regex=False)]
                                                if not loading_match.empty:
                                                    logger.info(f"🔍 Found contains match for port_of_loading: '{port_of_loading}'")
                                            
                                            # Strategy 3: Reverse contains match (Excel name in JSON value)
                                            if loading_match is None or loading_match.empty:
                                                for idx, row in unlocode_df.iterrows():
                                                    excel_name = str(row.iloc[0]).strip()
                                                    if excel_name and excel_name.lower() in port_of_loading.lower():
                                                        loading_match = unlocode_df.iloc[[idx]]
                                                        logger.info(f"🔍 Found reverse match for port_of_loading: '{port_of_loading}' contains '{excel_name}'")
                                                        break
                                        
                                        if loading_match is not None and not loading_match.empty:
                                            airport_code = loading_match.iloc[0, 1]  # Column 2: 5-character Airport Code
                                            shipment['port_of_loading'] = airport_code
                                            logger.info(f"✅ MAPPED port_of_loading: '{port_of_loading}' -> '{airport_code}'")
                                        else:
                                            logger.warning(f"❌ NO MATCH found for port_of_loading: '{port_of_loading}'")
                                
                                # Map port_of_discharge
                                port_of_discharge = shipment.get('port_of_discharge', '')
                                if port_of_discharge:
                                    # First check if the value is already a valid 5-character airport code in column 2
                                    existing_airport_code = unlocode_df[unlocode_df.iloc[:, 1].str.upper() == port_of_discharge.upper()]
                                    if not existing_airport_code.empty:
                                        logger.info(f"✅ SKIPPED port_of_discharge mapping: '{port_of_discharge}' is already a valid 5-character airport code")
                                    else:
                                        # Try multiple matching strategies
                                        discharge_match = None
                                        
                                        # Check if it's a 3-character code (IATA code)
                                        is_3_char_code = len(port_of_discharge.strip()) == 3 and port_of_discharge.isalpha()
                                        
                                        if is_3_char_code:
                                            logger.info(f"🔍 Detected 3-character code for port_of_discharge: '{port_of_discharge}' - searching in column 3")
                                            # Strategy 1: Search in column 3 (3-character Airport Code)
                                            if unlocode_df.shape[1] >= 3:  # Check if column 3 exists
                                                code_match = unlocode_df[unlocode_df.iloc[:, 2].str.upper() == port_of_discharge.upper()]
                                                if not code_match.empty:
                                                    discharge_match = code_match
                                                    logger.info(f"🔍 Found 3-character code match for port_of_discharge: '{port_of_discharge}'")
                                            else:
                                                logger.warning(f"❌ Column 3 not found in UNLocode sheet for 3-character code mapping")
                                        else:
                                            logger.info(f"🔍 Detected airport name for port_of_discharge: '{port_of_discharge}' - searching in column 1")
                                            # Strategy 1: Exact match (airport name)
                                            exact_match = unlocode_df[unlocode_df.iloc[:, 0].str.lower() == port_of_discharge.lower()]
                                            if not exact_match.empty:
                                                discharge_match = exact_match
                                                logger.info(f"🔍 Found exact match for port_of_discharge: '{port_of_discharge}'")
                                            
                                            # Strategy 2: Contains match (if no exact match)
                                            if discharge_match is None or discharge_match.empty:
                                                discharge_match = unlocode_df[unlocode_df.iloc[:, 0].str.contains(port_of_discharge, case=False, na=False, regex=False)]
                                                if not discharge_match.empty:
                                                    logger.info(f"🔍 Found contains match for port_of_discharge: '{port_of_discharge}'")
                                            
                                            # Strategy 3: Reverse contains match (Excel name in JSON value)
                                            if discharge_match is None or discharge_match.empty:
                                                for idx, row in unlocode_df.iterrows():
                                                    excel_name = str(row.iloc[0]).strip()
                                                    if excel_name and excel_name.lower() in port_of_discharge.lower():
                                                        discharge_match = unlocode_df.iloc[[idx]]
                                                        logger.info(f"🔍 Found reverse match for port_of_discharge: '{port_of_discharge}' contains '{excel_name}'")
                                                        break
                                        
                                        if discharge_match is not None and not discharge_match.empty:
                                            airport_code = discharge_match.iloc[0, 1]  # Column 2: 5-character Airport Code
                                            shipment['port_of_discharge'] = airport_code
                                            logger.info(f"✅ MAPPED port_of_discharge: '{port_of_discharge}' -> '{airport_code}'")
                                        else:
                                            logger.warning(f"❌ NO MATCH found for port_of_discharge: '{port_of_discharge}'")
                        
                        # 2. US STATES MAPPING
                        if 'US States' in excel_sheets:
                            us_states_df = excel_sheets['US States']
                            
                            # Map source_state
                            source_state = shipment.get('source_state', '')
                            if source_state:
                                # First check if the value is already a valid state code in column 2
                                existing_state_code = us_states_df[us_states_df.iloc[:, 1].str.upper() == source_state.upper()]
                                if not existing_state_code.empty:
                                    logger.info(f"✅ SKIPPED source_state mapping: '{source_state}' is already a valid state code")
                                    shipment['source_country'] = 'US'  # Set country to US if state code is valid
                                else:
                                    # Try multiple matching strategies for states
                                    state_match = None
                                    
                                    # Strategy 1: Exact match
                                    exact_match = us_states_df[us_states_df.iloc[:, 0].str.lower() == source_state.lower()]
                                    if not exact_match.empty:
                                        state_match = exact_match
                                        logger.info(f"🔍 Found exact match for source_state: '{source_state}'")
                                    
                                    # Strategy 2: Contains match
                                    if state_match is None or state_match.empty:
                                        state_match = us_states_df[us_states_df.iloc[:, 0].str.contains(source_state, case=False, na=False, regex=False)]
                                        if not state_match.empty:
                                            logger.info(f"🔍 Found contains match for source_state: '{source_state}'")
                                    
                                    # Strategy 3: Reverse contains match
                                    if state_match is None or state_match.empty:
                                        for idx, row in us_states_df.iterrows():
                                            excel_name = str(row.iloc[0]).strip()
                                            if excel_name and excel_name.lower() in source_state.lower():
                                                state_match = us_states_df.iloc[[idx]]
                                                logger.info(f"🔍 Found reverse match for source_state: '{source_state}' contains '{excel_name}'")
                                                break
                                    
                                    if state_match is not None and not state_match.empty:
                                        state_code = state_match.iloc[0, 1]  # Column 2: State Code
                                        shipment['source_state'] = state_code
                                        shipment['source_country'] = 'US'  # Set country to US if state is found
                                        logger.info(f"✅ MAPPED source_state: '{source_state}' -> '{state_code}' (country: US)")
                                    else:
                                        # No match found in US States sheet - set to empty string (not a US state)
                                        logger.warning(f"❌ NO MATCH found for source_state: '{source_state}' - setting to empty string (not a US state)")
                                        shipment['source_state'] = ""
                            
                            # Map destination_state
                            destination_state = shipment.get('destination_state', '')
                            if destination_state:
                                # First check if the value is already a valid state code in column 2
                                existing_state_code = us_states_df[us_states_df.iloc[:, 1].str.upper() == destination_state.upper()]
                                if not existing_state_code.empty:
                                    logger.info(f"✅ SKIPPED destination_state mapping: '{destination_state}' is already a valid state code")
                                    shipment['destination_country'] = 'US'  # Set country to US if state code is valid
                                else:
                                    # Try multiple matching strategies for states
                                    state_match = None
                                    
                                    # Strategy 1: Exact match
                                    exact_match = us_states_df[us_states_df.iloc[:, 0].str.lower() == destination_state.lower()]
                                    if not exact_match.empty:
                                        state_match = exact_match
                                        logger.info(f"🔍 Found exact match for destination_state: '{destination_state}'")
                                    
                                    # Strategy 2: Contains match
                                    if state_match is None or state_match.empty:
                                        state_match = us_states_df[us_states_df.iloc[:, 0].str.contains(destination_state, case=False, na=False, regex=False)]
                                        if not state_match.empty:
                                            logger.info(f"🔍 Found contains match for destination_state: '{destination_state}'")
                                    
                                    # Strategy 3: Reverse contains match
                                    if state_match is None or state_match.empty:
                                        for idx, row in us_states_df.iterrows():
                                            excel_name = str(row.iloc[0]).strip()
                                            if excel_name and excel_name.lower() in destination_state.lower():
                                                state_match = us_states_df.iloc[[idx]]
                                                logger.info(f"🔍 Found reverse match for destination_state: '{destination_state}' contains '{excel_name}'")
                                                break
                                    
                                    if state_match is not None and not state_match.empty:
                                        state_code = state_match.iloc[0, 1]  # Column 2: State Code
                                        shipment['destination_state'] = state_code
                                        shipment['destination_country'] = 'US'  # Set country to US if state is found
                                        logger.info(f"✅ MAPPED destination_state: '{destination_state}' -> '{state_code}' (country: US)")
                                    else:
                                        # No match found in US States sheet - set to empty string (not a US state)
                                        logger.warning(f"❌ NO MATCH found for destination_state: '{destination_state}' - setting to empty string (not a US state)")
                                        shipment['destination_state'] = ""
                        
                        # 3. COUNTRIES MAPPING
                        if 'Countries' in excel_sheets:
                            countries_df = excel_sheets['Countries']
                            
                            # Map source_country (only if not already set to US from state mapping)
                            source_country = shipment.get('source_country', '')
                            if source_country and source_country != 'US':
                                # First check if the value is already a valid country code in column 2
                                existing_country_code = countries_df[countries_df.iloc[:, 1].str.upper() == source_country.upper()]
                                if not existing_country_code.empty:
                                    logger.info(f"✅ SKIPPED source_country mapping: '{source_country}' is already a valid country code")
                                else:
                                    # Try multiple matching strategies for countries
                                    country_match = None
                                    
                                    # Strategy 1: Exact match
                                    exact_match = countries_df[countries_df.iloc[:, 0].str.lower() == source_country.lower()]
                                    if not exact_match.empty:
                                        country_match = exact_match
                                        logger.info(f"🔍 Found exact match for source_country: '{source_country}'")
                                    
                                    # Strategy 2: Contains match
                                    if country_match is None or country_match.empty:
                                        country_match = countries_df[countries_df.iloc[:, 0].str.contains(source_country, case=False, na=False, regex=False)]
                                        if not country_match.empty:
                                            logger.info(f"🔍 Found contains match for source_country: '{source_country}'")
                                    
                                    # Strategy 3: Reverse contains match
                                    if country_match is None or country_match.empty:
                                        for idx, row in countries_df.iterrows():
                                            excel_name = str(row.iloc[0]).strip()
                                            if excel_name and excel_name.lower() in source_country.lower():
                                                country_match = countries_df.iloc[[idx]]
                                                logger.info(f"🔍 Found reverse match for source_country: '{source_country}' contains '{excel_name}'")
                                                break
                                    
                                    if country_match is not None and not country_match.empty:
                                        country_code = country_match.iloc[0, 1]  # Column 2: Country Code
                                        shipment['source_country'] = country_code
                                        logger.info(f"✅ MAPPED source_country: '{source_country}' -> '{country_code}'")
                                    else:
                                        logger.warning(f"❌ NO MATCH found for source_country: '{source_country}'")
                            
                            # Map destination_country (only if not already set to US from state mapping)
                            destination_country = shipment.get('destination_country', '')
                            if destination_country and destination_country != 'US':
                                # First check if the value is already a valid country code in column 2
                                existing_country_code = countries_df[countries_df.iloc[:, 1].str.upper() == destination_country.upper()]
                                if not existing_country_code.empty:
                                    logger.info(f"✅ SKIPPED destination_country mapping: '{destination_country}' is already a valid country code")
                                else:
                                    # Try multiple matching strategies for countries
                                    country_match = None
                                    
                                    # Strategy 1: Exact match
                                    exact_match = countries_df[countries_df.iloc[:, 0].str.lower() == destination_country.lower()]
                                    if not exact_match.empty:
                                        country_match = exact_match
                                        logger.info(f"🔍 Found exact match for destination_country: '{destination_country}'")
                                    
                                    # Strategy 2: Contains match
                                    if country_match is None or country_match.empty:
                                        country_match = countries_df[countries_df.iloc[:, 0].str.contains(destination_country, case=False, na=False, regex=False)]
                                        if not country_match.empty:
                                            logger.info(f"🔍 Found contains match for destination_country: '{destination_country}'")
                                    
                                    # Strategy 3: Reverse contains match
                                    if country_match is None or country_match.empty:
                                        for idx, row in countries_df.iterrows():
                                            excel_name = str(row.iloc[0]).strip()
                                            if excel_name and excel_name.lower() in destination_country.lower():
                                                country_match = countries_df.iloc[[idx]]
                                                logger.info(f"🔍 Found reverse match for destination_country: '{destination_country}' contains '{excel_name}'")
                                                break
                                    
                                    if country_match is not None and not country_match.empty:
                                        country_code = country_match.iloc[0, 1]  # Column 2: Country Code
                                        shipment['destination_country'] = country_code
                                        logger.info(f"✅ MAPPED destination_country: '{destination_country}' -> '{country_code}'")
                                    else:
                                        logger.warning(f"❌ NO MATCH found for destination_country: '{destination_country}'")
        
        logger.info("Airport and location mapping completed successfully")
        return freight_data
        
    except Exception as e:
        logger.error(f"Error in airport and location mapping: {e}")
        return freight_data

def normalize_magno_addresses(freight_data: Dict[str, Any], vendor_reference_id: str) -> Dict[str, Any]:
    """
    Normalize MAGNO addresses - specifically handle GUARULHOS, SP case.
    When source_address contains "GUARULHOS, SP", set source_city to "São Paulo" and source_state to "SP".
    
    Args:
        freight_data: The freight data dictionary
        vendor_reference_id: The vendor reference ID to check if it's MAGNO
        
    Returns:
        The freight data with normalized addresses
    """
    try:
        # Only apply to MAGNO vendor
        if vendor_reference_id and vendor_reference_id.upper() != "MAGNO":
            return freight_data
        
        logger.info("🔧 Normalizing MAGNO addresses...")
        
        if 'data' in freight_data and isinstance(freight_data['data'], list):
            for data_item in freight_data['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        if not isinstance(shipment, dict):
                            continue
                        
                        # Check source_address for GUARULHOS, SP pattern
                        source_address = shipment.get('source_address', '')
                        if source_address and isinstance(source_address, str):
                            # Check if address contains "GUARULHOS, SP" (case-insensitive)
                            if 'GUARULHOS' in source_address.upper() and 'SP' in source_address.upper():
                                # Check if it's the pattern "GUARULHOS, SP" or similar
                                pattern = r'GUARULHOS[,\s]+SP'
                                if re.search(pattern, source_address, re.IGNORECASE):
                                    # Set source_city to "São Paulo" and source_state to "SP"
                                    shipment['source_city'] = 'São Paulo'
                                    shipment['source_country'] = 'BR'  # Brazil
                                    logger.info(f"✅ Normalized MAGNO address: GUARULHOS, SP -> source_city='São Paulo")
                                    logger.info(f"   Original source_address: {source_address}")
        
        logger.info("🎯 MAGNO address normalization completed")
        return freight_data
        
    except Exception as e:
        logger.error(f"Error normalizing MAGNO addresses: {e}")
        return freight_data

def apply_business_rules(freight_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply business rules after charge mapping.
    """
    try:
        logger.info("🔧 Applying business rules...")
        
        if 'data' in freight_data and isinstance(freight_data['data'], list):
            for data_item in freight_data['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        if 'charges' in shipment and isinstance(shipment['charges'], list):
                            
                            # Collect all carrier types and charge names for rule processing
                            carrier_types = []
                            charge_names = []
                            
                            for charge in shipment['charges']:
                                if 'carrier_type' in charge:
                                    carrier_types.append(charge['carrier_type'])
                                if 'charge_name' in charge:
                                    charge_names.append(charge['charge_name'])  # Keep original case
                            
                            logger.info(f"📊 Found carrier types: {carrier_types}")
                            logger.info(f"📊 Found charge names: {charge_names}")
                            
                            # Rule 7-8: Service Type Rules
                            if 'origin_service_type' in shipment:
                                if 'PC' in carrier_types:
                                    shipment['origin_service_type'] = 'D'
                                    logger.info("✅ origin_service_type set to 'D' (found PC carrier type)")
                                else:
                                    shipment['origin_service_type'] = 'A'
                                    logger.info("✅ origin_service_type set to 'A' (no PC carrier type)")
                            
                            if 'destination_service_type' in shipment:
                                if 'OC' in carrier_types:
                                    shipment['destination_service_type'] = 'D'
                                    logger.info("✅ destination_service_type set to 'D' (found OC carrier type)")
                                else:
                                    shipment['destination_service_type'] = 'A'
                                    logger.info("✅ destination_service_type set to 'A' (no OC carrier type)")
                            
                            # Rule 9: Service Code Rule
                            if 'custom' in shipment and isinstance(shipment['custom'], dict):
                                service_code = shipment['custom'].get('service_code', '')
                                if any(keyword in service_code for keyword in ['speed', 'express', 'priority']):
                                    shipment['custom']['service_code'] = 'EXP'
                                    logger.info("✅ service_code set to 'EXP' (found speed/express/priority)")
                                else:
                                    shipment['custom']['service_code'] = 'DEF'
                                    logger.info("✅ service_code set to 'DEF' (no speed/express/priority)")
                            
                            # Rule 10-12: Hazardous Material Rules
                            hazardous_keywords = ['Hazardous', 'dangerous', 'HazMat', 'dg', 'Dg']
                            has_hazardous = any(any(keyword in charge_name for keyword in hazardous_keywords) for charge_name in charge_names)
                            
                            if 'custom' in shipment and isinstance(shipment['custom'], dict):
                                if has_hazardous:
                                    shipment['custom']['hazardous_material'] = 'TRUE'
                                    shipment['custom']['cargo'] = 'FALSE'
                                    logger.info("✅ hazardous_material set to 'TRUE', cargo set to 'FALSE' (found hazardous keywords)")
                                else:
                                    shipment['custom']['hazardous_material'] = 'FALSE'
                                    shipment['custom']['cargo'] = 'TRUE'
                                    logger.info("✅ hazardous_material set to 'FALSE', cargo set to 'TRUE' (no hazardous keywords)")
                            
                            # Rule 13: Temperature Control Rule
                            temp_keywords = ['Thermal blanket', 'temperature', 'cold', 'cold storage']
                            has_temp_control = any(any(keyword in charge_name for keyword in temp_keywords) for charge_name in charge_names)
                            
                            if 'custom' in shipment and isinstance(shipment['custom'], dict):
                                if has_temp_control:
                                    shipment['custom']['temperature_control'] = 'TRUE'
                                    logger.info("✅ temperature_control set to 'TRUE' (found temperature keywords)")
                                else:
                                    shipment['custom']['temperature_control'] = 'FALSE'
                                    logger.info("✅ temperature_control set to 'FALSE' (no temperature keywords)")
        
        logger.info("🎯 Business rules applied successfully")
        
        # Remove carrier_type from charges (used only for business rules, not in final output)
        if 'data' in freight_data and isinstance(freight_data['data'], list):
            for data_item in freight_data['data']:
                if 'shipments' in data_item and isinstance(data_item['shipments'], list):
                    for shipment in data_item['shipments']:
                        if 'charges' in shipment and isinstance(shipment['charges'], list):
                            for charge in shipment['charges']:
                                if 'carrier_type' in charge:
                                    del charge['carrier_type']
        
        logger.info("🧹 Removed carrier_type from final JSON output")
        return freight_data
        
    except Exception as e:
        logger.error(f"Error applying business rules: {e}")
        return freight_data

def calculate_weighted_confidence(json_data):
    """
    Calculate the weighted confidence score from structured JSON data, 
    giving different priorities to different fields with highest priority to
    invoice number, invoice date, payment due date, bill of lading number, and charges.
    
    Args:
        json_data: The JSON data containing confidence scores
        
    Returns:
        float: Weighted confidence score
    """
    # Define field priorities (weights) - higher number means more important
    field_priorities = {
        # Highest priority fields as specified
        "invoiceNumber": 10.0,
        "invoiceDate": 10.0,
        "payment_due_date": 10.0,
        "bill_of_lading_number": 10.0,
        
        # Charge fields (also high priority)
        "charge_code": 8.0,
        "charge_name": 8.0,
        "charge_gross_amount": 10.0,
        "charge_reference_id": 8.0,
        "mode": 8.0,
        "charge_reference_type": 8.0,
        
        # Secondary but still important fields
        "vendor_reference_id": 5.0,
        "total_invoice_value": 6.0,
        "currency": 5.0,
        
        # Less critical fields
        "source_name": 3.0,
        "destination_name": 3.0,
        "shipment_weight": 3.0,
        "shipment_volume": 3.0,
        "shipment_weight_quantifier": 2.0,
        "shipment_volume_uom": 2.0,
        
        # Default weight for any other fields
        "default": 1.0
    }
    
    weighted_scores = []
    total_weight = 0
    
    def process_confidence_score(field_name, confidence):
        """Process a single confidence score with appropriate weighting"""
        weight = field_priorities.get(field_name, field_priorities["default"])
        weighted_scores.append(confidence * weight)
        return weight
    
    def extract_weighted_confidence(obj, path=""):
        """Recursively extract weighted confidence scores from nested objects."""
        if isinstance(obj, dict):
            if 'confidence' in obj:
                # For charge fields, use the specific field name instead of the full path
                if 'charges' in path:
                    # Extract the field name after the last dot
                    parts = path.split('.')
                    if len(parts) > 1:
                        field_name = parts[-1]  # Get last part after dot
                    else:
                        field_name = path
                    
                    weight = process_confidence_score(field_name, obj['confidence'])
                else:
                    # For top-level fields, use the full path
                    field_name = path.split('.')[-1] if '.' in path else path
                    weight = process_confidence_score(field_name, obj['confidence'])
                
                nonlocal total_weight
                total_weight += weight
            
            # Recursively process all values in the dictionary
            for key, value in obj.items():
                new_path = f"{path}.{key}" if path else key
                extract_weighted_confidence(value, new_path)
                
        elif isinstance(obj, list):
            # Process each item in the list
            for i, item in enumerate(obj):
                new_path = f"{path}[{i}]"
                extract_weighted_confidence(item, new_path)
    
    # Start extraction
    extract_weighted_confidence(json_data)
    
    # Calculate weighted average if there are any scores
    if weighted_scores:
        return sum(weighted_scores) / total_weight if total_weight > 0 else 0.0
    else:
        return 0.0  # Return 0 if no confidence scores found

def get_secret(secret_name: str) -> Dict[str, str]:
    """Retrieve a secret from AWS Secrets Manager."""
    try:
        client = boto3.client('secretsmanager', region_name=REGION)
        response = client.get_secret_value(SecretId=secret_name)
        logger.info(f"Successfully retrieved secret: {secret_name}")
        return json.loads(response['SecretString'])
    except Exception as e:
        logger.error(f"Error retrieving secret {secret_name}: {str(e)}")
        raise

# Get SMTP credentials from Secrets Manager
try:
    SMTP_CONFIG = get_secret(SMTP_SECRET_NAME)
    SMTP_USERNAME = SMTP_CONFIG["SMTP_USERNAME"]
    SMTP_PASSWORD = SMTP_CONFIG["SMTP_PASSWORD"]
    logger.info(f"Successfully retrieved SMTP credentials from Secrets Manager using secret: {SMTP_SECRET_NAME}")
except Exception as e:
    logger.error(f"Failed to retrieve SMTP credentials from secret {SMTP_SECRET_NAME}: {str(e)}")
    SMTP_USERNAME = ""
    SMTP_PASSWORD = ""

def update_attachment_status(
    dynamodb_client,
    table_name,
    email_id,
    attachment_id,
    status,
    error=None,
    output_path=None,
    is_create=False,
    carrier_name=None,
    mode=None,
    invoice_number=None,
    invoice_date=None,  # Invoice date
    missing_critical_field=None,
    textract_failed=None,
    classification_failed=None,
    extraction_failed=None,
    format_failed=None,
    timeout_occurred=None,
    missing_fields=None,  # Array of missing field names (legacy - for backward compatibility)
    external_field_errors=None,  # Array of external field errors (mandatory fields are empty)
    internal_field_errors=None,  # Array of internal field errors (format validation failures)
    confidence_score=None,  # Overall confidence score
    extracted_fields=None,  # Parameter to store extracted field values
    api_response=None,  # Parameter to store API response details
    filename=None,  # Original filename
    file_type=None,  # File type (PDF, etc.)
    processing_type=None,  # Processing type (Invoice, etc.)
    s3_path=None,  # S3 path of the original file
    type=None,  # Record type (Attachment, etc.)
    to_email=None  # Email address to send notifications to
):
    """Update/create the status of an attachment in DynamoDB, setting correct timestamps and failure flags."""
    try:
        logger.info(f"Updating DynamoDB status for email_id: {email_id}, attachment_id: {attachment_id}")
        logger.info(f"Status: {status}, is_create: {is_create}")
        
        now = datetime.utcnow()
        now_ms = int(now.timestamp() * 1000)
        now_iso = now.isoformat()  # No trailing Z; always UTC

        update_expr = 'SET #status = :status, updated_at = :updated_at, updated_at_iso = :updated_at_iso'
        expr_attrs = {'#status': 'status'}
        expr_values = {
            ':status': {'S': status},
            ':updated_at': {'N': str(now_ms)},
            ':updated_at_iso': {'S': now_iso}
        }

        # Only set created_at if is_create is True (i.e., this is a new item)
        if is_create:
            update_expr += ', created_at = :created_at, created_at_iso = :created_at_iso'
            expr_values[':created_at'] = {'N': str(now_ms)}
            expr_values[':created_at_iso'] = {'S': now_iso}
            logger.info("Setting created_at timestamp for new item")

        # Add carrier_name if provided
        if carrier_name:
            update_expr += ', carrier_name = :carrier_name'
            expr_values[':carrier_name'] = {'S': carrier_name}
            logger.info(f"Setting carrier_name: {carrier_name}")

        # Add overall confidence score if provided
        if confidence_score is not None:
            update_expr += ', confidence_score = :confidence_score'
            expr_values[':confidence_score'] = {'N': str(confidence_score)}
            logger.info(f"Setting confidence_score: {confidence_score}")
            
        # Add extracted fields with their values, confidence scores, and explanations
        if extracted_fields is not None and isinstance(extracted_fields, list):
            update_expr += ', extracted_fields = :extracted_fields'
            # Convert list to DynamoDB list format
            extracted_fields_list = {'L': []}
            for field in extracted_fields:
                field_map = {'M': {}}
                # Add field_name and value (required)
                if 'field_name' in field:
                    field_map['M']['field_name'] = {'S': str(field['field_name'])}
                if 'value' in field:
                    field_map['M']['value'] = {'S': str(field['value'])}
                # Add confidence if available
                if 'confidence' in field:
                    if isinstance(field['confidence'], (int, float)):
                        field_map['M']['confidence'] = {'N': str(field['confidence'])}
                    else:
                        field_map['M']['confidence'] = {'S': str(field['confidence'])}
                # Add explanation if available
                if 'explanation' in field:
                    field_map['M']['explanation'] = {'S': str(field['explanation'])}
                
                extracted_fields_list['L'].append(field_map)
            expr_values[':extracted_fields'] = extracted_fields_list
            logger.info(f"Setting extracted_fields with {len(extracted_fields)} fields")
            
        # Add API response details if provided
        if api_response is not None:
            update_expr += ', api_response = :api_response'
            api_response_map = {'M': {}}
            
            # Add status code
            if 'status_code' in api_response:
                api_response_map['M']['status_code'] = {'N': str(api_response['status_code'])}
                
            # Add success flag
            if 'success' in api_response:
                api_response_map['M']['success'] = {'BOOL': api_response['success']}
                
            # Add timestamp
            if 'timestamp' in api_response:
                api_response_map['M']['timestamp'] = {'S': api_response['timestamp']}
                
            # Add response body if available (as a string)
            if 'body' in api_response:
                api_response_map['M']['body'] = {'S': str(api_response['body'])}
                
            expr_values[':api_response'] = api_response_map
            logger.info(f"Setting api_response with status_code: {api_response.get('status_code')}")

        # Add mode if provided - use expression attribute name since "mode" is a reserved keyword
        if mode:
            update_expr += ', #trans_mode = :trans_mode'
            expr_attrs['#trans_mode'] = 'mode'
            expr_values[':trans_mode'] = {'S': mode}
            logger.info(f"Setting mode: {mode}")

        # Add invoice_number if provided (including empty strings)
        if invoice_number is not None:
            update_expr += ', invoice_number = :invoice_number'
            expr_values[':invoice_number'] = {'S': invoice_number}
            logger.info(f"Setting invoice_number: {invoice_number}")

        # Add invoice_date if provided (including empty strings)
        if invoice_date is not None:
            update_expr += ', invoice_date = :invoice_date'
            expr_values[':invoice_date'] = {'S': invoice_date}
            logger.info(f"Setting invoice_date: {invoice_date}")

        # Add specific failure flags if provided
        if missing_critical_field is not None:
            update_expr += ', missing_critical_field = :missing_critical_field'
            expr_values[':missing_critical_field'] = {'N': str(missing_critical_field)}
            logger.info(f"Setting missing_critical_field: {missing_critical_field}")

        # Add array of missing fields (formatted like extracted_fields) - legacy support
        if missing_fields is not None:
            update_expr += ', missing_fields = :missing_fields'
            # Convert list to DynamoDB list format with Map objects
            missing_fields_list = {'L': []}
            if isinstance(missing_fields, list):
                for field in missing_fields:
                    field_map = {'M': {}}
                    # Add field_name and explanation (required for missing fields)
                    if 'field_name' in field:
                        field_map['M']['field_name'] = {'S': str(field['field_name'])}
                    if 'explanation' in field:
                        field_map['M']['explanation'] = {'S': str(field['explanation'])}
                    
                    missing_fields_list['L'].append(field_map)
                expr_values[':missing_fields'] = missing_fields_list
                logger.info(f"Setting missing_fields with {len(missing_fields)} fields")
            else:
                # If missing_fields is not a list, default to empty list
                expr_values[':missing_fields'] = {'L': []}
                logger.warning("missing_fields is not a list, setting to empty array")

        # Add array of external field errors (mandatory fields are empty)
        if external_field_errors is not None:
            update_expr += ', external_field_errors = :external_field_errors'
            # Convert list to DynamoDB list format with Map objects
            external_errors_list = {'L': []}
            if isinstance(external_field_errors, list):
                for field in external_field_errors:
                    field_map = {'M': {}}
                    # Add field_name and explanation (required for external field errors)
                    if 'field_name' in field:
                        field_map['M']['field_name'] = {'S': str(field['field_name'])}
                    if 'explanation' in field:
                        field_map['M']['explanation'] = {'S': str(field['explanation'])}
                    
                    external_errors_list['L'].append(field_map)
                expr_values[':external_field_errors'] = external_errors_list
                logger.info(f"Setting external_field_errors with {len(external_field_errors)} fields")
            else:
                # If external_field_errors is not a list, default to empty list
                expr_values[':external_field_errors'] = {'L': []}
                logger.warning("external_field_errors is not a list, setting to empty array")

        # Add array of internal field errors (format validation failures)
        if internal_field_errors is not None:
            update_expr += ', internal_field_errors = :internal_field_errors'
            # Convert list to DynamoDB list format with Map objects
            internal_errors_list = {'L': []}
            if isinstance(internal_field_errors, list):
                for field in internal_field_errors:
                    field_map = {'M': {}}
                    # Add field_name and explanation (required for internal field errors)
                    if 'field_name' in field:
                        field_map['M']['field_name'] = {'S': str(field['field_name'])}
                    if 'explanation' in field:
                        field_map['M']['explanation'] = {'S': str(field['explanation'])}
                    
                    internal_errors_list['L'].append(field_map)
                expr_values[':internal_field_errors'] = internal_errors_list
                logger.info(f"Setting internal_field_errors with {len(internal_field_errors)} fields")
            else:
                # If internal_field_errors is not a list, default to empty list
                expr_values[':internal_field_errors'] = {'L': []}
                logger.warning("internal_field_errors is not a list, setting to empty array")

        if textract_failed is not None:
            update_expr += ', textract_failed = :textract_failed'
            expr_values[':textract_failed'] = {'N': str(textract_failed)}
            logger.info(f"Setting textract_failed: {textract_failed}")

        if classification_failed is not None:
            update_expr += ', classification_failed = :classification_failed'
            expr_values[':classification_failed'] = {'N': str(classification_failed)}
            logger.info(f"Setting classification_failed: {classification_failed}")

        if extraction_failed is not None:
            update_expr += ', extraction_failed = :extraction_failed'
            expr_values[':extraction_failed'] = {'N': str(extraction_failed)}
            logger.info(f"Setting extraction_failed: {extraction_failed}")

        if format_failed is not None:
            update_expr += ', format_failed = :format_failed'
            expr_values[':format_failed'] = {'N': str(format_failed)}
            logger.info(f"Setting format_failed: {format_failed}")
            
        if timeout_occurred is not None:
            update_expr += ', timeout_occurred = :timeout_occurred'
            expr_values[':timeout_occurred'] = {'N': str(timeout_occurred)}
            logger.info(f"Setting timeout_occurred: {timeout_occurred}")

        # Add filename if provided
        if filename:
            update_expr += ', filename = :filename'
            expr_values[':filename'] = {'S': filename}
            logger.info(f"Setting filename: {filename}")

        # Add file_type if provided
        if file_type:
            update_expr += ', file_type = :file_type'
            expr_values[':file_type'] = {'S': file_type}
            logger.info(f"Setting file_type: {file_type}")

        # Add processing_type if provided
        if processing_type:
            update_expr += ', processing_type = :processing_type'
            expr_values[':processing_type'] = {'S': processing_type}
            logger.info(f"Setting processing_type: {processing_type}")

        # Add s3_path if provided
        if s3_path:
            update_expr += ', s3_path = :s3_path'
            expr_values[':s3_path'] = {'S': s3_path}
            logger.info(f"Setting s3_path: {s3_path}")

        # Add type if provided
        if type:
            update_expr += ', #type = :type'
            expr_attrs['#type'] = 'type'
            expr_values[':type'] = {'S': type}
            logger.info(f"Setting type: {type}")

        # Add to_email if provided
        if to_email:
            update_expr += ', to_email = :to_email'
            expr_values[':to_email'] = {'S': to_email}
            logger.info(f"Setting to_email: {to_email}")

        if error:
            update_expr += ', #err = :error'
            expr_attrs['#err'] = 'error'
            error_dict = {
                'message': {'S': error.get('message', '')},
                'error_code': {'S': error.get('error_code', '')},
                'timestamp': {'S': now_iso}
            }
            # Add rejection_reason if present
            if 'rejection_reason' in error:
                error_dict['rejection_reason'] = {'S': error.get('rejection_reason', '')}
            # Add carrier if present
            if 'carrier' in error:
                error_dict['carrier'] = {'S': error.get('carrier', '')}
            # Add sender_email if present
            if 'sender_email' in error:
                error_dict['sender_email'] = {'S': error.get('sender_email', '')}
            
            expr_values[':error'] = {'M': error_dict}
            logger.info(f"Setting error: {error.get('error_code')} - {error.get('message')}")
            if 'rejection_reason' in error:
                logger.info(f"Rejection reason: {error.get('rejection_reason')}")

        if output_path:
            update_expr += ', output_path = :output_path, completed_at = :completed_at, completed_at_iso = :completed_at_iso'
            expr_values[':output_path'] = {'S': output_path}
            expr_values[':completed_at'] = {'N': str(now_ms)}
            expr_values[':completed_at_iso'] = {'S': now_iso}
            logger.info(f"Setting output_path: {output_path}")

        logger.info(f"DynamoDB update expression: {update_expr}")
        logger.info(f"Expression attribute names: {expr_attrs}")
        logger.info(f"DynamoDB update expression values prepared")
        
        response = dynamodb_client.update_item(
            TableName=table_name,
            Key={
                'pk': {'S': f"EMAIL#{email_id}"},
                'sk': {'S': f"ATTACHMENT#{attachment_id}"}
            },
            UpdateExpression=update_expr,
            ExpressionAttributeNames=expr_attrs,
            ExpressionAttributeValues=expr_values,
            ReturnValues='ALL_NEW'
        )
        logger.info(f"DynamoDB update successful for email_id: {email_id}, attachment_id: {attachment_id}.")
        logger.info(f"DynamoDB update completed successfully")

        # Update the parent email job counters (only on state transitions)
        counter_update = {
            'PROCESSING': 'processing_attachments',
            'COMPLETED': 'completed_attachments',
            'FAILED': 'failed_attachments'
        }

        if status in counter_update:
            counter_field = counter_update[status]
            logger.info(f"Updating parent email counter: {counter_field}")
            counter_response = dynamodb_client.update_item(
                TableName=table_name,
                Key={
                    'pk': {'S': f"EMAIL#{email_id}"},
                    'sk': {'S': 'METADATA'}
                },
                UpdateExpression=f'ADD {counter_field} :inc',
                ExpressionAttributeValues={
                    ':inc': {'N': '1'}
                },
                ReturnValues='ALL_NEW'
            )
            logger.info(f"Counter update successful for email_id: {email_id}.")
            logger.info(f"Counter update completed successfully")

        return True
    except Exception as e:
        logger.error(f"Failed to update attachment status: {str(e)}", exc_info=True)
        return False

def format_invoice_data_as_html(data: Dict[str, Any]) -> str:
    """Format invoice data as HTML with specific fields as requested."""
    try:
        logger.info(f"Formatting invoice data as HTML, data keys: {list(data.keys())}")
        
        # Extract data from the structure (handle both nested and flat formats)
        invoice_data = data
        if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
            invoice_data = data['data'][0]
        
        html = f"""
        <div style="font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto;">
            
            <h2>Invoice Details</h2>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <th style="text-align: left; padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Field</th>
                    <th style="text-align: left; padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Value</th>
                </tr>
        """
        
        # Invoice Details - specific fields as requested
        invoice_fields = [
            'invoice_number', 'invoice_date', 'vendor_reference_id', 'payment_due_date',
            'bill_of_lading_number', 'total_invoice_value'
        ]
        
        for field in invoice_fields:
            value = invoice_data.get(field, "")
            if value is None:
                value = ""
            html += f"""
            <tr>
                <td style="padding: 8px; border: 1px solid #ddd;"><strong>{field.replace('_', ' ').title()}</strong></td>
                <td style="padding: 8px; border: 1px solid #ddd;">{value}</td>
            </tr>
            """
        
        html += "</table>"
        
        # Shipment Details
        if 'shipments' in invoice_data and isinstance(invoice_data['shipments'], list) and invoice_data['shipments']:
            html += """
            <h2>Shipment Details</h2>
            <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                <tr>
                    <th style="text-align: left; padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Field</th>
                    <th style="text-align: left; padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Value</th>
                </tr>
            """
            
            # Shipment fields as requested
            shipment_fields = [
                "shipment_number", "shipment_creation_date", "mode", "source_name", "source_city",
                "source_country", "source_state", "destination_name", "destination_city", 
                "destination_country", "destination_state", "shipment_weight", "shipment_volume", 
                "shipment_total_value"
            ]
            
            for shipment in invoice_data['shipments']:
                for field in shipment_fields:
                    value = shipment.get(field, "")
                    if value is None:
                        value = ""
                    html += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;"><strong>{field.replace('_', ' ').title()}</strong></td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{value}</td>
                    </tr>
                    """
            
            html += "</table>"
            
            # Charges section
            if "charges" in invoice_data['shipments'][0] and isinstance(invoice_data['shipments'][0]["charges"], list) and invoice_data['shipments'][0]["charges"]:
                html += """
                <h2>Charges</h2>
                <table style="width: 100%; border-collapse: collapse; margin-bottom: 20px;">
                    <tr>
                        <th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Charge Name</th>
                        <th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Charge Code</th>
                        <th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Charge Gross Amount</th>
                        <th style="padding: 8px; border: 1px solid #ddd; background-color: #f2f2f2;">Currency</th>
                    </tr>
                """
                
                for charge in invoice_data['shipments'][0]["charges"]:
                    html += f"""
                    <tr>
                        <td style="padding: 8px; border: 1px solid #ddd;">{charge.get('charge_name', '')}</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{charge.get('charge_code', '')}</td>
                        <td style="padding: 8px; border: 1px solid #ddd; text-align: right;">{charge.get('charge_gross_amount', '')}</td>
                        <td style="padding: 8px; border: 1px solid #ddd;">{charge.get('currency', '')}</td>
                    </tr>
                    """
                
                html += "</table>"
        
        html += """
            
        </div>
        """
        
        logger.info(f"Successfully formatted invoice data as HTML, length: {len(html)} characters")
        return html
        
    except Exception as e:
        logger.error(f"Error formatting invoice data as HTML: {str(e)}", exc_info=True)
        return f"""
        <div style="font-family: Arial, sans-serif; max-width: 800px; margin: 0 auto;">
            <p>Dear Recipient,</p>
            <h2>Invoice Details</h2>
            <p>Error formatting invoice data: {str(e)}</p>
            <p>Raw data preserved for processing.</p>
            <p>Best regards,<br><strong>PI</strong></p>
        </div>
        """

# =============================================================================
# EMAIL FUNCTIONS DISABLED - NOW HANDLED BY SEPARATE EMAIL LAMBDA FUNCTION
# =============================================================================
# 
# All email sending functionality has been moved to a separate Lambda function
# (email/lambda_function.py) to provide better separation of concerns and
# consolidated email reporting.
#
# The following functions are still defined but their calls have been commented out:
# - send_email() - Main email sending function
# - send_unclassified_notification() - Unclassified document notifications  
# - send_error_notification() - Error notifications
#
# Email sending is now handled by the dedicated email Lambda function which:
# - Fetches data from DynamoDB for the past 24 hours
# - Generates consolidated reports for Meta and J&J
# - Sends 2 emails with 3 recipients each
# - Includes Excel attachments with detailed invoice data
# =============================================================================

# def send_unclassified_notification(
#     smtp_server: str,
#     smtp_port: int,
#     smtp_username: str,
#     smtp_password: str,
#     from_email: str,
#     to_email: str,
#     subject: str,
#     original_body: str,
#     filename: str,
#     message_id: Optional[str] = None,
#     quoted_sender: Optional[str] = None,
#     quoted_date: Optional[str] = None,
#     quoted_subject: Optional[str] = None,
#     quoted_body: Optional[str] = None
# ) -> None:
#     """Send notification email for unclassified documents."""
#     # FUNCTION BODY COMMENTED OUT - EMAIL HANDLED BY SEPARATE LAMBDA
    try:
        logger.info(f"Preparing unclassified document notification to {to_email}")
        logger.info(f"Email subject: {subject}")
        logger.info(f"Filename: {filename}")
        
        msg = MIMEMultipart("alternative")
        
        # Set subject with "Re:" prefix if not already present
        if not subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: Unable to Process Invoice - {subject}"
        else:
            msg["Subject"] = f"Unable to Process Invoice - {subject}"
            
        logger.info(f"Email subject set to: {msg['Subject']}")
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Set proper threading headers
        if message_id:
            # Ensure message ID has angle brackets for proper threading
            if not message_id.startswith('<'):
                formatted_message_id = f"<{message_id}>"
            else:
                formatted_message_id = message_id
                
            logger.info(f"Setting email threading headers with message_id: {formatted_message_id}")
            msg["In-Reply-To"] = formatted_message_id
            msg["References"] = formatted_message_id
            
            # Add thread-topic for Outlook compatibility
            clean_subject = subject.replace("Re: ", "").replace("RE: ", "").strip()
            msg["Thread-Topic"] = clean_subject
            msg["Thread-Index"] = message_id.strip('<>')  # For Outlook
        
        # Build quoted block if info is provided
        quoted_block = ""
        if quoted_sender and quoted_date and quoted_subject and quoted_body:
            logger.info(f"Adding quoted content from: {quoted_sender}, date: {quoted_date}")
            quoted_block = f"""
            <div style='margin-top: 20px; border-top: 1px solid #ddd;'>
                <div style='color:gray; font-size:small; margin: 10px 0;'>
                    On {quoted_date}, {quoted_sender} wrote:<br>
                    <b>Subject:</b> {quoted_subject}
                </div>
                <div style='border-left:4px solid #ccc; padding-left:15px; margin:10px 0;'>
                    {quoted_body}
                </div>
            </div>
            """
        
        html_text = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .error-box {{
                    background-color: #fff3f3;
                    border-left: 4px solid #dc3545;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }}
                .file-info {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <p>Dear Recipient,</p>
            <div class="error-box">
                <h2 style="color: #dc3545; margin-top: 0;">Unable to Process Invoice</h2>
                <p>We were unable to process the attached document as an invoice. This could be due to one of the following reasons:</p>
                <ul>
                    <li>The document is not a valid invoice</li>
                    <li>The document format is not supported</li>
                    <li>The document is not from a supported carrier</li>
                    <li>The document is unclear or unreadable</li>
                </ul>
            </div>
            
            <div class="file-info">
                <p><strong>File Name:</strong> {filename}</p>
                <p><strong>Original Subject:</strong> {subject}</p>
            </div>
            
            <p>Please ensure that:</p>
            <ul>
                <li>The document is a clear, readable invoice</li>
                <li>The invoice is from one of our supported carriers</li>
                <li>The document is in PDF format</li>
                <li>All text in the document is selectable (not scanned as an image)</li>
            </ul>
            
            <p>Please send a new email with a valid invoice attachment.</p>
            <p>Best regards,<br><strong>PI</strong></p>
            <div class="disclaimer">
                This is an automated email from PiAgent. Please do not reply to this email.
            </div>
            {quoted_block}
        </body>
        </html>
        """
        
        logger.info(f"HTML email content length: {len(html_text)} characters")
        
        html_part = MIMEText(html_text, "html")
        msg.attach(html_part)
        
        # Send the email
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            logger.info(f"SMTP connection established, logging in as: {smtp_username}")
            server.login(smtp_username, smtp_password)
            logger.info(f"SMTP login successful, sending email to: {to_email}")
            server.send_message(msg)
            logger.info("Unclassified document notification sent successfully!")
            
    except Exception as e:
        logger.error(f"Error sending unclassified document notification: {str(e)}", exc_info=True)
        raise

def send_external_field_error_alert(
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    filename: str,
    external_field_errors: List[Dict[str, str]],
    invoice_number: str = None,
    message_id: Optional[str] = None,
    subject: Optional[str] = None,
    quoted_sender: Optional[str] = None,
    quoted_date: Optional[str] = None,
    quoted_subject: Optional[str] = None,
    quoted_body: Optional[str] = None
) -> None:
    """
    Send instant email alert to customer for external field errors (missing mandatory fields).
    This is sent when mandatory fields are empty or missing.
    """
    try:
        logger.info(f"Sending external field error alert to customer: {to_email} for file: {filename}")
        
        # Create subject
        if not subject:
            email_subject = f"Re: Invoice Processing Failed - Missing Required Fields - {filename}"
            if invoice_number:
                email_subject = f"Re: Invoice Processing Failed - Missing Required Fields - Invoice {invoice_number} - {filename}"
        else:
            if not subject.lower().startswith("re:"):
                email_subject = f"Re: {subject}"
            else:
                email_subject = subject
        
        # Create HTML content
        error_details = ""
        for error in external_field_errors:
            field_name = error.get("field_name", "Unknown field")
            explanation = error.get("explanation", "Field is missing")
            error_details += f"<li><strong>{field_name}:</strong> {explanation}</li>"
        
        # Build quoted block if info is provided
        quoted_block = ""
        if quoted_sender and quoted_date and quoted_subject and quoted_body:
            logger.info(f"Adding quoted content from: {quoted_sender}, date: {quoted_date}")
            quoted_block = f"""
            <div style='margin-top: 20px; border-top: 1px solid #ddd;'>
                <div style='color:gray; font-size:small; margin: 10px 0;'>
                    On {quoted_date}, {quoted_sender} wrote:<br>
                    <b>Subject:</b> {quoted_subject}
                </div>
                <div style='border-left:4px solid #ccc; padding-left:15px; margin:10px 0;'>
                    {quoted_body}
                </div>
            </div>
            """
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background-color: #dc3545; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .error-list {{ background-color: #fff3f3; padding: 20px; margin: 20px 0; border-left: 4px solid #dc3545; }}
                .file-info {{ background-color: #f8f9fa; padding: 15px; margin: 15px 0; border-radius: 4px; }}
                .footer {{ background-color: #f0f0f0; padding: 15px; text-align: center; font-size: 12px; }}
                .disclaimer {{ font-size: 12px; color: #666; margin-top: 20px; padding: 10px; background-color: #f8f9fa; border-left: 4px solid #007bff; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>Invoice Processing Failed - Missing Required Fields</h2>
            </div>
            <div class="content">
                <p>Dear Customer,</p>
                
                <div class="file-info">
                    <p><strong>File Name:</strong> {filename}</p>
                    {f'<p><strong>Invoice Number:</strong> {invoice_number}</p>' if invoice_number else ''}
                </div>
                
                <div class="error-list">
                    <h3 style="color: #dc3545; margin-top: 0;">Required Details Missing</h3>
                    <p>The following required details are missing from the invoice, making it invalid:</p>
                    <ul style="margin-left: 20px;">
                        {error_details}
                    </ul>
                    <p><strong>Please provide the missing information and resubmit the document.</strong></p>
                </div>
                
                <p>To resubmit your invoice, please send a new email with all required fields properly filled in.</p>
                
                <p>Best regards,<br><strong>Pando Invoice Processing System</strong></p>
                
                <div class="disclaimer">
                    This is an automated email from the Pando Invoice Processing System. Please do not reply to this email.
                </div>
                {quoted_block}
            </div>
            <div class="footer">
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        plain_text = f"""
Invoice Processing Failed - Missing Required Fields

Dear Customer,

File Name: {filename}
{f'Invoice Number: {invoice_number}' if invoice_number else ''}

The following required details are missing from the invoice, making it invalid:

{chr(10).join([f"- {error.get('field_name', 'Unknown field')}: {error.get('explanation', 'Field is missing')}" for error in external_field_errors])}

Please provide the missing information and resubmit the document.

To resubmit your invoice, please send a new email with all required fields properly filled in.

Best regards,
Pando Invoice Processing System

This is an automated email from the Pando Invoice Processing System. Please do not reply to this email.
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = email_subject
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Set proper threading headers
        if message_id:
            # Ensure message ID has angle brackets for proper threading
            if not message_id.startswith('<'):
                formatted_message_id = f"<{message_id}>"
            else:
                formatted_message_id = message_id
                
            logger.info(f"Setting email threading headers with message_id: {formatted_message_id}")
            msg["In-Reply-To"] = formatted_message_id
            msg["References"] = formatted_message_id
            
            # Add thread-topic for Outlook compatibility
            clean_subject = subject.replace("Re: ", "").replace("RE: ", "").strip() if subject else email_subject.replace("Re: ", "").replace("RE: ", "").strip()
            msg["Thread-Topic"] = clean_subject
            msg["Thread-Index"] = message_id.strip('<>')  # For Outlook
        
        # Attach parts
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg)
            logger.info(f"External field error alert sent successfully to customer: {to_email}")
            
    except Exception as e:
        logger.error(f"Failed to send external field error alert: {str(e)}")


def send_internal_field_error_alert(
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    from_email: str,
    to_emails: List[str],  # Changed to accept list of emails
    filename: str,
    internal_field_errors: List[Dict[str, str]],
    invoice_number: str = None
) -> None:
    """
    Send instant email alert for internal field format validation errors to Pando team.
    This is sent when mandatory fields are present but not in the correct format.
    """
    try:
        logger.info(f"Sending internal field error alert for file: {filename} to Pando team: {to_emails}")
        
        # Create subject
        subject = f"URGENT: Field Format Validation Failed - {filename}"
        if invoice_number:
            subject = f"URGENT: Field Format Validation Failed - Invoice {invoice_number} - {filename}"
        
        # Create HTML content
        error_details = ""
        for error in internal_field_errors:
            field_name = error.get("field_name", "Unknown field")
            explanation = error.get("explanation", "Format validation failed")
            error_details += f"<li><strong>{field_name}:</strong> {explanation}</li>"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .header {{ background-color: #ff4444; color: white; padding: 20px; text-align: center; }}
                .content {{ padding: 20px; }}
                .error-list {{ background-color: #f8f8f8; padding: 15px; border-left: 4px solid #ff4444; }}
                .footer {{ background-color: #f0f0f0; padding: 15px; text-align: center; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2>🚨 URGENT: Field Format Validation Failed</h2>
            </div>
            <div class="content">
                <p><strong>File:</strong> {filename}</p>
                {f'<p><strong>Invoice Number:</strong> {invoice_number}</p>' if invoice_number else ''}
                <p><strong>Issue:</strong> Mandatory fields are present but not in the correct format required by the API.</p>
                <p><strong>Action Required:</strong> Please review and correct the field formats as listed below.</p>
                
                <div class="error-list">
                    <h3>Field Format Errors:</h3>
                    <ul>
                        {error_details}
                    </ul>
                </div>
                
                <p><strong>Note:</strong> This is an internal mapping issue. The fields are present but need to be formatted correctly according to API requirements.</p>
            </div>
            <div class="footer">
                <p>This is an automated alert from the Pando Invoice Processing System.</p>
                <p>Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
            </div>
        </body>
        </html>
        """
        
        # Create plain text version
        plain_text = f"""
URGENT: Field Format Validation Failed

File: {filename}
{f'Invoice Number: {invoice_number}' if invoice_number else ''}

Issue: Mandatory fields are present but not in the correct format required by the API.
Action Required: Please review and correct the field formats as listed below.

Field Format Errors:
{chr(10).join([f"- {error.get('field_name', 'Unknown field')}: {error.get('explanation', 'Format validation failed')}" for error in internal_field_errors])}

Note: This is an internal mapping issue. The fields are present but need to be formatted correctly according to API requirements.

This is an automated alert from the Pando Invoice Processing System.
Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
        """
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email
        
        # Handle multiple recipients
        if isinstance(to_emails, list):
            msg["To"] = ", ".join(to_emails)
            recipient_list = to_emails
        else:
            msg["To"] = to_emails
            recipient_list = [to_emails]
        
        # Attach parts
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)
        
        # Send email
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg, to_addrs=recipient_list)
            logger.info(f"Internal field error alert sent successfully to Pando team: {', '.join(recipient_list)}")
            
    except Exception as e:
        logger.error(f"Failed to send internal field error alert: {str(e)}")


def send_received_status_email(
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    attachments: List[Dict[str, Any]],
    email_id: str,
    message_id: Optional[str] = None,
    original_sender: Optional[str] = None,
    original_date: Optional[str] = None,
    original_subject: Optional[str] = None,
    original_body: Optional[str] = None
) -> tuple:
    """
    Send a "received" status email to notify the sender that attachments have been picked up for processing.
    This is sent after carrier identification and authorization check passes.
    
    Returns:
        tuple: (message_id, email_body, email_subject, email_sender, email_date)
    """
    try:
        logger.info(f"Sending RECEIVED status email to: {to_email}")
        
        msg = MIMEMultipart("alternative")
        
        # Set subject with "Re:" prefix if not already present
        if not subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: {subject}"
        else:
            msg["Subject"] = subject
            
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Set proper threading headers
        if message_id:
            # Ensure message ID has angle brackets for proper threading
            if not message_id.startswith('<'):
                formatted_message_id = f"<{message_id}>"
            else:
                formatted_message_id = message_id
                
            logger.info(f"Setting email threading headers with message_id: {formatted_message_id}")
            msg["In-Reply-To"] = formatted_message_id
            msg["References"] = formatted_message_id
            
            # Add thread-topic for Outlook compatibility
            clean_subject = subject.replace("Re: ", "").replace("RE: ", "").strip()
            msg["Thread-Topic"] = clean_subject
            msg["Thread-Index"] = message_id.strip('<>')  # For Outlook
        
        # Build attachment list
        attachment_list = ""
        if attachments:
            attachment_list = "<ul>"
            for att in attachments:
                filename = att.get('filename', 'Unknown')
                attachment_list += f"<li>{filename}</li>"
            attachment_list += "</ul>"
        else:
            attachment_list = "<p>No attachments found</p>"
        
        # Build quoted block if info is provided
        quoted_block = ""
        if original_sender and original_date and original_subject and original_body:
            logger.info(f"Adding quoted content from: {original_sender}, date: {original_date}")
            quoted_block = f"""
            <div style='margin-top: 20px; border-top: 1px solid #ddd;'>
                <div style='color:gray; font-size:small; margin: 10px 0;'>
                    On {original_date}, {original_sender} wrote:<br>
                    <b>Subject:</b> {original_subject}
                </div>
                <div style='border-left:4px solid #ccc; padding-left:15px; margin:10px 0;'>
                    {original_body}
                </div>
            </div>
            """
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .status-box {{
                    background-color: #e8f5e9;
                    border-left: 4px solid #4caf50;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }}
            </style>
        </head>
        <body>
            <p>Dear Recipient,</p>
            
            <div class="status-box">
                <h2 style="color: #2e7d32; margin-top: 0;">Attachments Picked Up for Processing</h2>
                <p><strong>Status: RECEIVED</strong></p>
                <p>Your email attachments have been received and are being processed:</p>
                {attachment_list}
            </div>
            
            <p>Best regards,<br><strong>PI</strong></p>
            
            <div class="disclaimer">
                This is an automated email from PiAgent. Please do not reply to this email.
            </div>
            {quoted_block}
        </body>
        </html>
        """
        
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)
        
        # Send the email
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            logger.info(f"SMTP connection established, logging in as: {smtp_username}")
            server.login(smtp_username, smtp_password)
            logger.info(f"SMTP login successful, sending RECEIVED status email to: {to_email}")
            server.send_message(msg)
            logger.info("RECEIVED status email sent successfully!")
        
        # Return message details for threading
        return (message_id or msg['Message-ID'], html_content, msg['Subject'], to_email, original_date or datetime.now().isoformat())
            
    except Exception as e:
        logger.error(f"Error sending RECEIVED status email: {str(e)}", exc_info=True)
        raise


def send_authorization_rejection_email(
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    email_id: str,
    message_id: Optional[str] = None,
    original_sender: Optional[str] = None,
    original_date: Optional[str] = None,
    original_subject: Optional[str] = None,
    original_body: Optional[str] = None,
    carrier: Optional[str] = None
) -> None:
    """
    Send a rejection email when sender is not authorized to send documents for the identified carrier.
    
    Args:
        smtp_server: SMTP server address
        smtp_port: SMTP server port
        smtp_username: SMTP username
        smtp_password: SMTP password
        from_email: Sender email address
        to_email: Recipient email address
        subject: Email subject
        email_id: Email identifier
        message_id: Original message ID for threading
        original_sender: Original sender email
        original_date: Original email date
        original_subject: Original email subject
        original_body: Original email body
        carrier: Identified carrier name
    """
    try:
        logger.info(f"Sending authorization rejection email to: {to_email}")
        
        msg = MIMEMultipart("alternative")
        
        # Set subject with "Re:" prefix if not already present
        if not subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: {subject}"
        else:
            msg["Subject"] = subject
            
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Set proper threading headers
        if message_id:
            # Ensure message ID has angle brackets for proper threading
            if not message_id.startswith('<'):
                formatted_message_id = f"<{message_id}>"
            else:
                formatted_message_id = message_id
                
            logger.info(f"Setting email threading headers with message_id: {formatted_message_id}")
            msg["In-Reply-To"] = formatted_message_id
            msg["References"] = formatted_message_id
            
            # Add thread-topic for Outlook compatibility
            clean_subject = subject.replace("Re: ", "").replace("RE: ", "").strip()
            msg["Thread-Topic"] = clean_subject
            msg["Thread-Index"] = message_id.strip('<>')  # For Outlook
        
        # Build quoted block if info is provided
        quoted_block = ""
        if original_sender and original_date and original_subject and original_body:
            logger.info(f"Adding quoted content from: {original_sender}, date: {original_date}")
            quoted_block = f"""
            <div style='margin-top: 20px; border-top: 1px solid #ddd;'>
                <div style='color:gray; font-size:small; margin: 10px 0;'>
                    On {original_date}, {original_sender} wrote:<br>
                    <b>Subject:</b> {original_subject}
                </div>
                <div style='border-left:4px solid #ccc; padding-left:15px; margin:10px 0;'>
                    {original_body}
                </div>
            </div>
            """
        
        # Format carrier-specific message
        if carrier:
            carrier_upper = carrier.upper()
            if 'MAGNO' in carrier_upper:
                carrier_message = f"You are not authorized to send Magno documents. Please send only documents that match your email domain."
            elif 'KWE' in carrier_upper:
                carrier_message = f"You are not authorized to send KWE documents. Please send only documents that match your email domain."
            else:
                carrier_message = f"You are not authorized to send {carrier} documents. Please send only documents that match your email domain."
        else:
            carrier_message = "You are not authorized to send this type of document."
        
        html_content = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .rejection-box {{
                    background-color: #fff3f3;
                    border-left: 4px solid #dc3545;
                    padding: 20px;
                    margin: 20px 0;
                }}
                .disclaimer {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }}
            </style>
        </head>
        <body>
            <p>Dear Recipient,</p>
            
            <div class="rejection-box">
                <h2 style="color: #dc3545; margin-top: 0;">You are not authorized to send this type of document</h2>
                <p><strong>Status: REJECTED</strong></p>
                <p>{carrier_message}</p>
            </div>
            
            <p>Best regards,<br><strong>PI</strong></p>
            
            <div class="disclaimer">
                This is an automated email from PiAgent. Please do not reply to this email.
            </div>
            {quoted_block}
        </body>
        </html>
        """
        
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)
        
        # Send the email
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            logger.info(f"SMTP connection established, logging in as: {smtp_username}")
            server.login(smtp_username, smtp_password)
            logger.info(f"SMTP login successful, sending authorization rejection email to: {to_email}")
            server.send_message(msg)
            logger.info("Authorization rejection email sent successfully!")
            
    except Exception as e:
        logger.error(f"Error sending authorization rejection email: {str(e)}", exc_info=True)
        raise


def send_error_notification(
    smtp_server: str,
    smtp_port: int,
    smtp_username: str,
    smtp_password: str,
    from_email: str,
    to_email: str,
    subject: str,
    original_body: str,
    filename: str,
    message_id: Optional[str] = None,
    validation_errors: Dict[str, List] = None,
    quoted_sender: Optional[str] = None,
    quoted_date: Optional[str] = None,
    quoted_subject: Optional[str] = None,
    quoted_body: Optional[str] = None
) -> None:
    """Send notification email for validation errors."""
    try:
        logger.info(f"Preparing error notification to {to_email}")
        logger.info(f"Email subject: {subject}")
        logger.info(f"Filename: {filename}")
        logger.info(f"Validation errors: {json.dumps(validation_errors, indent=2)}")
        
        # Check if there are required field errors
        required_errors = []
        if validation_errors:
            required_errors = validation_errors.get('required_field_errors', [])
            logger.info(f"Found {len(required_errors)} required field errors")
            if required_errors:
                for i, err in enumerate(required_errors):
                    logger.info(f"  Error {i+1}: Field '{err.get('field')}' - {err.get('message')}")
        
        # Only send email if there are required field errors
        if not required_errors:
            logger.info("No required field errors present, skipping error notification email.")
            return
        
        msg = MIMEMultipart("alternative")
        
        # Set subject with "Re:" prefix if not already present
        if not subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: {subject}"
        else:
            msg["Subject"] = subject
            
        logger.info(f"Email subject set to: {msg['Subject']}")
        msg["From"] = from_email
        msg["To"] = to_email
        
        # Set proper threading headers
        if message_id:
            # Ensure message ID has angle brackets for proper threading
            if not message_id.startswith('<'):
                formatted_message_id = f"<{message_id}>"
            else:
                formatted_message_id = message_id
            
            logger.info(f"Setting email threading headers with message_id: {formatted_message_id}")
            msg["In-Reply-To"] = formatted_message_id
            msg["References"] = formatted_message_id
            
            # Add thread-topic for Outlook compatibility
            clean_subject = subject.replace("Re: ", "").replace("RE: ", "").strip()
            msg["Thread-Topic"] = clean_subject
            msg["Thread-Index"] = message_id.strip('<>')  # For Outlook
        
        # Prepare error HTML for required field errors only
        error_html = '''
        <div class="validation-errors" style="background-color: #fff3f3; padding: 20px; margin: 20px 0;">
            <h3 style="color: #dc3545;">Required Details Missing</h3>
            <p>The following required details are missing from the invoice, making it invalid:</p>
            <ul style="margin-left: 20px;">
        '''
        for err in required_errors:
            error_html += f'<li><strong>{err["field"]}</strong>: {err["message"]}</li>'
        error_html += '''
            </ul>
            <p>Please provide the missing information and resubmit the document.</p>
        </div>
        '''

        # Build quoted block if info is provided
        quoted_block = ""
        if quoted_sender and quoted_date and quoted_subject and quoted_body:
            logger.info(f"Adding quoted content from: {quoted_sender}, date: {quoted_date}")
            quoted_block = f"""
            <div style='margin-top: 20px; border-top: 1px solid #ddd;'>
                <div style='color:gray; font-size:small; margin: 10px 0;'>
                    On {quoted_date}, {quoted_sender} wrote:<br>
                    <b>Subject:</b> {quoted_subject}
                </div>
                <div style='border-left:4px solid #ccc; padding-left:15px; margin:10px 0;'>
                    {quoted_body}
                </div>
            </div>
            """
        
        html_text = f"""
        <html>
        <head>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .disclaimer {{
                    font-size: 12px;
                    color: #666;
                    margin-top: 20px;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-left: 4px solid #007bff;
                }}
                .file-info {{
                    background-color: #f8f9fa;
                    padding: 15px;
                    margin: 15px 0;
                    border-radius: 4px;
                }}
            </style>
        </head>
        <body>
            <div class="disclaimer">
                This is an automated email from PiAgent. Please do not reply to this email.
            </div>
            <p>Dear Recipient,</p>
            <div class="file-info">
                <p><strong>File Name:</strong> {filename}</p>
                <p><strong>Original Subject:</strong> {subject}</p>
            </div>
            
            {error_html}
            <p>Best regards,<br><strong>PI</strong></p>
            
            {quoted_block}
        </body>
        </html>
        """
        
        logger.info(f"HTML email content length: {len(html_text)} characters")
        
        html_part = MIMEText(html_text, "html")
        msg.attach(html_part)
        
        # Send the email
        logger.info(f"Connecting to SMTP server: {smtp_server}:{smtp_port}")
        context = ssl.create_default_context()
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls(context=context)
            logger.info(f"SMTP connection established, logging in as: {smtp_username}")
            server.login(smtp_username, smtp_password)
            logger.info(f"SMTP login successful, sending email to: {to_email}")
            server.send_message(msg)
            logger.info("Error notification sent successfully!")
            
    except Exception as e:
        logger.error(f"Error sending error notification: {str(e)}", exc_info=True)
        raise

def invoke_validator_lambda(lambda_arn, extracted_info, validation_method, input_type):
    """Invoke a validator Lambda function to validate the extracted information."""
    logger.info(f"Invoking validator Lambda: {lambda_arn}")
    logger.info(f"Validation method: {validation_method}, input_type: {input_type}")
    
    payload = {
        "input_type": input_type,
        "validation_method": validation_method,
        "data": {
            "Invoice": extracted_info
        }
    }
    
    logger.info(f"Validator payload keys: {list(payload.keys())}")
    logger.info(f"Data structure: {list(payload['data'].keys())}")
    logger.info(f"Validator payload prepared")
    
    try:
        logger.info("Sending request to validator Lambda")
        response = lambda_client.invoke(
            FunctionName=lambda_arn,
            InvocationType='RequestResponse',
            Payload=json.dumps(payload)
        )
        
        # Check for Lambda execution errors
        if 'FunctionError' in response:
            logger.error(f"Validator Lambda execution error: {response.get('FunctionError')}")
            logger.error(f"Error details: {response.get('Payload', '').read().decode('utf-8')}")
            return None
            
        # Read and decode the response payload
        response_payload = response['Payload'].read().decode('utf-8')
        logger.info(f"Validator Lambda response received, payload length: {len(response_payload)}")
        
        # Parse the JSON response
        result = json.loads(response_payload)
        logger.info(f"Validator result keys: {list(result.keys())}")
        logger.info(f"Validator result received")
        
        # Log validation errors if present
        if 'validation_errors' in result:
            validation_errors = result['validation_errors']
            logger.info(f"Validation errors found: {validation_errors}")
        else:
            logger.info("No validation errors found in response")
            
        return result
    except Exception as e:
        logger.error(f"Error invoking validator Lambda: {str(e)}", exc_info=True)
        return None

def extract_validation_errors(validation_result, validation_method, input_type):
    """Extract validation errors from the validator Lambda response."""
    logger.info(f"Extracting validation errors from result")
    # Suppressing full validation result dump to reduce log volume
    
    if not validation_result or not isinstance(validation_result, dict):
        logger.warning("Invalid validation result format")
        return {"required_field_errors": [], "other_field_errors": []}
    
    # Check if validation_errors key exists
    if 'validation_errors' not in validation_result:
        logger.warning("No validation_errors key in validation result")
        return {"required_field_errors": [], "other_field_errors": []}
    
    validation_errors = validation_result['validation_errors']
    
    # Initialize result structure
    result = {
        "required_field_errors": [],
        "other_field_errors": []
    }
    
    # Process errors based on validation method and input type
    if validation_method == "generic" and input_type == "invoice":
        logger.info("Processing generic invoice validation errors")
        
        # Extract required field errors
        required_errors = validation_errors.get('required_fields', {})
        logger.info(f"Found {len(required_errors)} required field errors")
        
        for field_name, error_msg in required_errors.items():
            error_entry = {
                "field": field_name,
                "message": error_msg
            }
            result["required_field_errors"].append(error_entry)
            logger.info(f"Required field error: {field_name} - {error_msg}")
        
        # Extract other field errors
        other_errors = validation_errors.get('other_errors', {})
        logger.info(f"Found {len(other_errors)} other field errors")
        
        for field_name, error_msg in other_errors.items():
            error_entry = {
                "field": field_name,
                "message": error_msg
            }
            result["other_field_errors"].append(error_entry)
            logger.info(f"Other field error: {field_name} - {error_msg}")
    
    logger.info(f"Extracted {len(result['required_field_errors'])} required field errors and {len(result['other_field_errors'])} other field errors")
    logger.info(f"Validation errors extraction complete")
    return result

def validate_external_mandatory_fields(payload):
    """
    Validates that all mandatory fields are present and non-empty (external validation).
    This is the first check - if this fails, no API call should be made.
    
    Args:
        payload (dict): The API payload to validate
        
    Returns:
        tuple: (is_valid: bool, external_field_errors: list)
    """
    logger.info("=== VALIDATING EXTERNAL MANDATORY FIELDS (EMPTY CHECK) ===")
    
    external_field_errors = []
    
    def add_external_error(field_name, explanation):
        external_field_errors.append({
            "field_name": field_name,
            "explanation": explanation
        })
    
    # Check if payload has data array
    if not isinstance(payload, dict) or "data" not in payload:
        add_external_error("payload.data", "Payload data array is missing")
        return False, external_field_errors
    
    data_array = payload.get("data", [])
    if not isinstance(data_array, list) or len(data_array) == 0:
        add_external_error("payload.data", "Payload data array is empty")
        return False, external_field_errors
    
    # Validate each data item
    for i, data_item in enumerate(data_array):
        if not isinstance(data_item, dict):
            add_external_error(f"data[{i}]", "Data item is not a valid object")
            continue
            
        logger.info(f"Validating external fields for data item {i}")
        
        # 1. Top-level mandatory fields
        top_level_fields = [
            "invoice_number", "invoice_date", "payment_due_date", "vendor_reference_id",
            "currency", "total_invoice_value", "bill_of_lading_number", "bill_to_name",
            "bill_to_address"
        ]
        
        for field in top_level_fields:
            value = data_item.get(field)
            if not value or (isinstance(value, str) and value.strip() == ""):
                add_external_error(field, "Field is empty or missing")
        
        # 2. Check documents_attachment and document_extraction
        documents_attachment = data_item.get("documents_attachment", [])
        if not isinstance(documents_attachment, list) or len(documents_attachment) == 0:
            add_external_error("documents_attachment", "Documents attachment array is missing or empty")
        else:
            # Validate each document attachment
            for k, doc in enumerate(documents_attachment):
                if not isinstance(doc, dict):
                    add_external_error(f"documents_attachment[{k}]", "Document attachment is not a valid object")
                    continue
                
                # Check required fields in each document attachment
                required_doc_fields = ["file_path", "bucket_name", "file_name", "file_extension", "type"]
                for field in required_doc_fields:
                    value = doc.get(field)
                    if not value or (isinstance(value, str) and value.strip() == ""):
                        add_external_error(f"documents_attachment[{k}].{field}", f"Document attachment {field} is empty or missing")
        
        document_extraction = data_item.get("document_extraction", {})
        if not isinstance(document_extraction, dict):
            add_external_error("document_extraction", "Document extraction is not a valid object")
        else:
            # Check required fields in document extraction
            required_extraction_fields = ["file_path", "bucket_name"]
            for field in required_extraction_fields:
                value = document_extraction.get(field)
                if not value or (isinstance(value, str) and value.strip() == ""):
                    add_external_error(f"document_extraction.{field}", f"Document extraction {field} is empty or missing")
        
        # 3. Validate shipments
        shipments = data_item.get("shipments", [])
        if not isinstance(shipments, list) or len(shipments) == 0:
            add_external_error("shipments", "Shipments array is missing or empty")
        else:
            for j, shipment in enumerate(shipments):
                if not isinstance(shipment, dict):
                    add_external_error(f"shipments[{j}]", "Shipment is not a valid object")
                    continue
                
                # Shipment mandatory fields
                shipment_fields = [
                    "shipment_number", "mode", "source_name", "destination_name",
                    "source_city", "source_country", "source_address",
                    "destination_city", "destination_country", "destination_address",
                    "shipment_weight", "shipment_weight_uom", "shipment_total_value",
                    "shipment_creation_date", "port_of_loading", "origin_service_type",
                    "destination_service_type", "port_of_discharge"
                ]
                
                for field in shipment_fields:
                    value = shipment.get(field)
                    if not value or (isinstance(value, str) and value.strip() == ""):
                        add_external_error(f"shipments[{j}].{field}", "Field is empty or missing")
                
                # 4. Validate custom fields in shipment
                custom = shipment.get("custom", {})
                if not isinstance(custom, dict):
                    add_external_error(f"shipments[{j}].custom", "Custom object is missing")
                else:
                    custom_fields = [
                        "service_code", "hazardous_material", "actual_weight",
                        "actual_weight_uom", "total_package", "cargo", "temperature_control"
                    ]
                    
                    for field in custom_fields:
                        value = custom.get(field)
                        if value is None or value == "":
                            add_external_error(f"shipments[{j}].custom.{field}", "Field is empty or missing")
    
    is_valid = len(external_field_errors) == 0
    
    logger.info(f"External validation completed: Valid={is_valid}, External errors={len(external_field_errors)}")
    
    if external_field_errors:
        logger.warning(f"External field errors: {external_field_errors}")
    
    return is_valid, external_field_errors


def validate_internal_field_formats(payload):
    """
    Validates that all mandatory fields are in the correct format (internal validation).
    This is the second check - if this fails, no API call should be made and an instant email alert should be sent.
    
    Args:
        payload (dict): The API payload to validate
        
    Returns:
        tuple: (is_valid: bool, internal_field_errors: list)
    """
    logger.info("=== VALIDATING INTERNAL FIELD FORMATS ===")
    
    internal_field_errors = []
    
    def add_internal_error(field_name, explanation):
        internal_field_errors.append({
            "field_name": field_name,
            "explanation": explanation
        })
    
    # Check if payload has data array
    if not isinstance(payload, dict) or "data" not in payload:
        add_internal_error("payload.data", "Payload data array is missing")
        return False, internal_field_errors
    
    data_array = payload.get("data", [])
    if not isinstance(data_array, list) or len(data_array) == 0:
        add_internal_error("payload.data", "Payload data array is empty")
        return False, internal_field_errors
    
    # Validate each data item
    for i, data_item in enumerate(data_array):
        if not isinstance(data_item, dict):
            add_internal_error(f"data[{i}]", "Data item is not a valid object")
            continue
            
        logger.info(f"Validating internal formats for data item {i}")
        
        # 1. Top-level field format validation
        for field in ["invoice_date", "payment_due_date"]:
            value = data_item.get(field)
            if value and isinstance(value, str) and value.strip() != "":
                # Validate ISO date format
                if not value.endswith("Z") or "T" not in value:
                    add_internal_error(field, "Date must be in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
        
        if "total_invoice_value" in data_item:
            value = data_item.get("total_invoice_value")
            if value is not None:
                # Validate numeric value
                if not isinstance(value, (int, float)) or value <= 0:
                    add_internal_error("total_invoice_value", "Must be a positive number")
        
        # 2. Validate shipments array formats
        shipments = data_item.get("shipments", [])
        if isinstance(shipments, list):
            for j, shipment in enumerate(shipments):
                if not isinstance(shipment, dict):
                    continue
                
                # Validate shipment field formats
                for field in ["shipment_creation_date"]:
                    value = shipment.get(field)
                    if value and isinstance(value, str) and value.strip() != "":
                        # Validate ISO date format
                        if not value.endswith("Z") or "T" not in value:
                            add_internal_error(f"shipments[{j}].{field}", "Date must be in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)")
                
                for field in ["shipment_weight", "shipment_total_value"]:
                    value = shipment.get(field)
                    if value is not None:
                        # Validate numeric value
                        if not isinstance(value, (int, float)) or value <= 0:
                            add_internal_error(f"shipments[{j}].{field}", "Must be a positive number")
                
                if "shipment_weight_uom" in shipment:
                    value = shipment.get("shipment_weight_uom")
                    if value and isinstance(value, str) and value.strip() != "":
                        # Must be "KG" only (2 characters)
                        if value != "KG":
                            add_internal_error(f"shipments[{j}].shipment_weight_uom", "Must be 'KG' only (2 characters)")
                
                for field in ["source_country", "destination_country"]:
                    value = shipment.get(field)
                    if value and isinstance(value, str) and value.strip() != "":
                        # Must be 2 character country code
                        if len(value) != 2:
                            add_internal_error(f"shipments[{j}].{field}", "Must be 2 character country code")
                
                for field in ["port_of_loading", "port_of_discharge"]:
                    value = shipment.get(field)
                    if value and isinstance(value, str) and value.strip() != "":
                        # Must be 5 character code
                        if len(value) != 5:
                            add_internal_error(f"shipments[{j}].{field}", "Must be 5 character code")
                
                for field in ["origin_service_type", "destination_service_type"]:
                    value = shipment.get(field)
                    if value and isinstance(value, str) and value.strip() != "":
                        # Must be "D" or "A"
                        if value not in ["D", "A"]:
                            add_internal_error(f"shipments[{j}].{field}", "Must be 'D' or 'A'")
                
                # 3. Validate custom fields in shipment
                custom = shipment.get("custom", {})
                if isinstance(custom, dict):
                    if "actual_weight" in custom:
                        value = custom.get("actual_weight")
                        if value is not None:
                            # Validate numeric value
                            if not isinstance(value, (int, float)) or value <= 0:
                                add_internal_error(f"shipments[{j}].custom.actual_weight", "Must be a positive number")
                    
                    if "actual_weight_uom" in custom:
                        value = custom.get("actual_weight_uom")
                        if value and isinstance(value, str) and value.strip() != "":
                            # Must be "KG" only (2 characters)
                            if value != "KG":
                                add_internal_error(f"shipments[{j}].custom.actual_weight_uom", "Must be 'KG' only (2 characters)")
                    
                    for field in ["hazardous_material", "cargo", "temperature_control"]:
                        value = custom.get(field)
                        if value is not None:
                            # Must be boolean
                            if not isinstance(value, bool):
                                add_internal_error(f"shipments[{j}].custom.{field}", "Must be boolean (true/false)")
                    
                    if "total_package" in custom:
                        value = custom.get("total_package")
                        if value is not None:
                            # Must be string or number
                            if not isinstance(value, (str, int, float)) or (isinstance(value, str) and value.strip() == ""):
                                add_internal_error(f"shipments[{j}].custom.total_package", "Must be a valid package count")
    
    is_valid = len(internal_field_errors) == 0
    
    logger.info(f"Internal validation completed: Valid={is_valid}, Internal errors={len(internal_field_errors)}")
    
    if internal_field_errors:
        logger.warning(f"Internal field errors: {internal_field_errors}")
    
    return is_valid, internal_field_errors

def validate_api_payload(payload):
    """
    Validates and completes the API payload against the reference schema.
    Adds any missing fields with default values from the schema.
    Removes specific fields like cost_center, project_code, and company_code.
    
    Args:
        payload (dict): The API payload to validate
        
    Returns:
        dict: The validated and completed payload
    """
    logger.info("=== VALIDATING API PAYLOAD ===")
    
    def ensure_field_exists(data, schema_item, path=""):
        """Recursively ensure all fields from schema exist in data."""
        if isinstance(schema_item, dict):
            for key, value in schema_item.items():
                current_path = f"{path}.{key}" if path else key
                if key not in data:
                    data[key] = value
                    logger.info(f"Added missing field: {current_path} = {value}")
                elif isinstance(value, dict):
                    ensure_field_exists(data[key], value, current_path)
                elif isinstance(value, list) and value:
                    # Handle list of objects
                    if isinstance(value[0], dict):
                        if not isinstance(data[key], list):
                            data[key] = []
                        for i, item in enumerate(data[key]):
                            if isinstance(item, dict):
                                ensure_field_exists(item, value[0], f"{current_path}[{i}]")
        elif isinstance(schema_item, list) and schema_item:
            if not isinstance(data, list):
                data = []
            for i, item in enumerate(data):
                if isinstance(item, dict) and isinstance(schema_item[0], dict):
                    ensure_field_exists(item, schema_item[0], f"{path}[{i}]")
    
    # Process the payload data array
    if "data" in payload and isinstance(payload["data"], list):
        for i, data_item in enumerate(payload["data"]):
            logger.info(f"Validating data item {i}")
            ensure_field_exists(data_item, API_PAYLOAD_SCHEMA, f"data[{i}]")
    
    logger.info("=== API PAYLOAD VALIDATION COMPLETED ===")
    return payload


def calculate_payment_due_date(invoice_date: str) -> str:
    """
    Calculate payment due date by adding 90 days to the invoice date.
    
    Args:
        invoice_date: Invoice date in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)
        
    Returns:
        Payment due date in ISO format (YYYY-MM-DDTHH:MM:SS.sssZ)
    """
    try:
        if not invoice_date or not isinstance(invoice_date, str):
            logger.warning(f"Invalid invoice_date: {invoice_date}")
            return ""
        
        # Parse the invoice date
        # Handle different possible formats
        if invoice_date.endswith('Z'):
            # ISO format with Z suffix
            invoice_dt = datetime.fromisoformat(invoice_date.replace('Z', '+00:00'))
        elif 'T' in invoice_date:
            # ISO format without Z suffix
            invoice_dt = datetime.fromisoformat(invoice_date)
        else:
            # Try parsing as date only
            try:
                invoice_dt = datetime.strptime(invoice_date, '%Y-%m-%d')
            except ValueError:
                logger.error(f"Unable to parse invoice_date: {invoice_date}")
                return ""
        
        # Add 90 days
        from datetime import timedelta
        due_date = invoice_dt + timedelta(days=90)
        
        # Convert back to ISO format with Z suffix
        due_date_iso = due_date.strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        logger.info(f"📅 Calculated payment_due_date: {invoice_date} + 90 days = {due_date_iso}")
        return due_date_iso
        
    except Exception as e:
        logger.error(f"Error calculating payment due date from {invoice_date}: {e}")
        return ""




def list_files_in_s3_path(s3_path: str) -> List[Dict[str, str]]:
    """
    List all files in the given S3 path and return their details.
    
    Args:
        s3_path: S3 path in format s3://bucket-name/path/
        
    Returns:
        List of dictionaries containing file details
    """
    try:
        if not s3_path or not s3_path.startswith('s3://'):
            logger.warning(f"Invalid S3 path: {s3_path}")
            return []
        
        # Parse S3 path
        s3_path_clean = s3_path.replace('s3://', '')
        if '/' in s3_path_clean:
            bucket_name = s3_path_clean.split('/')[0]
            prefix = '/'.join(s3_path_clean.split('/')[1:])
        else:
            bucket_name = s3_path_clean
            prefix = ""
        
        logger.info(f"Listing files in S3 bucket: {bucket_name}, prefix: {prefix}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # List objects
        response = s3_client.list_objects_v2(
            Bucket=bucket_name,
            Prefix=prefix
        )
        
        files = []
        if 'Contents' in response:
            for obj in response['Contents']:
                # Skip directories (objects ending with '/')
                if obj['Key'].endswith('/'):
                    continue
                
                file_name = obj['Key'].split('/')[-1]
                file_extension = os.path.splitext(file_name)[1].lower().replace('.', '')
                
                files.append({
                    'file_path': obj['Key'],
                    'bucket_name': bucket_name,
                    'file_name': file_name,
                    'file_extension': file_extension or 'pdf',  # Default to pdf if no extension
                    'type': 'Freight Invoice Docket'  # Default type for all email attachments
                })
                
                logger.info(f"Found file: {file_name} ({file_extension})")
        
        logger.info(f"Total files found: {len(files)}")
        return files
        
    except Exception as e:
        logger.error(f"Error listing files in S3 path {s3_path}: {e}")
        return []


class APIHandler:
    def _get_full_vendor_name(self, carrier_name):
        """Map carrier name to full vendor name."""
        vendor_mapping = {
            "MAGNO": "MAGNO INTERNATIONAL LP",
            "KWE": "KINTETSU WORLD EXPRESS(USA)INC"
        }
        return vendor_mapping.get(carrier_name, carrier_name)
    
    def _normalize_boolean(self, value):
        """Normalize boolean values to proper true/false."""
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            value_lower = value.lower().strip()
            if value_lower in ['true', '1', 'yes', 'y', 'on']:
                return True
            elif value_lower in ['false', '0', 'no', 'n', 'off']:
                return False
        return value  # Return original if can't normalize
    
    def _normalize_number(self, value, field_type='float'):
        """Normalize numeric values."""
        if isinstance(value, (int, float)):
            return value
        if isinstance(value, str):
            # Remove common formatting
            cleaned = value.replace(',', '').replace('$', '').replace('%', '').strip()
            
            # First, try to extract the first number from the string
            # This handles cases like "1 Piece (s)", "163.000 KGS", "5 boxes", etc.
            number_match = re.search(r'(\d+\.?\d*)', cleaned)
            if number_match:
                cleaned = number_match.group(1)
            else:
                # If no number found, try removing units first and then extracting
                # Remove common weight/unit strings (KGS, KG, lbs, lb, Piece, etc.)
                cleaned = re.sub(r'\s*(KGS?|LBS?|KG|kg|g|G|OZ|oz|TON|ton|Piece|piece|Pieces|pieces|Box|box|Boxes|boxes)\s*', '', cleaned, flags=re.IGNORECASE)
                # Try extracting number again after unit removal
                number_match = re.search(r'(\d+\.?\d*)', cleaned)
                if number_match:
                    cleaned = number_match.group(1)
            
            try:
                if field_type == 'int':
                    return int(float(cleaned))
                else:
                    return float(cleaned)
            except (ValueError, TypeError):
                logger.warning(f"Could not normalize number from '{value}', defaulting to 0")
                return 0
        return 0
    
    def _normalize_string(self, value):
        """Normalize string values."""
        if value is None:
            return ""
        if isinstance(value, (int, float)):
            return str(value)
        if isinstance(value, str):
            return value.strip()
        return str(value)
    
    def _apply_field_constraints(self, data):
        """Apply data type constraints and normalization to payload data."""
        logger.info("=== APPLYING FIELD CONSTRAINTS ===")
        
        # Define field type constraints
        field_constraints = {
            # String fields
            "invoice_number": "string",
            "invoice_date": "string", 
            "payment_due_date": "string",
            "payment_terms": "string",
            "vendor_reference_id": "string",
            "currency": "string",
            "bill_of_lading_number": "string",
            "bill_to_name": "string",
            "bill_to_gst": "string",
            "bill_to_address": "string",
            "bill_to_phone_number": "string",
            "bill_to_email": "string",
            "billing_entity_name": "string",
            "shipment_number": "string",
            "mode": "string",
            "pro_number": "string",
            "source_name": "string",
            "destination_name": "string",
            "source_code": "string",
            "source_city": "string",
            "source_state": "string",
            "source_country": "string",
            "source_zone": "string",
            "source_zip_code": "string",
            "source_address": "string",
            "destination_code": "string",
            "destination_city": "string",
            "destination_state": "string",
            "destination_country": "string",
            "destination_zone": "string",
            "destination_zip_code": "string",
            "destination_address": "string",
            "shipment_weight_uom": "string",
            "shipment_volume_uom": "string",
            "shipment_distance_uom": "string",
            "shipment_creation_date": "string",
            "charge_code": "string",
            "charge_name": "string",
            "currency": "string",
            "port_of_loading": "string",
            "origin_service_type": "string",
            "destination_service_type": "string",
            "port_of_discharge": "string",
            "service_code": "string",
            "container_type": "string",
            "special_handling": "string",
            "unnumber_count": "string",
            "data_loggercount": "string",
            "container_count": "string",
            "origin_terminal_handling_days_count": "string",
            "destination_terminal_handling_days_count": "string",
            "thermal_blanket": "string",
            "uld_extra_lease_day": "string",
            "actual_weight_uom": "string",
            "lane_id": "string",
            "charge_code": "string",
            "charge_name": "string",
            "reference_number": "string",
            "special_instructions": "string",
            "priority": "string",
            "pay_as_present": "string",
            "vendor_name": "string",
            "shipper_email": "string",
            "sender_email": "string",
            
            # Numeric fields
            "total_invoice_value": "float",
            "total_tax_amount": "int",
            "shipment_weight": "float",
            "shipment_volume": "int",
            "shipment_distance": "int",
            "shipment_total_value": "float",
            "shipment_tax_value": "int",
            "actual_weight": "float",
            "total_package": "int",
            "charge_gross_amount": "float",
            "charge_tax_amount": "int",
            "client_id": "int",
            
            # Boolean fields
            "hazardous_material": "boolean",
            "temperature_control": "boolean",
            "cargo": "boolean",
            
        }
        
        def process_nested_data(obj, path=""):
            """Recursively process nested data structures."""
            if isinstance(obj, dict):
                for key, value in obj.items():
                    current_path = f"{path}.{key}" if path else key
                    
                    # Apply constraints based on field type
                    if key in field_constraints:
                        constraint_type = field_constraints[key]
                        
                        if constraint_type == "string":
                            obj[key] = self._normalize_string(value)
                        elif constraint_type == "float":
                            obj[key] = self._normalize_number(value, 'float')
                        elif constraint_type == "int":
                            obj[key] = self._normalize_number(value, 'int')
                        elif constraint_type == "boolean":
                            obj[key] = self._normalize_boolean(value)
                        
                        logger.info(f"✅ Applied {constraint_type} constraint to {current_path}: {obj[key]}")
                    
                    # Recursively process nested objects
                    if isinstance(value, (dict, list)):
                        process_nested_data(value, current_path)
                        
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    current_path = f"{path}[{i}]"
                    process_nested_data(item, current_path)
        
        # Process the entire data structure
        process_nested_data(data)
        logger.info("=== FIELD CONSTRAINTS APPLIED ===")
        return data
    
    def create_invoice_payload(self, flattened_data, email_details, carrier_name, input_key, input_bucket, jj_result=None, output_bucket=None, all_final_jsons=None, email_attachments=None):
        """Create the comprehensive invoice payload for the external API."""
        logger.info("=== CREATING COMPREHENSIVE INVOICE PAYLOAD ===")
        
        # Get documents from email_attachments S3 path instead of clustered PDFs
        documents_attachment = []
        
        if email_attachments:
            logger.info(f"Processing email_attachments from S3 path: {email_attachments}")
            documents_attachment = list_files_in_s3_path(email_attachments)
            logger.info(f"Found {len(documents_attachment)} files in email_attachments")
        else:
            logger.info("No email_attachments provided, falling back to clustered PDFs")
            # Fallback to original clustered PDF logic if email_attachments not provided
            if jj_result and 'clustering_report' in jj_result and 'saved_clusters' in jj_result['clustering_report']:
                saved_clusters = jj_result['clustering_report']['saved_clusters']
                logger.info(f"Processing {len(saved_clusters)} clustered PDFs for documents_attachment")
                
                # Only include clustered PDFs that correspond to our final JSONs (excluding freight)
                final_json_names = set(all_final_jsons.keys()) if all_final_jsons else set()
                final_json_names.discard('freight.json')  # Remove freight.json from attachments
                
                for cluster_id, cluster_info in saved_clusters.items():
                    if isinstance(cluster_info, dict):
                        # Extract cluster information from J_J.py structure
                        doc_type = cluster_info.get('doc_type', 'Unknown')
                        s3_path = cluster_info.get('s3_path', '')
                        safe_doc_type = cluster_info.get('safe_doc_type', '')
                        
                        # Map cluster types to JSON names
                        cluster_to_json_mapping = {
                            'Air Waybill': 'airwaybill.json',
                            'Commercial Invoice': 'commercial.json', 
                            'Packing List': 'packing.json',
                            'Delivery Note': 'delivery.json'
                        }
                        
                        # Only include if this cluster corresponds to one of our final JSONs
                        corresponding_json = cluster_to_json_mapping.get(doc_type)
                        if not (corresponding_json and corresponding_json in final_json_names):
                            logger.info(f"Skipping cluster {cluster_id} ({doc_type}) - not in final JSONs")
                            continue
                        
                        # Extract file name from s3_path
                        file_name = ""
                        if s3_path:
                            # s3_path format: s3://bucket-name/key/path
                            s3_path_clean = s3_path.replace('s3://', '')
                            if '/' in s3_path_clean:
                                file_name = s3_path_clean.split('/')[-1]
                        
                        # Determine file extension
                        file_extension = 'pdf'  # Default to PDF
                        if file_name:
                            if file_name.lower().endswith('.png'):
                                file_extension = 'png'
                            elif file_name.lower().endswith('.jpg') or file_name.lower().endswith('.jpeg'):
                                file_extension = 'jpg'
                            elif file_name.lower().endswith('.pdf'):
                                file_extension = 'pdf'
                        
                        # Map document types to standardized types
                        document_type = "Unknown"
                        doc_type_lower = doc_type.lower()
                        if 'airway' in doc_type_lower or 'air waybill' in doc_type_lower:
                            document_type = "Airwaybill"
                        elif 'packing' in doc_type_lower:
                            document_type = "Packing"
                        elif 'delivery' in doc_type_lower:
                            document_type = "Delivery"
                        elif 'freight' in doc_type_lower:
                            document_type = "Freight"
                        elif 'commercial' in doc_type_lower:
                            document_type = "Commercial"
                        elif 'bill_of_lading' in doc_type_lower or 'bol' in doc_type_lower:
                            document_type = "Bill of Lading"
                        
                        # Extract bucket name from s3_path
                        bucket_name = output_bucket or input_bucket  # Use output bucket from event
                        if s3_path and s3_path.startswith('s3://'):
                            s3_path_clean = s3_path.replace('s3://', '')
                            if '/' in s3_path_clean:
                                bucket_name = s3_path_clean.split('/')[0]
                        
                        # Extract file path (S3 key without bucket)
                        file_path = ""
                        if s3_path and s3_path.startswith('s3://'):
                            s3_path_clean = s3_path.replace('s3://', '')
                            if '/' in s3_path_clean:
                                file_path = '/'.join(s3_path_clean.split('/')[1:])  # Remove bucket name
                        
                        documents_attachment.append({
                            "file_path": file_path,
                            "bucket_name": bucket_name,
                            "file_name": file_name,
                            "file_extension": file_extension,
                            "type": document_type
                        })
                        
                        logger.info(f"Added document attachment: {file_name} ({document_type}) from {s3_path}")
        
        # If no documents found from either email_attachments or clustered PDFs, add the original input file
        if not documents_attachment:
            logger.info("No documents found, using original input file")
            documents_attachment.append({
                "file_path": input_key,
                "bucket_name": input_bucket,
                "file_name": os.path.basename(input_key),
                "file_extension": "pdf",
                "type": "Original"
            })
        
        # Create document_extraction paths (all final JSONs)
        main_pdf_name = os.path.basename(input_key)
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
        # Create combined attachments.json and document_extraction
        document_extraction = {}
        if all_final_jsons:
            # Collect all attachment JSONs (excluding freight.json) and flatten structure
            flattened_attachments = {}
            for json_name, json_data in all_final_jsons.items():
                if json_name != 'freight.json':  # freight.json is the main payload, others are attachments
                    # Extract the inner content from each JSON file
                    if isinstance(json_data, dict):
                        for key, value in json_data.items():
                            flattened_attachments[key] = value
            
            # Create combined documents_attached.json if we have attachments
            if flattened_attachments:
                # Save combined documents_attached.json to S3
                attachments_path = f"Output/{clean_main_name}/final_json/documents_attached.json"
                attachments_s3_key = attachments_path
                
                try:
                    # Upload combined documents_attached.json to S3
                    s3.put_object(
                        Bucket=output_bucket or input_bucket,
                        Key=attachments_s3_key,
                        Body=json.dumps(flattened_attachments, indent=2, ensure_ascii=False),
                        ContentType='application/json'
                    )
                    logger.info(f"✅ Combined documents_attached.json saved to: s3://{output_bucket or input_bucket}/{attachments_s3_key}")
                    
                    # Set document_extraction as single object (not array)
                    document_extraction = {
                        "file_path": attachments_path,
                        "bucket_name": output_bucket or input_bucket
                    }
                    
                except Exception as e:
                    logger.error(f"❌ Failed to save combined attachments.json: {e}")
                    # Fallback to empty object if saving fails
                    document_extraction = {}
        
        if document_extraction:
            logger.info(f"Document extraction path: s3://{document_extraction['bucket_name']}/{document_extraction['file_path']}")
        else:
            logger.info("No attachment documents found for document_extraction")
        
        # Create the base payload structure with freight.json data
        # Extract data from the final processed JSON structure
        freight_data = None
        if all_final_jsons and 'freight.json' in all_final_jsons:
            freight_json = all_final_jsons['freight.json']
            # Extract the data array from the freight.json structure
            if 'data' in freight_json and isinstance(freight_json['data'], list) and len(freight_json['data']) > 0:
                freight_data = freight_json['data'][0]  # Get first (and usually only) data item
                logger.info("Using final processed freight.json data for API payload")
            else:
                logger.warning("No data array found in final processed freight.json")
                freight_data = None
        else:
            logger.warning("No final processed freight.json found, using flattened_data as fallback")
            if 'data' in flattened_data and isinstance(flattened_data['data'], list) and len(flattened_data['data']) > 0:
                freight_data = flattened_data['data'][0]  # Get first (and usually only) data item
            elif isinstance(flattened_data, dict) and 'data' not in flattened_data:
                # If it's already the final processed data (no nested 'data' array)
                freight_data = flattened_data
        
        if freight_data:
            # Calculate payment due date by adding 90 days to invoice date
            invoice_date = freight_data.get("invoice_date", "")
            calculated_payment_due_date = calculate_payment_due_date(invoice_date)
            
            payload_data = {
                "invoice_number": freight_data.get("invoice_number", ""),
                "invoice_date": invoice_date,
                "payment_due_date": calculated_payment_due_date,  # Use calculated value instead of LLM extraction
                "payment_terms": freight_data.get("payment_terms", ""),
                "vendor_reference_id": freight_data.get("vendor_reference_id", ""),
                "currency": freight_data.get("currency", ""),
                "total_invoice_value": freight_data.get("total_invoice_value", 0),
                "total_tax_amount": freight_data.get("total_tax_amount", 0),
                "bill_of_lading_number": freight_data.get("bill_of_lading_number", ""),
                "bill_to_name": freight_data.get("bill_to_name", ""),
                "bill_to_gst": freight_data.get("bill_to_gst", ""),
                "bill_to_address": freight_data.get("bill_to_address", ""),
                "bill_to_phone_number": freight_data.get("bill_to_phone_number", ""),
                "bill_to_email": freight_data.get("bill_to_email", ""),
                "cost_center": freight_data.get("cost_center", ""),
                "billing_entity_name": freight_data.get("billing_entity_name", ""),
                "documents_attachment": documents_attachment,
                "document_extraction": document_extraction,
                "shipments": freight_data.get("shipments", []),
                "taxes": [],
                "custom_charges": [],
                "custom": {
                    "shipper_email": email_details.get("status_email_sender", ""),
                    "sender_email": email_details.get("to", ""),
                    "reference_number": freight_data.get("reference_number", ""),
                    "special_instructions": freight_data.get("special_instructions", ""),
                    "priority": freight_data.get("priority", ""),
                    "pay_as_present": freight_data.get("pay_as_present", ""),
                    "client_id": 33,
                    "vendor_name": self._get_full_vendor_name(carrier_name)
                },
                "shipment_identifiers": freight_data.get("shipment_identifiers", {
                    "booking_number": "",
                    "container_numbers": []
                })
            }
        else:
            # Fallback if no freight data found
            logger.warning("No freight data found in flattened_data, using empty payload")
            # Calculate payment due date even for empty payload (will return empty string)
            calculated_payment_due_date = calculate_payment_due_date("")
            
            payload_data = {
                "invoice_number": "",
                "invoice_date": "",
                "payment_due_date": calculated_payment_due_date,  # Use calculated value
                "payment_terms": "",
                "vendor_reference_id": "",
                "currency": "",
                "total_invoice_value": 0,
                "total_tax_amount": 0,
                "bill_of_lading_number": "",
                "bill_to_name": "",
                "bill_to_gst": "",
                "bill_to_address": "",
                "bill_to_phone_number": "",
                "bill_to_email": "",
                "cost_center": "",
                "billing_entity_name": "",
                "documents_attachment": documents_attachment,
                "document_extraction": document_extraction,
                "shipments": [],
                "taxes": [],
                "custom_charges": [],
                "custom": {
                    "shipper_email": email_details.get("status_email_sender", ""),
                    "sender_email": email_details.get("to", ""),
                    "reference_number": "",
                    "special_instructions": "",
                    "priority": "",
                    "pay_as_present": "",
                    "client_id": 33,
                    "vendor_name": self._get_full_vendor_name(carrier_name)
                },
                "shipment_identifiers": {
                    "booking_number": "",
                    "container_numbers": []
                }
            }
        
        logger.info(f"Created comprehensive payload with {len(payload_data)} fields")
        logger.info(f"Documents attachment count: {len(documents_attachment)}")
        if document_extraction:
            logger.info(f"Document extraction: {document_extraction['file_path']}")
        else:
            logger.info("No document extraction available")
        
        # Construct the final payload structure
        payload = {
            "data": [payload_data]
        }
        
        # Hardcode lane_id to always be empty string in all shipments
        if payload.get("data") and isinstance(payload["data"], list):
            for data_item in payload["data"]:
                if isinstance(data_item, dict) and "shipments" in data_item:
                    shipments = data_item.get("shipments", [])
                    for shipment in shipments:
                        if isinstance(shipment, dict):
                            # Ensure custom object exists
                            if "custom" not in shipment:
                                shipment["custom"] = {}
                            # Hardcode lane_id to empty string
                            shipment["custom"]["lane_id"] = ""
                            logger.debug(f"Hardcoded lane_id to empty string for shipment")
        
        # Apply field constraints and data type normalization
        payload = self._apply_field_constraints(payload)
        
        # Re-apply hardcoded lane_id to ensure it's always empty string (after field constraints)
        if payload.get("data") and isinstance(payload["data"], list):
            for data_item in payload["data"]:
                if isinstance(data_item, dict) and "shipments" in data_item:
                    shipments = data_item.get("shipments", [])
                    for shipment in shipments:
                        if isinstance(shipment, dict):
                            # Ensure custom object exists
                            if "custom" not in shipment:
                                shipment["custom"] = {}
                            # Hardcode lane_id to empty string (always)
                            shipment["custom"]["lane_id"] = ""
                            
                            # Fix shipment_weight_uom: Convert "KGS" to "KG" (LLM sometimes incorrectly extracts "KGS")
                            if "shipment_weight_uom" in shipment:
                                weight_uom = shipment["shipment_weight_uom"]
                                if isinstance(weight_uom, str) and weight_uom.upper() == "KGS":
                                    shipment["shipment_weight_uom"] = "KG"
                                    logger.info(f"Fixed shipment_weight_uom: 'KGS' -> 'KG'")
                            
                            # Fix actual_weight_uom in custom section: Convert "KGS" to "KG"
                            if "custom" in shipment and isinstance(shipment["custom"], dict):
                                custom = shipment["custom"]
                                if "actual_weight_uom" in custom:
                                    actual_weight_uom = custom["actual_weight_uom"]
                                    if isinstance(actual_weight_uom, str) and actual_weight_uom.upper() == "KGS":
                                        custom["actual_weight_uom"] = "KG"
                                        logger.info(f"Fixed actual_weight_uom: 'KGS' -> 'KG'")
                
                # Hardcode taxes and custom_charges to always be empty lists
                if isinstance(data_item, dict):
                    data_item["taxes"] = []
                    data_item["custom_charges"] = []
                    logger.debug("Hardcoded taxes and custom_charges to empty lists")
        
        # Consolidate duplicate charges (by charge_name) before saving payload
        payload = consolidate_duplicate_charges(payload)
        
        # Save payload to S3 for local testing/debugging
        try:
            main_pdf_name = os.path.basename(input_key)
            clean_main_name = os.path.splitext(main_pdf_name)[0]
            clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
            payload_path = f"Output/{clean_main_name}/final_json/api_payload.json"
            
            save_json_to_s3(payload, output_bucket or input_bucket, payload_path)
            logger.info(f"💾 Saved API payload to S3: s3://{output_bucket or input_bucket}/{payload_path}")
        except Exception as e:
            logger.warning(f"Could not save API payload to S3: {e}")
        
        logger.info(f"Final comprehensive payload structure created")
        logger.info(f"=== COMPLETED create_invoice_payload ===")
        
        return payload

    def sending_json_to_external_api(self, payload):
        """Send the payload to the external API endpoint."""
        import requests
        
        url = API_ENDPOINT
        headers = {
            "Content-Type": "application/json",
            "internal-token": INTERNAL_TOKEN,
            "authorization": AUTHORIZATION_TOKEN
        }
        try:
            logger.info("=== SENDING PAYLOAD TO EXTERNAL API ===")
            logger.info(f"API URL: {url}")
            logger.info(f"Headers: {headers}")
            
            # Log the complete payload structure
            logger.info(f"Complete payload being sent:")
            logger.info(f"Payload type: {type(payload)}")
            logger.info(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'Not a dict'}")
            
            # Log the data array content
            if isinstance(payload, dict) and "data" in payload and isinstance(payload["data"], list):
                for i, data_item in enumerate(payload["data"]):
                    logger.info(f"Data item {i} keys: {list(data_item.keys()) if isinstance(data_item, dict) else 'Not a dict'}")
                    
                    # Specifically log the date fields
                    if isinstance(data_item, dict):
                        logger.info(f"Data item {i} invoiceDate: {data_item.get('invoiceDate')}")
                        logger.info(f"Data item {i} payment_due_date: {data_item.get('payment_due_date')}")
                        
                        # Log the first few charges if they exist
                        if 'charges' in data_item and isinstance(data_item['charges'], list) and data_item['charges']:
                            logger.info(f"Data item {i} first charge: {data_item['charges'][0]}")
            
            # Convert payload to JSON string for logging
            payload_json = json.dumps(payload, indent=2, ensure_ascii=False)
            logger.info(f"Payload JSON (first 2000 chars): {payload_json[:2000]}")
            if len(payload_json) > 2000:
                logger.info(f"Payload JSON (remaining chars): {payload_json[2000:]}")
            
            # Encode JSON string to UTF-8 bytes for proper Content-Length calculation
            payload_bytes = payload_json.encode('utf-8')
            logger.info(f"Payload size: {len(payload_bytes)} bytes")
            
            # Log request start time
            request_start_time = time.time()
            logger.info(f"Starting API request at: {datetime.now().isoformat()}")
            
            # Use data=payload_bytes to ensure correct Content-Length header
            response = requests.post(url, headers=headers, data=payload_bytes)
            
            # Log request end time and duration
            request_end_time = time.time()
            request_duration = request_end_time - request_start_time
            logger.info(f"API request completed in {request_duration:.2f} seconds")
            
            logger.info(f"External API response code: {response.status_code}")
            logger.info(f"External API response headers: {dict(response.headers)}")
            logger.info(f"External API response text: {response.text[:500] if response.text else 'No response text'}")
            
            # Try to parse response as JSON if possible
            try:
                if response.text:
                    response_json = response.json()
                    logger.info("API response parsed as JSON successfully")
            except json.JSONDecodeError:
                logger.warning("API response is not valid JSON")
                
            logger.info("=== COMPLETED SENDING TO EXTERNAL API ===")
            return response
        except Exception as api_err:
            logger.error("Failed to send payload to external API", exc_info=True)
            logger.error(f"API error details: {str(api_err)}")
            response = None
        return response


def extract_extracted_fields_from_jj_result(jj_result: dict) -> List[Dict[str, Any]]:
    """
    Extract extracted_fields from J_J.py result structure.
    This gets the fields with confidence, explanation, and value from the original JSON data.
    
    Args:
        jj_result: The result dictionary returned by J_J.py processing functions
        
    Returns:
        List of dictionaries with field_name, value, confidence, and explanation
    """
    extracted_fields = []
    
    if not jj_result.get('success'):
        logger.error(f"J_J.py processing failed: {jj_result.get('error')}")
        return extracted_fields
    
    # Extract individual results to get the JSON data directly
    individual_results = jj_result.get('individual_results', {})
    logger.info(f"Processing {len(individual_results)} individual results for extracted_fields")
    
    for cluster_id, cluster_result in individual_results.items():
        logger.info(f"Processing cluster {cluster_id}: success={cluster_result.get('success')}, has_extracted_info={'extracted_info' in cluster_result}")
        if cluster_result.get('success') and 'extracted_info' in cluster_result:
            doc_type = cluster_result.get('doc_type', 'unknown')
            extracted_info = cluster_result.get('extracted_info')
            logger.info(f"Cluster {cluster_id} doc_type: '{doc_type}' (lower: '{doc_type.lower()}')")
            
            # Only process freight_invoice for extracted_fields
            # Normalize the doc_type by replacing spaces with underscores and checking variations
            normalized_doc_type = doc_type.lower().replace(' ', '_').replace('-', '_')
            if normalized_doc_type in ['freight_invoice', 'freight_invoices'] or 'freight' in doc_type.lower() and 'invoice' in doc_type.lower():
                logger.info(f"Processing extracted_fields from cluster {cluster_id} ({doc_type})")
                logger.info(f"Extracted_info type: {type(extracted_info)}")
                logger.info(f"Extracted_info keys: {list(extracted_info.keys()) if isinstance(extracted_info, dict) else 'Not a dict'}")
                
                # Check if extracted_info has the expected structure with data
                if isinstance(extracted_info, dict) and 'data' in extracted_info:
                    data_list = extracted_info.get('data', [])
                    
                    logger.info(f"Found data with {len(data_list)} data items")
                    
                    # Process each data item in the list
                    for data_item in data_list:
                        if isinstance(data_item, dict):
                            logger.info(f"Processing data item with keys: {list(data_item.keys())}")
                            # Recursively extract all fields with value, confidence, explanation structure
                            def extract_all_fields(obj, path=""):
                                if isinstance(obj, dict):
                                    # Check if this is a field with value, confidence, explanation
                                    if all(key in obj for key in ['value', 'confidence', 'explanation']):
                                        field_name = path.split('.')[-1] if '.' in path else path
                                        if field_name:  # Only add if we have a field name
                                            extracted_fields.append({
                                                'field_name': field_name,
                                                'value': str(obj.get('value', '')),
                                                'confidence': obj.get('confidence', 0),
                                                'explanation': str(obj.get('explanation', ''))
                                            })
                                            logger.info(f"Extracted field: {field_name} = {obj.get('value')} (confidence: {obj.get('confidence')})")
                                    else:
                                        # Recursively process nested objects
                                        for key, value in obj.items():
                                            new_path = f"{path}.{key}" if path else key
                                            extract_all_fields(value, new_path)
                                elif isinstance(obj, list):
                                    # Process list items
                                    for i, item in enumerate(obj):
                                        new_path = f"{path}[{i}]"
                                        extract_all_fields(item, new_path)
                            
                            # Start extraction from the data item
                            extract_all_fields(data_item)
                else:
                    logger.warning(f"extracted_info doesn't have 'data' key. Available keys: {list(extracted_info.keys()) if isinstance(extracted_info, dict) else 'Not a dict'}")
                    # Fallback to recursive extraction for other structures
                    logger.info(f"Using fallback recursive extraction for cluster {cluster_id}")
                    
                    # Recursively extract fields with confidence, explanation, and value
                    def extract_fields_recursive(obj, path=""):
                        if isinstance(obj, dict):
                            # Check if this is a field with confidence, explanation, and value
                            if all(key in obj for key in ['confidence', 'explanation', 'value']):
                                field_name = path.split('.')[-1] if '.' in path else path
                                if field_name:  # Only add if we have a field name
                                    extracted_fields.append({
                                        'field_name': field_name,
                                        'value': str(obj.get('value', '')),
                                        'confidence': obj.get('confidence', 0),
                                        'explanation': str(obj.get('explanation', ''))
                                    })
                                    logger.info(f"Extracted field: {field_name} = {obj.get('value')} (confidence: {obj.get('confidence')})")
                            else:
                                # Recursively process nested objects
                                for key, value in obj.items():
                                    new_path = f"{path}.{key}" if path else key
                                    extract_fields_recursive(value, new_path)
                        elif isinstance(obj, list):
                            # Process list items
                            for i, item in enumerate(obj):
                                new_path = f"{path}[{i}]"
                                extract_fields_recursive(item, new_path)
                    
                    # Start extraction from the extracted_info
                    extract_fields_recursive(extracted_info)
                break  # Only process the first freight_invoice cluster
            else:
                logger.info(f"Skipping cluster {cluster_id} - doc_type '{doc_type}' doesn't match freight_invoice")
        else:
            logger.info(f"Skipping cluster {cluster_id} - not successful or no extracted_info")
    
    logger.info(f"Successfully extracted {len(extracted_fields)} fields from J_J.py results")
    return extracted_fields

def extract_json_data_from_jj_result(jj_result: dict) -> Dict[str, Any]:
    """
    Extract JSON data directly from J_J.py result structure.
    This gets the exact same JSONs that J_J.py processes and saves.
    
    Args:
        jj_result: The result dictionary returned by J_J.py processing functions
        
    Returns:
        Dictionary with filename -> JSON data mapping
    """
    json_data = {}
    
    if not jj_result.get('success'):
        logger.error(f"J_J.py processing failed: {jj_result.get('error')}")
        return json_data
    
    # Extract individual results to get the JSON data directly
    individual_results = jj_result.get('individual_results', {})
    
    # Track which document types we've seen for duplicate handling
    seen_doc_types = {}
    
    for cluster_id, cluster_result in individual_results.items():
        if cluster_result.get('success') and 'extracted_info' in cluster_result:
            doc_type = cluster_result.get('doc_type', 'unknown')
            extracted_info = cluster_result.get('extracted_info')
            
            # Create filename based on document type
            filename_map = {
                'freight_invoice': 'freight.json',
                'freight_invoices': 'freight.json',
                'air_waybill': 'airwaybill.json',
                'air_waybills': 'airwaybill.json',
                'airway_bill': 'airwaybill.json',
                'airway_bills': 'airwaybill.json',
                'bill_of_lading': 'airwaybill.json',  # Map bill_of_lading to airwaybill.json (same schema)
                'bill_of_ladings': 'airwaybill.json',
                'bol': 'airwaybill.json',
                'commercial_invoice': 'commercial.json',
                'commercial_invoices': 'commercial.json',
                'packing_list': 'packing.json',
                'packing_lists': 'packing.json',
                'delivery_note': 'delivery.json',
                'delivery_notes': 'delivery.json'
            }
            
            # Normalize document type
            doc_type_normalized = doc_type.lower().replace(' ', '_').replace('-', '_')
            filename = filename_map.get(doc_type_normalized)
            
            if not filename:
                # Fallback: create filename from document type
                filename = f"{doc_type_normalized}.json"
            
            # Handle duplicates: If both Air Waybill and Bill of Lading exist, prefer Air Waybill
            if filename == 'airwaybill.json':
                if filename in json_data:
                    # Check what we already have stored (from previous iteration)
                    existing_doc_type = seen_doc_types.get(filename, 'unknown')
                    
                    # If current is Bill of Lading and we already have something, check what it is
                    if doc_type_normalized in ['bill_of_lading', 'bill_of_ladings', 'bol']:
                        # This is a Bill of Lading
                        if existing_doc_type in ['bill_of_lading', 'bill_of_ladings', 'bol']:
                            # We already have a Bill of Lading - keep the first one
                            logger.info(f"Skipping {doc_type} (cluster {cluster_id}) - Bill of Lading already exists")
                            continue
                        else:
                            # We already have an Air Waybill - skip this Bill of Lading
                            logger.info(f"Skipping {doc_type} (cluster {cluster_id}) - Air Waybill already exists, preferring Air Waybill over Bill of Lading")
                            continue
                    else:
                        # This is an Air Waybill - always prefer it over Bill of Lading
                        if existing_doc_type in ['bill_of_lading', 'bill_of_ladings', 'bol']:
                            # We have a Bill of Lading, but now we have an Air Waybill - replace it
                            logger.info(f"Replacing Bill of Lading with Air Waybill (cluster {cluster_id}) - preferring Air Waybill over Bill of Lading")
                        # else: we already have an Air Waybill, this will overwrite it (which is fine)
            
            # Store the extracted info directly (this is the same data J_J.py saves to S3)
            json_data[filename] = extracted_info
            
            # Track that we've stored this document type (only update when we actually store)
            if filename == 'airwaybill.json':
                seen_doc_types[filename] = doc_type_normalized
            
            logger.info(f"Extracted {filename} from cluster {cluster_id} ({doc_type})")
            
            # Print the actual JSON structure we're getting from J_J.py
            logger.info(f"=== JSON STRUCTURE FOR {filename} ===")
            final_data = json_data[filename]
            logger.info(f"JSON keys: {list(final_data.keys()) if isinstance(final_data, dict) else 'Not a dict'}")
            if isinstance(final_data, dict):
                for key, value in final_data.items():
                    if isinstance(value, list) and len(value) > 0:
                        logger.info(f"  {key}: list with {len(value)} items")
                        if isinstance(value[0], dict):
                            logger.info(f"    First item keys: {list(value[0].keys())}")
                            # Show a sample of the nested structure
                            sample_key = list(value[0].keys())[0] if value[0] else "empty"
                            if sample_key != "empty" and isinstance(value[0][sample_key], dict):
                                logger.info(f"    Sample nested structure for '{sample_key}': {list(value[0][sample_key].keys())}")
                    elif isinstance(value, dict):
                        logger.info(f"  {key}: dict with keys {list(value.keys())}")
                    else:
                        logger.info(f"  {key}: {type(value).__name__}")
            logger.info(f"=== END JSON STRUCTURE FOR {filename} ===")
    
    logger.info(f"Successfully extracted {len(json_data)} JSON files from J_J.py results")
    return json_data

def process_json_data_directly(data: dict, filename: str, output_bucket: str, input_file_path: str) -> str:
    """
    Process JSON data directly (from J_J.py results) and print flattened version in logs.
    Does NOT save to S3 - only logs the flattened data for debugging.
    """
    try:
        logger.info(f"Processing JSON data directly for {filename}")
        
        # Flatten all sections and remove all metadata
        # The data from J_J.py might still have nested structure, so we need to flatten it properly
        logger.info(f"=== FLATTENING {filename} ===")
        logger.info(f"Input data keys: {list(data.keys()) if isinstance(data, dict) else 'Not a dict'}")
        flattened_data = flatten_structured_output(data)
        logger.info(f"Flattened data keys: {list(flattened_data.keys()) if isinstance(flattened_data, dict) else 'Not a dict'}")
        
        # Print flattened data in logs (do NOT save to S3)
        logger.info(f"=== FLATTENED JSON DATA FOR {filename} ===")
        logger.info(json.dumps(flattened_data, indent=2, default=str))
        logger.info(f"=== END FLATTENED JSON DATA FOR {filename} ===")
        
        logger.info(f"Successfully processed and logged flattened JSON for {filename} (not saved to S3)")
        return f"flattened_data_logged_for_{filename}"  # Return a placeholder since we're not saving to S3
        
    except Exception as e:
        logger.error(f"Error processing JSON data for {filename}: {e}")
        raise

def process_final_jsons(json_data: Dict[str, Any], excel_file_path: str, output_bucket: str, input_file_path: str, excel_bucket: str = "pando-j-and-j-invoice") -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Process final JSONs with Excel mapping for freight.json and save to final_json folder.
    """
    try:
        logger.info("=" * 60)
        logger.info("STEP 4: Process final JSONs with Excel mapping")
        logger.info("=" * 60)
        
        # Load Excel file from S3 (using the templates bucket)
        logger.info(f"Loading Excel file from: s3://{excel_bucket}/{excel_file_path}")
        excel_sheets = load_excel_from_s3(excel_bucket, excel_file_path)
        
        # Load airport mapping Excel file from S3
        airport_excel_path = excel_file_path.replace('value_mapping.xlsx', 'airport_mapping.xlsx')
        logger.info(f"Loading airport mapping Excel file from: s3://{excel_bucket}/{airport_excel_path}")
        try:
            airport_excel_sheets = load_excel_from_s3(excel_bucket, airport_excel_path)
            logger.info(f"Successfully loaded airport mapping Excel with sheets: {list(airport_excel_sheets.keys())}")
        except Exception as e:
            logger.warning(f"Could not load airport mapping Excel file: {e}")
            airport_excel_sheets = {}
        
        # Create the same folder structure as J_J.py but with final_json instead of Output_json
        main_pdf_name = os.path.basename(input_file_path)
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        
        final_results = []
        final_processed_jsons = {}  # Store the final processed JSONs
        
        for filename, data in json_data.items():
            try:
                logger.info(f"📝 Processing {filename} for final output...")
                
                # Special handling for freight.json - apply Excel mapping
                if filename == "freight.json":
                    logger.info(f"🔧 Applying Excel mapping to {filename}")
                    
                    # Extract vendor_reference_id from freight data
                    vendor_reference_id = None
                    if 'data' in data and isinstance(data['data'], list) and len(data['data']) > 0:
                        vendor_ref_raw = data['data'][0].get('vendor_reference_id', '')
                        
                        # Handle both nested and flat formats
                        if isinstance(vendor_ref_raw, dict) and 'value' in vendor_ref_raw:
                            vendor_reference_id = vendor_ref_raw['value']
                        elif isinstance(vendor_ref_raw, str):
                            vendor_reference_id = vendor_ref_raw
                    
                    if vendor_reference_id:
                        logger.info(f"Found vendor_reference_id: {vendor_reference_id}")
                        mapped_data = map_charges_with_excel(data, excel_sheets, vendor_reference_id)
                        
                        # Apply airport and location mapping after charge mapping
                        if airport_excel_sheets:
                            logger.info(f"🔧 Applying airport and location mapping to {filename}")
                            mapped_data = map_airports_and_locations_with_excel(mapped_data, airport_excel_sheets, vendor_reference_id)
                        else:
                            logger.warning(f"Airport mapping Excel not available, skipping airport/location mapping for {filename}")
                        
                        # Apply MAGNO address normalization after location mapping
                        mapped_data = normalize_magno_addresses(mapped_data, vendor_reference_id)
                    else:
                        logger.warning("No vendor_reference_id found in freight data, skipping mapping")
                        mapped_data = data
                else:
                    # For non-freight JSONs, use data as-is
                    logger.info(f"📋 Using {filename} as-is (no mapping needed)")
                    mapped_data = data
                
                # Create output key (store in final_json folder)
                output_key = f"Output/{clean_main_name}/final_json/{filename}"
                
                # Save the final data
                output_location = save_json_to_s3(mapped_data, output_bucket, output_key)
                
                # Store the final processed JSON data
                final_processed_jsons[filename] = mapped_data
                
                logger.info(f"✅ Successfully processed {filename} to final output: {output_location}")
                final_results.append({
                    'filename': filename,
                    'output_location': output_location,
                    'status': 'success',
                    'mapping_applied': filename == "freight.json"
                })
                
            except Exception as e:
                logger.error(f"❌ Error processing {filename}: {e}")
                final_results.append({
                    'filename': filename,
                    'error': str(e),
                    'status': 'failed',
                    'mapping_applied': False
                })
        
        return final_results, final_processed_jsons
        
    except Exception as e:
        logger.error(f"Error in final JSON processing: {e}")
        raise

def process_event(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process an event (either S3 event or custom event format) and return the result.
    This function can be called when importing lambda_function.py as a module.
    
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
            from urllib.parse import unquote_plus
            input_key = unquote_plus(s3_info["object"]["key"])
            logger.info(f"Processing file: {input_key} from bucket: {input_bucket}")
            
            # Create event structure for lambda_handler
            processed_event = {
                "input_key": input_key,
                "input_bucket": input_bucket,
                "carrier_name": None,
                "use_textract": True,
                "use_s3_clustering": True,
                "s3_bucket": input_bucket,
                "output_bucket": input_bucket,
                "excel_file_path": "templates/Excel_mapping/value_mapping.xlsx",
                "email_details": {},
                "job_id": f"JOB-{int(time.time())}",
                "attachment_id": f"ATT-{int(time.time())}"
            }
            
        else:
            # Handle custom event format with provided parameters
            logger.info("Processing custom event format with provided parameters")
            try:
                input_bucket = event['input_bucket']
                input_key = event['input_key']
                output_bucket = event.get('output_bucket', input_bucket)
                output_prefix = event.get('output_prefix', 'invoice/output/')
                email_details = event.get('email_details', {})
                job_id = event.get('job_id', 'unknown')
                attachment_id = event.get('attachment_id', 'unknown')
                
                logger.info(f"Using provided parameters: bucket={input_bucket}, key={input_key}")
                logger.info(f"Output bucket: {output_bucket}, output prefix: {output_prefix}")
                logger.info(f"Email details: to={email_details.get('to')}, subject={email_details.get('subject')}")
                logger.info(f"Job ID: {job_id}, Attachment ID: {attachment_id}")
                
                # Create event structure for lambda_handler
                processed_event = {
                    "input_key": input_key,
                    "input_bucket": input_bucket,
                    "carrier_name": None,
                    "use_textract": True,
                    "use_s3_clustering": True,
                    "s3_bucket": input_bucket,
                    "output_bucket": output_bucket,
                    "excel_file_path": "templates/Excel_mapping/value_mapping.xlsx",
                    "email_details": email_details,
                    "job_id": job_id,
                    "attachment_id": attachment_id
                }
                
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
                
                # Create event structure for lambda_handler
                processed_event = {
                    "input_key": input_key,
                    "input_bucket": input_bucket,
                    "carrier_name": None,
                    "use_textract": True,
                    "use_s3_clustering": True,
                    "s3_bucket": input_bucket,
                    "output_bucket": output_bucket,
                    "excel_file_path": "templates/Excel_mapping/value_mapping.xlsx",
                    "email_details": email_details,
                    "job_id": email_id,
                    "attachment_id": attachment_id
                }
        
        # Call the lambda_handler with the processed event
        result = lambda_handler(processed_event, None)
        return result
        
    except KeyError as e:
        # Handle missing keys in event structure
        logger.error(f"KeyError in event structure: {str(e)}")
        return {
            'statusCode': 400,
            'body': json.dumps({'error': f"Invalid event structure: {str(e)}"})
        }
    except Exception as e:
        logger.error(f"Error in event processing: {str(e)}", exc_info=True)  # Include stack trace
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }

def process_single_invoice(invoice_event: Dict[str, Any], context=None) -> Dict[str, Any]:
    """
    Process a single invoice event. This is the core processing logic extracted from lambda_handler.
    
    Args:
        invoice_event: Event dictionary with invoice processing parameters
        context: Lambda context (optional)
        
    Returns:
        Dictionary with processing results
    """
    start_time = time.time()
    logger.info(f"=== PROCESSING SINGLE INVOICE ===")
    
    try:
        logger.info(f"Processing invoice event: {json.dumps(invoice_event)}")
        
        # Extract parameters from event
        input_file = invoice_event.get('input_key')
        input_bucket = invoice_event.get('input_bucket')
        carrier_name = invoice_event.get('carrier_name')
        use_textract = invoice_event.get('use_textract', True)
        use_s3_clustering = invoice_event.get('use_s3_clustering', True)
        s3_bucket = invoice_event.get('s3_bucket', 'pando-j-and-j-output')
        output_bucket = invoice_event.get('output_bucket', 'pando-j-and-j-output')
        excel_file_path = invoice_event.get('excel_file_path', 'templates/Excel_mapping/value_mapping.xlsx')
        email_details = invoice_event.get('email_details', {})
        email_id = invoice_event.get('job_id', f'JOB-{int(time.time())}')
        attachment_id = invoice_event.get('attachment_id', f'ATT-{int(time.time())}')
        email_attachments = invoice_event.get('email_attachments')  # S3 path to email attachments
        
        # Log email_attachments if provided
        if email_attachments:
            logger.info(f"📎 Email attachments S3 path provided: {email_attachments}")
        else:
            logger.info("📎 No email_attachments provided, will use clustered PDFs as fallback")
        
        if not input_file:
            logger.error("No input_key provided in event")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'input_key is required'})
            }
        
        if not input_bucket:
            logger.error("No input_bucket provided in event")
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'input_bucket is required'})
            }
        
        # Initialize API handler
        api_handler = APIHandler()
        logger.info("API handler initialized")
        
        # Update DynamoDB status to PROCESSING
        logger.info("Updating DynamoDB status to PROCESSING")
        
        # Extract filename and determine file type
        input_filename = os.path.basename(input_file) if input_file else "unknown"
        file_extension = os.path.splitext(input_filename)[1].lower()
        file_type = "PDF" if file_extension == '.pdf' else file_extension.upper() if file_extension else "PDF"
        
        # Create S3 path
        s3_path = f"s3://{input_bucket}/{input_file}" if input_file else ""
        
        update_attachment_status(
            dynamodb_client,
            DYNAMODB_TABLE,
            email_id,
            attachment_id,
            'PROCESSING',
            is_create=True,
            filename=input_filename,
            file_type=file_type,
            processing_type="Invoice",
            s3_path=s3_path,
            type="Attachment",
            to_email=email_details.get('to') if email_details else None,
            invoice_number="",  # Will be updated later when freight_data is available
            invoice_date="",  # Will be updated later when freight_data is available
            extracted_fields=[]  # Will be updated later when J_J.py results are available
        )
        
        logger.info("=" * 60)
        logger.info("STEP 1: Process invoice using J_J.py")
        logger.info("=" * 60)
        logger.info(f"Processing invoice: {input_file}")
        logger.info(f"Carrier: {carrier_name or 'Auto-detect'}")
        logger.info(f"Use Textract: {use_textract}")
        logger.info(f"Use S3 clustering: {use_s3_clustering}")
        
        # Step 1: Process the invoice using J_J.py
        if use_s3_clustering:
            logger.info("Using S3-based clustering processing")
            # Download S3 file to temporary location for processing
            import tempfile
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                temp_file_path = temp_file.name
                logger.info(f"Downloading S3 file to temporary location: {temp_file_path}")
                
                # Download file from S3
                s3.download_file(input_bucket, input_file, temp_file_path)
                logger.info(f"Successfully downloaded S3 file: s3://{input_bucket}/{input_file}")
                
                # Set OUTPUT_DIR to Lambda-compatible path before processing
                import J_J
                original_output_dir = J_J.OUTPUT_DIR
                J_J.OUTPUT_DIR = '/tmp/jj_output'
                logger.info(f"Set OUTPUT_DIR to Lambda-compatible path: {J_J.OUTPUT_DIR}")
                
                # Override J_J.py constants to use correct output bucket and folder structure
                original_output_bucket = J_J.OUTPUT_BUCKET
                J_J.OUTPUT_BUCKET = output_bucket  # Use the output bucket from event
                logger.info(f"Set OUTPUT_BUCKET to: {J_J.OUTPUT_BUCKET}")
                
                try:
                    # Process the temporary file, but pass the original S3 key name for proper folder structure
                    original_filename = os.path.basename(input_file)  # Get original filename from S3 key
                    sender_email = email_details.get('to') if email_details else None  # 'to' field contains the sender email
                    jj_result = process_invoice_with_s3_clustering(temp_file_path, carrier_name, use_textract, output_bucket, original_filename, sender_email)
                finally:
                    # Restore original values
                    J_J.OUTPUT_DIR = original_output_dir
                    J_J.OUTPUT_BUCKET = original_output_bucket
                    logger.info(f"Restored OUTPUT_DIR to: {J_J.OUTPUT_DIR}")
                    logger.info(f"Restored OUTPUT_BUCKET to: {J_J.OUTPUT_BUCKET}")
                
                # Clean up temporary file
                try:
                    os.unlink(temp_file_path)
                    logger.info(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Could not clean up temporary file {temp_file_path}: {cleanup_error}")
        else:
            logger.info("Using local processing")
            # Set OUTPUT_DIR to Lambda-compatible path before processing
            import J_J
            original_output_dir = J_J.OUTPUT_DIR
            J_J.OUTPUT_DIR = '/tmp/jj_output'
            logger.info(f"Set OUTPUT_DIR to Lambda-compatible path: {J_J.OUTPUT_DIR}")
            
            # Override J_J.py constants to use correct output bucket and folder structure
            original_output_bucket = J_J.OUTPUT_BUCKET
            J_J.OUTPUT_BUCKET = output_bucket  # Use the output bucket from event
            logger.info(f"Set OUTPUT_BUCKET to: {J_J.OUTPUT_BUCKET}")
            
            try:
                # For local processing, use the input file name as the original filename
                original_filename = os.path.basename(input_file)
                sender_email = email_details.get('to') if email_details else None  # 'to' field contains the sender email
                jj_result = process_invoice_local(input_file, carrier_name, use_textract, original_filename, sender_email)
            finally:
                # Restore original values
                J_J.OUTPUT_DIR = original_output_dir
                J_J.OUTPUT_BUCKET = original_output_bucket
                logger.info(f"Restored OUTPUT_DIR to: {J_J.OUTPUT_DIR}")
                logger.info(f"Restored OUTPUT_BUCKET to: {J_J.OUTPUT_BUCKET}")
        
        if not jj_result.get('success'):
            error_msg = f"J_J.py processing failed: {jj_result.get('error')}"
            error_code = jj_result.get('error_code', 'EXTRACTION_FAILED')
            logger.error(error_msg)
            
            # Check if this is an authorization failure
            if error_code == 'AUTHORIZATION_FAILED':
                logger.warning(f"❌ Authorization failed: {error_msg}")
                logger.warning("🛑 STOPPING PROCESSING - Carrier and sender email do not match")
                sender_email = email_details.get('to') if email_details else None
                identified_carrier = jj_result.get('carrier', 'Unknown')
                
                # Create rejection reason message
                rejection_reason = f"Carrier name '{identified_carrier}' does not match sender email domain. Sender {sender_email} is not authorized to send {identified_carrier} documents."
                
                # Update DynamoDB status to REJECTED with reason
                logger.info("Updating DynamoDB status to REJECTED")
                logger.info(f"Rejection reason: {rejection_reason}")
                update_attachment_status(
                    dynamodb_client,
                    DYNAMODB_TABLE,
                    email_id,
                    attachment_id,
                    'REJECTED',
                    error={
                        'message': error_msg,
                        'error_code': 'AUTHORIZATION_FAILED',
                        'rejection_reason': rejection_reason,
                        'carrier': identified_carrier,
                        'sender_email': sender_email
                    },
                    filename=input_filename,
                    file_type=file_type,
                    processing_type="Invoice",
                    s3_path=s3_path,
                    type="Attachment",
                    to_email=sender_email,
                    invoice_number="",
                    invoice_date="",
                    extracted_fields=[]
                )
                
                # Send rejection email
                if sender_email:
                    try:
                        logger.info(f"Sending authorization rejection email to: {sender_email}")
                        # Get SMTP credentials
                        try:
                            SMTP_CONFIG = get_secret(SMTP_SECRET_NAME)
                            smtp_username = SMTP_CONFIG["SMTP_USERNAME"]
                            smtp_password = SMTP_CONFIG["SMTP_PASSWORD"]
                        except Exception as e:
                            logger.error(f"Failed to retrieve SMTP credentials: {str(e)}")
                            smtp_username = ""
                            smtp_password = ""
                        
                        if smtp_username and smtp_password:
                            send_authorization_rejection_email(
                                SMTP_SERVER, SMTP_PORT,
                                smtp_username, smtp_password,
                                FROM_EMAIL, sender_email,
                                email_details.get('subject', 'Invoice Processing') if email_details else 'Invoice Processing',
                                email_id,
                                email_details.get('message_id') if email_details else None,
                                original_sender=sender_email,
                                original_date=email_details.get('status_email_date') if email_details else None,
                                original_subject=email_details.get('subject', 'Invoice Processing') if email_details else 'Invoice Processing',
                                original_body=email_details.get('original_body', '') if email_details else '',
                                carrier=identified_carrier
                            )
                            logger.info(f"✅ Authorization rejection email sent to: {sender_email}")
                        else:
                            logger.warning("SMTP credentials not available, skipping rejection email")
                    except Exception as e:
                        logger.error(f"Error sending authorization rejection email: {str(e)}")
                
                logger.warning("🛑 Processing stopped - returning early due to authorization failure")
                return {
                    'statusCode': 403,
                    'body': json.dumps({
                        'error': error_msg,
                        'error_code': 'AUTHORIZATION_FAILED',
                        'carrier': identified_carrier,
                        'sender_email': sender_email,
                        'message': 'Processing stopped - carrier and sender email do not match'
                    })
                }
            
            # For other errors, handle as before
            rejection_reason = f"J_J.py processing failed: {error_msg}"
            update_attachment_status(
                dynamodb_client,
                DYNAMODB_TABLE,
                email_id,
                attachment_id,
                'FAILED',
                error={
                    'message': error_msg,
                    'error_code': error_code,
                    'rejection_reason': rejection_reason
                },
                extraction_failed=1,
                filename=input_filename,
                file_type=file_type,
                processing_type="Invoice",
                s3_path=s3_path,
                type="Attachment",
                to_email=email_details.get('to') if email_details else None,
                invoice_number="",  # Not available at this stage
                invoice_date="",  # Not available at this stage
                extracted_fields=[]  # Not available at this stage
            )
            return {
                'statusCode': 500,
                'body': json.dumps({'error': error_msg})
            }
        
        detected_carrier = jj_result.get('carrier', 'Unknown')
        logger.info("✅ J_J.py processing completed successfully")
        logger.info(f"Carrier: {detected_carrier}")
        logger.info(f"Total clusters: {jj_result.get('total_clusters', 0)}")
        logger.info(f"Successful clusters: {jj_result.get('successful_clusters', 0)}")
        
        # Step 1.5: Send RECEIVED status email after successful carrier identification and authorization
        # This is sent here because authorization check passed in J_J.py
        sender_email = email_details.get('to') if email_details else None
        if sender_email:
            try:
                logger.info("=" * 60)
                logger.info("STEP 1.5: Send RECEIVED status email (authorization passed)")
                logger.info("=" * 60)
                
                # Get attachments list from email_details or use empty list
                attachments_list = []
                if email_details.get('filename'):
                    attachments_list = [{'filename': email_details.get('filename', 'Unknown')}]
                
                send_received_status_email(
                    SMTP_SERVER, SMTP_PORT,
                    SMTP_USERNAME, SMTP_PASSWORD,
                    FROM_EMAIL, sender_email,
                    email_details.get('subject', 'Invoice Processing') if email_details else 'Invoice Processing',
                    attachments_list,
                    email_id,
                    email_details.get('message_id') if email_details else None,
                    original_sender=sender_email,
                    original_date=email_details.get('status_email_date') if email_details else None,
                    original_subject=email_details.get('subject', 'Invoice Processing') if email_details else 'Invoice Processing',
                    original_body=email_details.get('original_body', '') if email_details else ''
                )
                logger.info(f"✅ RECEIVED status email sent to: {sender_email}")
            except Exception as e:
                logger.error(f"Error sending RECEIVED status email: {str(e)}")
                # Continue processing even if email fails
        
        logger.info("=" * 60)
        logger.info("STEP 2: Extract JSON data from J_J.py results")
        logger.info("=" * 60)
        
        # Step 2: Extract JSON data directly from J_J.py results
        json_data = extract_json_data_from_jj_result(jj_result)
        
        # Step 2.5: Extract extracted_fields from J_J.py results (before flattening)
        extracted_fields_from_jj = extract_extracted_fields_from_jj_result(jj_result)
        logger.info(f"Extracted {len(extracted_fields_from_jj)} fields with confidence and explanation from J_J.py results")
        
        # Debug: Log the extracted fields structure
        if extracted_fields_from_jj:
            logger.info("Extracted fields structure:")
            for i, field in enumerate(extracted_fields_from_jj):
                logger.info(f"  Field {i}: {field}")
        else:
            logger.warning("No extracted fields found - this may indicate a data structure mismatch")
        
        if not json_data:
            error_msg = "No JSON data extracted from J_J.py results"
            logger.error(error_msg)
            rejection_reason = "No JSON data extracted from J_J.py results - extraction failed"
            update_attachment_status(
                dynamodb_client,
                DYNAMODB_TABLE,
                email_id,
                attachment_id,
                'FAILED',
                error={
                    'message': error_msg,
                    'error_code': 'EXTRACTION_FAILED',
                    'rejection_reason': rejection_reason
                },
                extraction_failed=1,
                filename=input_filename,
                file_type=file_type,
                processing_type="Invoice",
                s3_path=s3_path,
                type="Attachment",
                to_email=email_details.get('to') if email_details else None,
                invoice_number="",  # Not available at this stage
                invoice_date="",  # Not available at this stage
                extracted_fields=[]  # Not available at this stage
            )
            return {
                'statusCode': 500,
                'body': json.dumps({'error': error_msg})
            }
        
        logger.info(f"Successfully extracted {len(json_data)} JSON files:")
        for filename in json_data.keys():
            logger.info(f"  ✓ {filename}")
        
        logger.info("=" * 60)
        logger.info("STEP 3: Flatten JSONs and store in S3")
        logger.info("=" * 60)
        
        # Step 3: Flatten all JSON files and store in flatten_json folder
        flatten_results = []
        flattened_data = {}  # Store flattened data in memory
        for filename, data in json_data.items():
            try:
                logger.info(f"📝 Flattening {filename}...")
                # Flatten the data first
                flattened_json = flatten_structured_output(data)
                # Save to S3
                flatten_location = process_json_data_directly(data, filename, output_bucket, input_file)
                # Store flattened data for next step
                flattened_data[filename] = flattened_json
                logger.info(f"✅ Successfully flattened and logged {filename} (not saved to S3)")
                flatten_results.append({
                    'input_filename': filename,
                    'flatten_location': 'logged_in_cloudwatch',  # Not actually saved to S3
                    'status': 'success'
                })
            except Exception as e:
                logger.error(f"❌ Error flattening {filename}: {e}")
                flatten_results.append({
                    'input_filename': filename,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Count successful and failed processing
        successful = len([r for r in flatten_results if r['status'] == 'success'])
        failed = len([r for r in flatten_results if r['status'] == 'failed'])
        
        logger.info("=" * 60)
        logger.info("STEP 4: Process final JSONs with Excel mapping")
        logger.info("=" * 60)
        
        # Step 4: Process final JSONs with Excel mapping (using flattened data from memory)
        final_results, final_processed_jsons = process_final_jsons(flattened_data, excel_file_path, output_bucket, input_file, "pando-j-and-j-invoice")
        
        # Count successful and failed final processing
        final_successful = len([r for r in final_results if r['status'] == 'success'])
        final_failed = len([r for r in final_results if r['status'] == 'failed'])
        
        # Get the freight.json data for API processing (final processed with mapping and business rules)
        freight_data = None
        if 'freight.json' in final_processed_jsons:
            freight_data = final_processed_jsons['freight.json']
            logger.info("Found final processed freight.json for API processing")
        
        # Calculate confidence score using ORIGINAL JSON data (not flattened)
        try:
            # Use original JSON data from J_J.py results for confidence calculation
            original_freight_data = None
            if 'freight.json' in json_data:
                original_freight_data = json_data['freight.json']
                logger.info("Using original freight.json data for confidence calculation")
            
            average_confidence = calculate_weighted_confidence(original_freight_data) if original_freight_data else 0.0
            logger.info(f"Average confidence score: {average_confidence:.4f}")
        except Exception as conf_err:
            logger.warning(f"Could not calculate average confidence: {conf_err}")
            average_confidence = 0.0
        
        # Step 5: Send email notification if email details provided
        if email_details and email_details.get('to'):
            logger.info("=" * 60)
            logger.info("STEP 5: Send email notification")
            logger.info("=" * 60)
            
            # EMAIL SENDING COMMENTED OUT - NOW HANDLED BY SEPARATE EMAIL LAMBDA FUNCTION
            # try:
            #     # Format data as HTML
            #     html_content = format_invoice_data_as_html(freight_data) if freight_data else "No freight data available"
            #     
            #     # Send email
            #     send_email(
            #         smtp_server=SMTP_SERVER,
            #         smtp_port=SMTP_PORT,
            #         smtp_username=SMTP_USERNAME,
            #         smtp_password=SMTP_PASSWORD,
            #         from_email=FROM_EMAIL,
            #         to_email=email_details['to'],
            #         subject=email_details.get('subject', 'Invoice Processing Complete'),
            #         body=html_content,
            #         json_data=freight_data or {},
            #         message_id=email_details.get('message_id'),
            #         quoted_sender=email_details.get('quoted_sender'),
            #         quoted_date=email_details.get('quoted_date'),
            #         quoted_subject=email_details.get('quoted_subject'),
            #         quoted_body=email_details.get('quoted_body')
            #     )
            #     logger.info("✅ Email notification sent successfully")
            # except Exception as e:
            #     logger.error(f"❌ Error sending email: {e}")
            
            logger.info("📧 Email sending disabled - handled by separate email Lambda function")
        
        # Step 6: Send data to external API if freight data available
        api_success = False
        api_response_data = None
        
        if freight_data:
            logger.info("=" * 60)
            logger.info("STEP 6: Send data to external API")
            logger.info("=" * 60)
            
            try:
                # Create comprehensive API payload
                final_payload = api_handler.create_invoice_payload(
                    freight_data,
                    email_details,
                    detected_carrier,
                    input_file,
                    s3_bucket,
                    jj_result=jj_result,
                    output_bucket=output_bucket,
                    all_final_jsons=final_processed_jsons,  # Pass final processed JSONs (after mapping and business rules)
                    email_attachments=email_attachments  # Pass email_attachments S3 path
                )
                
                # Validate payload structure
                validated_payload = validate_api_payload(final_payload)
                
                # Ensure taxes and custom_charges are always empty lists (after validation)
                if validated_payload.get("data") and isinstance(validated_payload["data"], list):
                    for data_item in validated_payload["data"]:
                        if isinstance(data_item, dict):
                            data_item["taxes"] = []
                            data_item["custom_charges"] = []
                            logger.debug("Enforced taxes and custom_charges as empty lists after validation")
                
                # Extract invoice number early for use in email alerts
                final_invoice_number = ""
                if validated_payload and 'data' in validated_payload and validated_payload['data']:
                    payload_data = validated_payload['data'][0]  # Get first data item
                    final_invoice_number = payload_data.get("invoice_number", "")
                
                # Save final payload JSON after API payload validation (before mandatory field validation)
                try:
                    main_pdf_name = os.path.basename(input_file)
                    clean_main_name = os.path.splitext(main_pdf_name)[0]
                    clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
                    payload_path = f"Output/{clean_main_name}/final_json/api_payload.json"
                    
                    save_json_to_s3(validated_payload, output_bucket, payload_path)
                    logger.info(f"💾 Saved final API payload to S3: s3://{output_bucket}/{payload_path}")
                except Exception as e:
                    logger.warning(f"Could not save final API payload to S3: {e}")
                
                # Run both validation checks independently
                # Step 1: Validate external mandatory fields (empty check)
                external_valid, external_field_errors = validate_external_mandatory_fields(validated_payload)
                
                # Step 2: Validate internal field formats (format check) - ALWAYS runs regardless of external check
                internal_valid, internal_field_errors = validate_internal_field_formats(validated_payload)
                
                # Handle external validation errors
                if not external_valid:
                    logger.error(f"❌ External mandatory field validation failed. Missing fields: {len(external_field_errors)}")
                    logger.error(f"External field errors: {external_field_errors}")
                    
                    # Note: Instant alert email removed - errors will be included in daily consolidated email instead
                    logger.info("External field errors will be included in daily consolidated email report")
                    
                    # Skip API call - external validation failed (REJECTED)
                    api_response_data = {
                        'status_code': 0,
                        'success': False,
                        'timestamp': datetime.now().isoformat(),
                        'body': 'External mandatory field validation failed - API call skipped',
                        'external_field_errors': external_field_errors
                    }
                else:
                    logger.info("✅ External mandatory field validation passed")
                    
                    # Handle internal validation errors (but still proceed with API call)
                    if not internal_valid:
                        logger.error(f"❌ Internal field format validation failed. Format errors: {len(internal_field_errors)}")
                        logger.error(f"Internal field errors: {internal_field_errors}")
                        
                        # Send instant email alert for internal field errors to Pando team
                        try:
                            if PANDO_TEAM_EMAILS:
                                send_internal_field_error_alert(
                                    SMTP_SERVER,
                                    SMTP_PORT,
                                    SMTP_USERNAME,
                                    SMTP_PASSWORD,
                                    FROM_EMAIL,
                                    PANDO_TEAM_EMAILS,
                                    input_filename,
                                    internal_field_errors,
                                    final_invoice_number
                                )
                                logger.info("✅ Internal field error alert sent to Pando team")
                            else:
                                logger.warning("No Pando team emails configured for internal field error alert")
                        except Exception as e:
                            logger.error(f"Failed to send internal field error alert: {str(e)}")
                        
                        # NOTE: Even if internal validation fails, we still hit the API
                        logger.info("⚠️ Internal validation failed but proceeding with API call")
                    else:
                        logger.info("✅ Internal field format validation passed")
                    
                    # Proceed with API call (regardless of internal validation result)
                    response = api_handler.sending_json_to_external_api(validated_payload)
                    
                    # Create API response data for DynamoDB
                    api_response_data = {
                        'status_code': response.status_code if response else 0,
                        'success': response.status_code == 200 if response else False,
                        'timestamp': datetime.now().isoformat(),
                        'body': response.text if response and hasattr(response, 'text') else ''
                    }
                    
                    if response and response.status_code == 200:
                        api_success = True
                        logger.info("✅ API request successful")
                    else:
                        logger.error(f"❌ API request failed with status code: {response.status_code if response else 'No response'}")
                
                # Store both error types for final DynamoDB update
                final_external_errors = external_field_errors if 'external_field_errors' in locals() else []
                final_internal_errors = internal_field_errors if 'internal_field_errors' in locals() else []
                    
            except Exception as e:
                logger.error(f"❌ Error sending data to API: {e}")
                api_response_data = {
                    'status_code': 0,
                    'success': False,
                    'timestamp': datetime.now().isoformat(),
                    'body': str(e)
                }
        
        # Update DynamoDB status to COMPLETED
        logger.info("Updating DynamoDB status to COMPLETED...")
        
        # Extract invoice data from final payload if available
        final_invoice_number = ""
        final_invoice_date = ""
        if 'final_payload' in locals() and final_payload and 'data' in final_payload and final_payload['data']:
            payload_data = final_payload['data'][0]  # Get first data item
            final_invoice_number = payload_data.get("invoice_number", "")
            final_invoice_date = payload_data.get("invoice_date", "")
            logger.info(f"Extracted from final payload - invoice_number: {final_invoice_number}, invoice_date: {final_invoice_date}")
        else:
            # Fallback to freight_data if final_payload not available
            final_invoice_number = freight_data.get("invoice_number", "") if freight_data else ""
            final_invoice_date = freight_data.get("invoice_date", "") if freight_data else ""
            logger.info(f"Using freight_data fallback - invoice_number: {final_invoice_number}, invoice_date: {final_invoice_date}")
        
        # Determine final status based on validation and API results
        # If external validation failed, status is REJECTED (no API call made)
        rejection_reason = None
        if 'final_external_errors' in locals() and final_external_errors:
            final_status = 'REJECTED'
            # Create rejection reason with missing field names
            missing_field_names = [err.get('field_name', 'Unknown field') for err in final_external_errors if isinstance(err, dict)]
            if missing_field_names:
                rejection_reason = f"Missing mandatory fields: {', '.join(missing_field_names)}"
            else:
                rejection_reason = f"Missing mandatory fields: {len(final_external_errors)} field(s) missing"
            logger.info(f"❌ Processing rejected due to {len(final_external_errors)} external errors (missing mandatory fields)")
        elif api_success:
            # If API was successful, mark as SUCCESS (regardless of internal validation errors)
            final_status = 'SUCCESS'
            logger.info("✅ Processing completed successfully")
        else:
            # Otherwise mark as FAILED (API errors, processing failures, etc.)
            # Note: Internal validation errors don't change status - API call was still made
            final_status = 'FAILED'
            # Get API error message if available
            if 'api_response_data' in locals() and api_response_data:
                api_error = api_response_data.get('body', '') or api_response_data.get('error', '')
                if api_error:
                    rejection_reason = f"API processing error: {api_error[:200]}"  # Limit length
                else:
                    rejection_reason = "API processing failed"
            else:
                rejection_reason = "Processing failed - API call unsuccessful"
            logger.info("❌ Processing failed")
        
        # Determine missing_critical_field flag based on external errors
        has_external_errors = 'final_external_errors' in locals() and final_external_errors
        missing_critical_field_flag = 1 if has_external_errors else 0
        
        # Prepare error object if status is REJECTED or FAILED
        error_obj = None
        if final_status in ['REJECTED', 'FAILED'] and rejection_reason:
            error_obj = {
                'message': rejection_reason,
                'error_code': 'EXTERNAL_VALIDATION_FAILED' if final_status == 'REJECTED' else 'API_ERROR',
                'rejection_reason': rejection_reason
            }
        
        update_attachment_status(
            dynamodb_client,
            DYNAMODB_TABLE,
            email_id,
            attachment_id,
            final_status,
            output_path=f"s3://{output_bucket}/Output/{os.path.basename(input_file)}/final_json/",
            carrier_name=detected_carrier,
            mode=freight_data.get("mode", "") if freight_data else "",
            invoice_number=final_invoice_number,
            invoice_date=final_invoice_date,
            missing_critical_field=missing_critical_field_flag,
            textract_failed=0,
            classification_failed=0,
            extraction_failed=0,
            format_failed=0,
            missing_fields=[],  # Legacy field - keeping empty for backward compatibility
            external_field_errors=final_external_errors if 'final_external_errors' in locals() else [],
            error=error_obj,
            internal_field_errors=final_internal_errors if 'final_internal_errors' in locals() else [],
            confidence_score=average_confidence,
            extracted_fields=extracted_fields_from_jj,  # Use extracted fields from J_J.py results
            api_response=api_response_data,
            filename=input_filename,
            file_type=file_type,
            processing_type="Invoice",
            s3_path=s3_path,
            type="Attachment",
            to_email=email_details.get('to') if email_details else None
        )
        
        # Calculate execution time
        end_time = time.time()
        execution_time = end_time - start_time
        
        response = {
            'statusCode': 200,
            'body': json.dumps({
                'message': f'Processed {len(json_data)} files: {successful} flattened, {final_successful} final processed',
                'jj_processing': {
                    'success': jj_result.get('success'),
                    'carrier': jj_result.get('carrier'),
                    'total_clusters': jj_result.get('total_clusters'),
                    'successful_clusters': jj_result.get('successful_clusters'),
                    'failed_clusters': jj_result.get('failed_clusters'),
                    'skipped_clusters': jj_result.get('skipped_clusters')
                },
                'flatten_results': flatten_results,
                'final_results': final_results,
                'api_success': api_success,
                'confidence_score': average_confidence,
                'execution_time_seconds': execution_time,
                'summary': {
                    'total_files': len(json_data),
                    'flattened_successful': successful,
                    'flattened_failed': failed,
                    'final_successful': final_successful,
                    'final_failed': final_failed
                }
            })
        }
        
        logger.info("=" * 60)
        logger.info("🎉 PROCESSING COMPLETE!")
        logger.info("=" * 60)
        logger.info(f"✅ J_J.py processing: {jj_result.get('successful_clusters', 0)}/{jj_result.get('total_clusters', 0)} clusters successful")
        logger.info(f"✅ JSON flattening: {successful}/{len(json_data)} files successful")
        logger.info(f"✅ Final processing: {final_successful}/{len(json_data)} files successful")
        logger.info(f"✅ API success: {api_success}")
        logger.info(f"✅ Confidence score: {average_confidence:.4f}")
        logger.info(f"✅ Execution time: {execution_time:.2f} seconds")
        
        # Calculate the output paths for logging
        main_pdf_name = os.path.basename(input_file)
        clean_main_name = os.path.splitext(main_pdf_name)[0]
        clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
        flatten_path = f"Output/{clean_main_name}/flatten_json/"  # Only for logging, not stored
        final_path = f"Output/{clean_main_name}/final_json/"
        logger.info(f"📁 Flattened JSONs logged (not saved to S3): {flatten_path}")
        logger.info(f"📁 Final JSONs saved to: {final_path}")
        logger.info("=" * 60)
        
        return response
        
    except Exception as e:
        end_time = time.time()
        execution_time = end_time - start_time
        logger.error(f"Lambda function failed: {e}")
        logger.error(f"=== LAMBDA HANDLER FAILED after {execution_time:.2f} seconds ===")
        
        # Update DynamoDB status to FAILED
        try:
            if 'email_id' in locals() and 'attachment_id' in locals():
                # Extract filename and file type for error case
                input_filename = os.path.basename(input_file) if 'input_file' in locals() and input_file else "unknown"
                file_extension = os.path.splitext(input_filename)[1].lower()
                file_type = "PDF" if file_extension == '.pdf' else file_extension.upper() if file_extension else "PDF"
                s3_path = f"s3://{input_bucket}/{input_file}" if 'input_bucket' in locals() and 'input_file' in locals() and input_file else ""
                
                rejection_reason = f"Lambda function processing error: {str(e)[:200]}"
                update_attachment_status(
                    dynamodb_client,
                    DYNAMODB_TABLE,
                    email_id,
                    attachment_id,
                    'FAILED',
                    error={
                        'message': str(e),
                        'error_code': 'PROCESSING_ERROR',
                        'rejection_reason': rejection_reason
                    },
                    missing_fields=[],
                    confidence_score=None,
                    extracted_fields=[],  # Not available in error case
                    api_response=None,
                    filename=input_filename,
                    file_type=file_type,
                    processing_type="Invoice",
                    s3_path=s3_path,
                    type="Attachment",
                    to_email=email_details.get('to') if 'email_details' in locals() and email_details else None,
                    invoice_number="",  # Not available in error case
                    invoice_date=""  # Not available in error case
                )
        except Exception as db_err:
            logger.error(f"Failed to update DynamoDB: {str(db_err)}")
            
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'execution_time_seconds': execution_time
            })
        }

def lambda_handler(event, context):
    """
    AWS Lambda handler function for processing invoices with full integration.
    
    Supports two event formats:
    1. SQS Event (for batch processing):
       {
           "Records": [
               {
                   "body": "{\"input_key\": \"...\", \"input_bucket\": \"...\", ...}",
                   "messageId": "...",
                   "receiptHandle": "..."
               }
           ]
       }
    
    2. Direct Invocation (backward compatibility):
       {
           "input_key": "path/to/invoice.pdf",
           "input_bucket": "bucket-name",
           "carrier_name": "KWE",
           "use_textract": true,
           "use_s3_clustering": true,
           "s3_bucket": "pando-j-and-j-output",
           "output_bucket": "pando-j-and-j-output",
           "excel_file_path": "templates/Excel_mapping/value_mapping.xlsx",
           "email_details": {...},
           "job_id": "unique-job-id",
           "attachment_id": "unique-attachment-id"
       }
    """
    handler_start_time = time.time()
    logger.info(f"=== LAMBDA HANDLER STARTED ===")
    
    # Handle context safely for both AWS Lambda and local testing
    if context:
        logger.info(f"Lambda function ARN: {context.invoked_function_arn}")
        logger.info(f"Lambda function version: {context.function_version}")
        logger.info(f"Lambda request ID: {context.aws_request_id}")
        logger.info(f"Lambda memory limit: {context.memory_limit_in_mb}MB")
        logger.info(f"Lambda time remaining: {context.get_remaining_time_in_millis()}ms")
    else:
        logger.info("Running in local mode - context not available")
    
    try:
        logger.info(f"Lambda function started with event: {json.dumps(event)}")
        
        # Check if this is an SQS event
        if 'Records' in event and len(event['Records']) > 0:
            # Check if first record has SQS structure (SQS events have 'body' and 'eventSource' fields)
            first_record = event['Records'][0]
            if 'body' in first_record and (first_record.get('eventSource') == 'aws:sqs' or 'receiptHandle' in first_record):
                # This is an SQS event
                logger.info(f"📬 Detected SQS event with {len(event['Records'])} message(s)")
                
                # Process up to 5 records (SQS batch limit)
                records_to_process = event['Records'][:5]
                logger.info(f"Processing {len(records_to_process)} record(s) from SQS batch")
                
                results = []
                for i, record in enumerate(records_to_process):
                    try:
                        logger.info(f"Processing SQS record {i+1}/{len(records_to_process)}")
                        
                        # Parse the message body (should be JSON string)
                        message_body = record.get('body', '{}')
                        try:
                            invoice_event = json.loads(message_body)
                        except json.JSONDecodeError as e:
                            logger.error(f"Failed to parse SQS message body as JSON: {e}")
                            logger.error(f"Message body: {message_body}")
                            results.append({
                                'statusCode': 400,
                                'body': json.dumps({
                                    'error': f'Invalid JSON in SQS message body: {str(e)}',
                                    'messageId': record.get('messageId', 'unknown')
                                })
                            })
                            continue
                        
                        # Process the invoice
                        result = process_single_invoice(invoice_event, context)
                        results.append(result)
                        
                        logger.info(f"✅ Completed processing SQS record {i+1}/{len(records_to_process)}")
                        
                    except Exception as e:
                        logger.error(f"❌ Error processing SQS record {i+1}: {str(e)}", exc_info=True)
                        results.append({
                            'statusCode': 500,
                            'body': json.dumps({
                                'error': str(e),
                                'messageId': record.get('messageId', 'unknown')
                            })
                        })
                
                # Return aggregated results
                handler_end_time = time.time()
                handler_execution_time = handler_end_time - handler_start_time
                
                successful = len([r for r in results if r.get('statusCode') == 200])
                failed = len([r for r in results if r.get('statusCode') != 200])
                
                logger.info(f"=== SQS BATCH PROCESSING COMPLETE ===")
                logger.info(f"✅ Successful: {successful}/{len(results)}")
                logger.info(f"❌ Failed: {failed}/{len(results)}")
                logger.info(f"⏱️ Total execution time: {handler_execution_time:.2f} seconds")
                
                return {
                    'statusCode': 200,
                    'body': json.dumps({
                        'message': f'Processed {len(results)} invoice(s) from SQS',
                        'successful': successful,
                        'failed': failed,
                        'total': len(results),
                        'execution_time_seconds': handler_execution_time,
                        'results': results
                    })
                }
        
        # Not an SQS event - process as direct invocation (backward compatibility)
        logger.info("📧 Processing as direct invocation (backward compatibility mode)")
        result = process_single_invoice(event, context)
        return result
        
    except Exception as e:
        handler_end_time = time.time()
        handler_execution_time = handler_end_time - handler_start_time
        logger.error(f"Lambda handler failed: {e}", exc_info=True)
        logger.error(f"=== LAMBDA HANDLER FAILED after {handler_execution_time:.2f} seconds ===")
        
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'execution_time_seconds': handler_execution_time
            })
        }

def main():
    """
    Main function for local testing and execution.
    """
    print("=" * 60)
    print("J&J INVOICE PROCESSING LAMBDA FUNCTION")
    print("=" * 60)
    
    # Default configuration - UPDATE THESE VALUES
    DEFAULT_INPUT_FILE = '/Users/aruniga/Downloads/KWEI/Invoice #1.pdf'  # Change this to your PDF file path
    DEFAULT_CARRIER = None  # Will auto-detect if None
    DEFAULT_S3_BUCKET = "pando-j-and-j-output"  # Change this to your S3 bucket
    DEFAULT_EXCEL_FILE = "templates/Excel_mapping/value_mapping.xlsx"  # Change this to your Excel mapping file path
    DEFAULT_EMAIL_TO = "test@example.com"  # Change this to your email for notifications
    
    # Check if processing an event file
    import sys
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
            
            if result['statusCode'] == 200:
                print(f"\n✅ Processing completed successfully!")
            else:
                print(f"\n❌ Processing failed: {result.get('body', 'Unknown error')}")
                
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
    
    # Check if command line arguments are provided
    if len(sys.argv) >= 2:
        input_file = sys.argv[1]
        print(f"Using command line input file: {input_file}")
    else:
        input_file = DEFAULT_INPUT_FILE
        print(f"Using default input file: {input_file}")
        print("Note: You can override with: python3 lambda_function.py <pdf_file_path>")
        print("Or process an event with: python3 lambda_function.py --event <event_json_file>")
    
    if len(sys.argv) >= 3:
        carrier = sys.argv[2]
        print(f"Using command line carrier: {carrier}")
    else:
        carrier = DEFAULT_CARRIER
        print(f"Using carrier: {carrier or 'Auto-detect'}")
    
    if len(sys.argv) >= 4:
        s3_bucket = sys.argv[3]
        print(f"Using command line S3 bucket: {s3_bucket}")
    else:
        s3_bucket = DEFAULT_S3_BUCKET
        print(f"Using S3 bucket: {s3_bucket}")
    
    if len(sys.argv) >= 5:
        excel_file = sys.argv[4]
        print(f"Using command line Excel file: {excel_file}")
    else:
        excel_file = DEFAULT_EXCEL_FILE
        print(f"Using Excel file: {excel_file}")
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"\n❌ Error: File not found: {input_file}")
        print("Please provide a valid PDF file path.")
        print("Usage: python3 lambda_function.py <pdf_file_path> [carrier] [s3_bucket]")
        return
    
    # Create event for lambda handler
    event = {
        "input_key": input_file,
        "input_bucket": s3_bucket,  # For local files, use the S3 bucket as input bucket
        "carrier_name": carrier,
        "use_textract": True,
        "use_s3_clustering": True,  # Use S3 clustering for better processing
        "s3_bucket": s3_bucket,
        "output_bucket": s3_bucket,
        "excel_file_path": excel_file,
        "email_details": {
            "to": DEFAULT_EMAIL_TO,
            "subject": f"Invoice Processing: {os.path.basename(input_file)}",
            "message_id": f"msg-{int(time.time())}",
            "original_body": "Invoice processing completed"
        },
        "job_id": f"JOB-{int(time.time())}",
        "attachment_id": f"ATT-{int(time.time())}"
    }
    
    # Calculate the output paths that will be used (same as J_J.py structure)
    main_pdf_name = os.path.basename(input_file)
    clean_main_name = os.path.splitext(main_pdf_name)[0]
    clean_main_name = re.sub(r'[\\/*?:"<>|]', "_", clean_main_name)
    flatten_path = f"Output/{clean_main_name}/flatten_json/"  # Only for logging, not stored
    final_path = f"Output/{clean_main_name}/final_json/"
    
    print(f"\n🚀 Starting invoice processing...")
    print(f"📄 Input file: {input_file}")
    print(f"🏢 Carrier: {carrier or 'Auto-detect'}")
    print(f"☁️  S3 bucket: {s3_bucket}")
    print(f"📊 Excel mapping file: {excel_file}")
    print(f"📁 Flattened JSONs will be logged (not saved to S3): {flatten_path}")
    print(f"📁 Final JSONs will be saved to: {final_path}")
    print("=" * 60)
    
    try:
        # Call the lambda handler
        result = lambda_handler(event, None)
        
        # Parse and display results
        if result['statusCode'] == 200:
            body = json.loads(result['body'])
            print("\n✅ PROCESSING COMPLETED SUCCESSFULLY!")
            print("=" * 60)
            print(f"📊 Summary: {body['message']}")
            
            # Show J_J.py processing results
            jj_processing = body.get('jj_processing', {})
            if jj_processing:
                print(f"\n🔍 J_J.py Processing Results:")
                print(f"  • Carrier: {jj_processing.get('carrier', 'Unknown')}")
                print(f"  • Total clusters: {jj_processing.get('total_clusters', 0)}")
                print(f"  • Successful: {jj_processing.get('successful_clusters', 0)}")
                print(f"  • Failed: {jj_processing.get('failed_clusters', 0)}")
                print(f"  • Skipped: {jj_processing.get('skipped_clusters', 0)}")
            
            # Show flattening results
            flatten_results = body.get('flatten_results', [])
            if flatten_results:
                print(f"\n📝 JSON Flattening Results:")
                for result_item in flatten_results:
                    filename = result_item.get('input_filename', 'Unknown')
                    status = result_item.get('status', 'Unknown')
                    if status == 'success':
                        location = result_item.get('flatten_location', 'Unknown')
                        print(f"  ✅ {filename}: {location}")
                    else:
                        error = result_item.get('error', 'Unknown error')
                        print(f"  ❌ {filename}: {error}")
            
            # Show final processing results
            final_results = body.get('final_results', [])
            if final_results:
                print(f"\n🎯 Final Processing Results:")
                for result_item in final_results:
                    filename = result_item.get('filename', 'Unknown')
                    status = result_item.get('status', 'Unknown')
                    mapping_applied = result_item.get('mapping_applied', False)
                    if status == 'success':
                        location = result_item.get('output_location', 'Unknown')
                        mapping_status = " (with Excel mapping)" if mapping_applied else " (as-is)"
                        print(f"  ✅ {filename}: {location}{mapping_status}")
                    else:
                        error = result_item.get('error', 'Unknown error')
                        print(f"  ❌ {filename}: {error}")
            
            print(f"\n📁 Flattened JSONs logged (not saved to S3): {flatten_path}")
            print(f"📁 Final JSONs saved to: {final_path}")
            
        else:
            print(f"\n❌ PROCESSING FAILED!")
            print("=" * 60)
            body = json.loads(result['body'])
            print(f"Error: {body.get('error', 'Unknown error')}")
            
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
