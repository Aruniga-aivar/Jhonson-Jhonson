import csv
import hashlib
import io
import json
import logging
import os
import random
import smtplib
import ssl
import time
import uuid
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.utils import formatdate, make_msgid
from collections import defaultdict

import boto3
from boto3.dynamodb.conditions import Key, Attr
from botocore.config import Config
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils.dataframe import dataframe_to_rows

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Initialize AWS clients
dynamodb = boto3.resource("dynamodb")
s3 = boto3.client("s3", config=Config(signature_version='s3v4'))
secretsmanager = boto3.client("secretsmanager")

# Get environment variables
JNJ_TABLE_NAME = os.environ.get("JNJ_DYNAMODB_TABLE", "jnj_logs")
OUTPUT_BUCKET = os.environ.get("OUTPUT_BUCKET", "")
OUTPUT_PREFIX = os.environ.get("OUTPUT_PREFIX", "invoice-reports/")

# Email configuration
SMTP_SERVER = os.environ.get("SMTP_SERVER", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "587"))
SMTP_SECRET_NAME = os.environ.get("SMTP_SECRET_NAME", "")
FROM_EMAIL_JNJ = os.environ.get("FROM_EMAIL_JNJ", "")

# Email recipients - J&J recipients only
JNJ_RECIPIENTS = [
   "Shashank.tiwari@pando.ai",
    "sadhanand.moorthy@pando.ai",
    "CString3@ITS.JNJ.com",
    "PPearson@ITS.JNJ.com",
    "jeeva@pando.ai" 
]


class DecimalEncoder(json.JSONEncoder):
    """Helper class to convert Decimal types to floats for JSON serialization"""

    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super(DecimalEncoder, self).default(obj)


def get_smtp_credentials(secret_name):
    """Retrieve SMTP credentials from AWS Secrets Manager"""
    try:
        if not secret_name:
            logger.error("SMTP_SECRET_NAME environment variable not set")
            return None, None
            
        response = secretsmanager.get_secret_value(SecretId=secret_name)
        secret_string = response['SecretString']
        secret_data = json.loads(secret_string)
        
        username = secret_data.get('SMTP_USERNAME')
        password = secret_data.get('SMTP_PASSWORD')
        
        if not username or not password:
            logger.error(f"SMTP credentials not found in secret {secret_name}")
            return None, None
            
        logger.info(f"Successfully retrieved SMTP credentials from secret {secret_name}")
        return username, password
        
    except Exception as e:
        logger.error(f"Error retrieving SMTP credentials from secret {secret_name}: {str(e)}")
        return None, None


def extract_field_value(extracted_fields, field_name):
    """Extract a specific field value from the extracted_fields array"""
    if not extracted_fields:
        return None

    for field in extracted_fields:
        if field.get("field_name") == field_name:
            return field.get("value")
    return None


def extract_missing_fields(item):
    """Extract missing fields from DynamoDB item"""
    missing_fields = []
    
    # Get missing_fields array from DynamoDB
    missing_fields_data = item.get("missing_fields", [])
    
    if isinstance(missing_fields_data, list):
        for field in missing_fields_data:
            if isinstance(field, dict) and "M" in field:
                # Convert DynamoDB map to Python dict
                field_data = {}
                for k, v in field["M"].items():
                    if "S" in v:
                        field_data[k] = v["S"]
                missing_fields.append(field_data)
            elif isinstance(field, dict):
                missing_fields.append(field)
    
    return missing_fields


def extract_external_field_errors(item):
    """Extract external field errors (missing mandatory fields) from DynamoDB item"""
    external_errors = []
    
    # Get external_field_errors array from DynamoDB
    external_errors_data = item.get("external_field_errors", [])
    
    if isinstance(external_errors_data, list):
        for field in external_errors_data:
            if isinstance(field, dict) and "M" in field:
                # Convert DynamoDB map to Python dict
                field_data = {}
                for k, v in field["M"].items():
                    if "S" in v:
                        field_data[k] = v["S"]
                external_errors.append(field_data)
            elif isinstance(field, dict):
                external_errors.append(field)
    
    return external_errors


def extract_internal_field_errors(item):
    """Extract internal field errors (format validation failures) from DynamoDB item"""
    internal_errors = []
    
    # Get internal_field_errors array from DynamoDB
    internal_errors_data = item.get("internal_field_errors", [])
    
    if isinstance(internal_errors_data, list):
        for field in internal_errors_data:
            if isinstance(field, dict) and "M" in field:
                # Convert DynamoDB map to Python dict
                field_data = {}
                for k, v in field["M"].items():
                    if "S" in v:
                        field_data[k] = v["S"]
                internal_errors.append(field_data)
            elif isinstance(field, dict):
                internal_errors.append(field)
    
    return internal_errors


def extract_rejection_reason(item):
    """Extract rejection_reason from error object in DynamoDB item"""
    error = item.get("error", {})
    
    # Handle DynamoDB map format (error.M.rejection_reason.S)
    if isinstance(error, dict) and "M" in error:
        error_map = error["M"]
        rejection_reason_attr = error_map.get("rejection_reason", {})
        if isinstance(rejection_reason_attr, dict) and "S" in rejection_reason_attr:
            return rejection_reason_attr["S"]
        # Also check for direct string value
        elif isinstance(rejection_reason_attr, str):
            return rejection_reason_attr
    
    # Handle regular dictionary format
    elif isinstance(error, dict):
        return error.get("rejection_reason", "")
    
    return ""


def extract_shipment_number(extracted_fields):
    """Extract shipment number from extracted_fields array"""
    # First try direct shipment_number field
    shipment_num = extract_field_value(extracted_fields, "shipments[0].shipment_number")

    # If not found, try other possible paths
    if not shipment_num:
        shipment_num = extract_field_value(extracted_fields, "shipment_number")

    return shipment_num


def extract_invoice_date(extracted_fields):
    """Extract invoice date from extracted_fields array"""
    return extract_field_value(extracted_fields, "invoice_date")


def extract_payment_due_date(extracted_fields):
    """Extract payment due date from extracted_fields array"""
    return extract_field_value(extracted_fields, "payment_due_date")


def get_api_status_and_message(api_response):
    """Extract API status and message from the API response"""
    status = "Failed"
    message = "No API response available"

    if not api_response:
        return status, message

    # Extract status
    success = api_response.get("success", False)
    status = "Success" if success else "Failed"

    # Extract message from body
    body = api_response.get("body", "")
    if body:
        try:
            # Try to parse the body as JSON
            body_json = json.loads(body) if isinstance(body, str) else body
            message = body_json.get("message", body_json.get("error", str(body)))
        except Exception:
            # If parsing fails, use the body as is
            message = str(body)[:200]  # Limit message length

    return status, message


def get_email_metadata(table, email_id):
    """Get email metadata from DynamoDB"""
    try:
        response = table.get_item(
            Key={"pk": f"EMAIL#{email_id}", "sk": "METADATA"}
        )
        if "Item" in response:
            return response["Item"]
        return None
    except Exception as e:
        logger.error(f"Error fetching email metadata: {str(e)}")
        return None


def extract_sender_email(metadata):
    """Extract sender email from metadata with proper handling of different formats"""
    if not metadata:
        return None

    # Check for metadata.M.sender structure (DynamoDB format)
    if "metadata" in metadata:
        metadata_obj = metadata["metadata"]

        # Handle DynamoDB M (map) format
        if isinstance(metadata_obj, dict) and "M" in metadata_obj:
            sender_attr = metadata_obj["M"].get("sender", {})
            if "S" in sender_attr:
                return sender_attr["S"]

        # Handle regular dictionary format
        elif isinstance(metadata_obj, dict) and "sender" in metadata_obj:
            return metadata_obj["sender"]

    # Try direct 'sender' field
    if "sender" in metadata:
        return metadata["sender"]

    # Try 'from' field
    if "from" in metadata:
        return metadata["from"]

    # Try email_details object
    if "email_details" in metadata:
        email_details = metadata["email_details"]

        # Handle dictionary format
        if isinstance(email_details, dict):
            return email_details.get("sender") or email_details.get("from")

        # Handle JSON string format
        elif isinstance(email_details, str):
            try:
                email_details_dict = json.loads(email_details)
                return email_details_dict.get("sender") or email_details_dict.get("from")
            except Exception:
                pass

    # If all else fails, log the metadata structure for debugging
    logger.warning(
        f"Could not find sender email in metadata: {json.dumps(metadata, default=str)}"
    )
    return None


def extract_date(item):
    """Extract date from created_at_iso field"""
    if "created_at_iso" in item:
        try:
            date_str = item["created_at_iso"]
            if isinstance(date_str, str):
                return date_str.split("T")[0]  # Extract just the date part
        except Exception:
            pass

    return None


def combine_status_values(api_status, processing_status, field_status=None, missing_critical_field=None):
    """Combine API status and processing status into a single status value"""
    # Check for unclassified field status first
    if field_status and field_status.lower() == "unclassified":
        return "Failed-Unclassified Document"
    
    # Check for missing critical fields (rejected by system)
    if missing_critical_field and int(missing_critical_field) == 1:
        return "Rejected by system"
    
    # Handle specific status values from DynamoDB
    if processing_status == "REJECTED":
        return "Rejected by system"
    
    # If either status is "Failed", the result is "Failed"
    if api_status and api_status.lower() == "failed":
        return "Failed - API error"
    if processing_status and processing_status.lower() == "failed":
        return "Failed"
    
    # If both are "Success", the result is "Success"
    if (api_status and api_status.lower() == "success" and 
        processing_status and processing_status.lower() == "success"):
        return "Success"
    
    # If one is success and the other is empty/unknown, return the non-empty one
    if api_status and api_status.lower() == "success":
        return "Success"
    if processing_status and processing_status.lower() == "success":
        return "Success"
    
    # Default case - return the first non-empty status or "Unknown"
    return api_status or processing_status or "Unknown"


def generate_excel(invoice_data, s3_links=None, client_name="Invoice Report"):
    """Generate Excel file from invoice data with formatting"""
    try:
        # Create a new workbook
        wb = Workbook()
        ws = wb.active
        ws.title = "Invoice Summary Report"
        
        # Define headers
        headers = [
            "S.No",
            "Invoice Number",
            "Invoice Date",
            "Carrier Name/SCAC",
            "Status of the invoice",
            "Reason for rejection (if any)",
            "Action Required"
        ]
        
        # Style for headers
        header_font = Font(bold=True, color="FFFFFF")
        header_fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
        header_alignment = Alignment(horizontal="center", vertical="center")
        
        # Add headers
        for col, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col, value=header)
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = header_alignment
        
        # Add data rows
        for row_idx, invoice in enumerate(invoice_data, 2):
            # Combine API status and processing status
            combined_status = combine_status_values(
                invoice.get("api_status", ""),
                invoice.get("status", ""),
                invoice.get("field_status", ""),
                invoice.get("missing_critical_field", 0)
            )
            
            # Format invoice date - handle DynamoDB date format
            invoice_date = invoice.get("invoice_date", "")
            if invoice_date:
                # Handle DynamoDB date format: 2025-04-08T00:00:00.000Z
                if "T" in invoice_date:
                    invoice_date = invoice_date.split("T")[0]  # Extract just the date part
                elif " " in invoice_date:
                    invoice_date = invoice_date.split(" ")[0]  # Handle space-separated format
                # If it's already in YYYY-MM-DD format, keep as is
            else:
                invoice_date = ""
            
            # Get carrier name/SCAC
            carrier_name = invoice.get("carrier_name", "")
            if not carrier_name:
                # Extract from sender email if carrier_name is not available
                sender_email = invoice.get("sender_email", "")
                if sender_email and sender_email != "unknown@example.com":
                    carrier_name = sender_email
                else:
                    carrier_name = "Unknown"
            
            # Determine reason for rejection
            # First, try to get rejection_reason from the invoice data (extracted from error object)
            rejection_reason = invoice.get("rejection_reason", "")
            
            # If no rejection_reason found, fall back to status-based determination
            if not rejection_reason and combined_status in ["Failed - API error", "Failed", "Rejected by system"]:
                if combined_status == "Failed - API error":
                    rejection_reason = "API processing error"
                elif combined_status == "Failed":
                    rejection_reason = "Processing failure"
                elif combined_status == "Rejected by system":
                    # Get missing field details
                    missing_fields = invoice.get("missing_fields", [])
                    if missing_fields:
                        field_names = [field.get("field_name", "Unknown field") for field in missing_fields]
                        rejection_reason = f"Missing mandatory fields: {', '.join(field_names)}"
                    else:
                        rejection_reason = "Missing mandatory fields"
                else:
                    rejection_reason = "Processing error"
            
            # Determine action required based on error types
            action_required = "No action required"
            external_errors = invoice.get("external_field_errors", [])
            internal_errors = invoice.get("internal_field_errors", [])
            
            # Priority: External errors > Internal errors > Success > Other failures
            if external_errors and len(external_errors) > 0:
                # External errors (missing mandatory fields from customer side) - highest priority
                action_required = "Action required by sender"
            elif internal_errors and len(internal_errors) > 0:
                # Internal errors (format validation failures) - alert already sent to Pando team, no action needed in report
                action_required = "No action required"
            elif combined_status == "Success":
                # Success - no action needed
                action_required = "No action required"
            else:
                # Other failures (API errors, unclassified, etc.) - may need investigation
                action_required = "Review required"
            
            row_data = [
                row_idx - 1,  # S.No (row number starting from 1)
                invoice.get("invoice_number", ""),
                invoice_date,
                carrier_name,
                combined_status,
                rejection_reason,
                action_required
            ]
            
            for col, value in enumerate(row_data, 1):
                ws.cell(row=row_idx, column=col, value=value)
        
        # Auto-adjust column widths
        for column in ws.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            ws.column_dimensions[column_letter].width = adjusted_width
        
        # Add borders
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in ws.iter_rows():
            for cell in row:
                cell.border = thin_border
        
        # Save to BytesIO
        excel_buffer = io.BytesIO()
        wb.save(excel_buffer)
        excel_buffer.seek(0)
        
        return excel_buffer.getvalue()
        
    except Exception as e:
        logger.error(f"Error generating Excel file: {str(e)}")
        return None


def generate_csv(invoice_data, s3_links=None):
    """Generate CSV content from invoice data"""
    # Define CSV headers
    headers = [
        "Invoice #",
        "Shipment #",
        "Amount",
        "Currency",
        "Invoice Date",
        "Payment Due Date",
        "Ingestion Status",
        "Sender Email ID",
        "Invoice PDF Link",
        "Filename",
    ]

    # Create CSV in memory
    csv_buffer = io.StringIO()
    csv_writer = csv.writer(csv_buffer)

    # Write headers
    csv_writer.writerow(headers)

    # Write data rows
    for invoice in invoice_data:
        # Combine API status and processing status
        combined_status = combine_status_values(
            invoice.get("api_status", ""),
            invoice.get("status", ""),
            invoice.get("field_status", ""),
            invoice.get("missing_critical_field", 0)
        )
        
        # Get S3 link for this invoice
        s3_link = ""
        if s3_links and invoice.get("email_id") in s3_links:
            s3_link = s3_links[invoice.get("email_id")]
        
        csv_writer.writerow(
            [
                invoice.get("invoice_number", ""),
                invoice.get("shipment_number", ""),
                invoice.get("total_invoice_value", ""),
                invoice.get("currency", ""),
                invoice.get("invoice_date", ""),
                invoice.get("payment_due_date", ""),
                combined_status,
                invoice.get("sender_email", ""),
                s3_link,
                invoice.get("filename", ""),
            ]
        )

    # Get the CSV content
    csv_content = csv_buffer.getvalue()
    csv_buffer.close()

    return csv_content


def generate_html_table(invoice_data, s3_links=None):
    """Generate HTML table from invoice data"""
    if not invoice_data:
        return "<p>No invoice data available.</p>"
    
    # Count different status types
    status_counts = {
        "Success": 0,
        "Failed - API error": 0,
        "Failed": 0,
        "Failed-Unclassified Document": 0,
        "Rejected by system": 0,
        "Other": 0
    }
    
    # Start building the HTML table
    html_table = """
    <div style="overflow-x: auto; margin: 20px 0;">
        <table style="border-collapse: collapse; width: 100%; font-family: Arial, sans-serif; min-width: 1200px;">
            <thead>
                <tr>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Invoice #</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Shipment #</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Amount</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Currency</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Invoice Date</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Payment Due Date</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Ingestion Status</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Sender Email</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Invoice PDF Link</th>
                    <th style="border: 1px solid #ccc; padding: 12px; text-align: left; font-weight: bold;">Filename</th>
                </tr>
            </thead>
            <tbody>
    """
    
    # Add data rows and count statuses
    for i, invoice in enumerate(invoice_data):
        
        # Format the amount without currency prefix
        amount = invoice.get("total_invoice_value", "")
        currency = invoice.get("currency", "")
        formatted_amount = amount or ""
        
        # Combine API status and processing status
        combined_status = combine_status_values(
            invoice.get("api_status", ""),
            invoice.get("status", ""),
            invoice.get("field_status", ""),
            invoice.get("missing_critical_field", 0)
        )
        
        # Count the status
        if combined_status in status_counts:
            status_counts[combined_status] += 1
        else:
            status_counts["Other"] += 1
        
        # Get S3 link for this invoice
        s3_link = ""
        if s3_links and invoice.get("email_id") in s3_links:
            s3_link = s3_links[invoice.get("email_id")]
        
        html_table += f"""
                <tr>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("invoice_number", "")}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("shipment_number", "")}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{formatted_amount}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{currency}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("invoice_date", "")}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("payment_due_date", "")}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{combined_status}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("sender_email", "")}</td>
                    <td style="border: 1px solid #ccc; padding: 10px;"><a href="{s3_link}" target="_blank" style="color: #0066cc; text-decoration: none;">{s3_link[:50] + "..." if len(s3_link) > 50 else s3_link}</a></td>
                    <td style="border: 1px solid #ccc; padding: 10px;">{invoice.get("filename", "")}</td>
                </tr>
        """
    
    # Close the table
    html_table += """
            </tbody>
        </table>
    </div>
    """
    
    # Add summary section
    total_invoices = len(invoice_data)
    success_count = status_counts["Success"]
    rejected_count = status_counts["Rejected by system"]
    failed_count = status_counts["Failed - API error"] + status_counts["Failed"] + status_counts["Failed-Unclassified Document"]
    
    summary_html = f"""
    <div style="margin: 20px 0;">
        <h3>Invoice Processing Summary</h3>
        <p><strong>Total Invoices:</strong> {total_invoices}</p>
        <p><strong>Successfully Ingested into system:</strong> {success_count}</p>
        <p><strong>Rejected by system:</strong> {rejected_count}</p>
        <p><strong>Failed to ingest:</strong> {failed_count}</p>
        <p><em>Rejected includes missing mandatory fields for processing invoices.</em></p>
        <p><em>Failed includes API errors, Unclassified documents, and other processing failures</em></p>
    </div>
    """
    
    return html_table + summary_html


def get_carrier_name_from_email(email):
    """Extract carrier name from email address"""
    if not email:
        return "Unknown Carrier"
    
    # Extract domain part and clean it up
    if "@" in email:
        domain = email.split("@")[0]
        # Clean up common patterns
        domain = domain.replace(".", " ").replace("_", " ").replace("-", " ")
        # Capitalize first letter of each word
        return " ".join(word.capitalize() for word in domain.split())
    
    return "Unknown Carrier"


# CARRIER-SPECIFIC EMAIL FUNCTION - COMMENTED OUT FOR LATER USE
# def send_carrier_email(to_email, subject, body, excel_content, filename, smtp_username, smtp_password, carrier_name, total_invoices, success_count, rejected_count, failed_count):
#     """Send email to carrier with carrier-specific content"""
#     try:
#         # Create a multipart message and set headers
#         msg = MIMEMultipart("alternative")
#         
#         # Set subject with "Re:" prefix if not already present
#         if not subject.lower().startswith("re:"):
#             msg["Subject"] = f"Re: {subject}"
#         else:
#             msg["Subject"] = subject
#             
#         msg["From"] = FROM_EMAIL
#         msg["To"] = to_email
#         
#         # Create carrier-specific HTML content
#         html_content = f"""
#         <!DOCTYPE html>
#         <html>
#         <head>
#             <meta charset="UTF-8">
#             <style>
#                 body {{
#                     font-family: Arial, sans-serif;
#                     line-height: 1.6;
#                     color: #333;
#                     max-width: 1200px;
#                     margin: 0 auto;
#                     padding: 20px;
#                 }}
#                 .header {{
#                     padding: 20px 0;
#                     margin-bottom: 20px;
#                     border-bottom: 1px solid #ccc;
#                 }}
#                 .summary {{
#                     padding: 15px 0;
#                     margin-bottom: 20px;
#                 }}
#                 .footer {{
#                     margin-top: 30px;
#                     padding-top: 20px;
#                     border-top: 1px solid #ccc;
#                 }}
#             </style>
#         </head>
#         <body>
#             <div class="header">
#                 <h2 style="margin: 0;">Invoice Ingestion Report</h2>
#                 <p style="margin: 5px 0 0 0;">Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
#             </div>
#             
#             <div class="summary">
#                 <p>Kindly find the consolidated invoice ingestion report attached for the past 24 hours.</p>
#                 <p>This report contains <strong>{total_invoices}</strong> invoices which were received from <strong>{carrier_name}</strong>.</p>
#                 <p>The excel file attached provides the details of each invoice along with its status. In case the status of the invoice is rejected, you are requested to resubmit the invoice after making necessary corrections as suggested in the excel file.</p>
#                 <p><strong>Please send (email) revised invoices after making necessary corrections as suggested in the excel attachment to:</strong></p>
#                 <ul>
#                     <li>invoices.meta@pibypando.ai (Meta)</li>
#                     <li>invoices.jnjnorthamerica@pibypando.ai (J&J)</li>
#                 </ul>
#             </div>
#             
#             <div style="margin: 20px 0;">
#                 <h3>Invoice Processing Summary</h3>
#                 <p><strong>Total number of Invoices:</strong> {total_invoices}</p>
#                 <p><strong>Successfully Ingested into system:</strong> {success_count}</p>
#                 <p><strong>Rejected by system:</strong> {rejected_count}</p>
#                 <p><strong>Failed to ingest:</strong> {failed_count}</p>
#                 <p><em>Rejected includes missing mandatory fields for processing invoices.</em></p>
#                 <p><em>Failed includes API errors, Unclassified documents, and other processing failures</em></p>
#             </div>
#             
#             <div class="footer">
#                 <p>This report was generated automatically. Please contact support if you have any questions.</p>
#                 <p>Best regards,<br>Pando Invoice Processing System</p>
#             </div>
#         </body>
#         </html>
#         """
# 
#         # Create plain text version
#         plain_text = f"""
# Invoice Ingestion Report - {datetime.now().strftime('%Y-%m-%d')}
# 
# Kindly find the consolidated invoice ingestion report attached for the past 24 hours.
# 
# This report contains {total_invoices} invoices which were received from {carrier_name}.
# 
# The excel file attached provides the details of each invoice along with its status. In case the status of the invoice is rejected, you are requested to resubmit the invoice after making necessary corrections as suggested in the excel file.
# 
# Please send (email) revised invoices after making necessary corrections as suggested in the excel attachment to:
# - invoices.meta@pibypando.ai (Meta)
# - invoices.jnjnorthamerica@pibypando.ai (J&J)
# 
# Invoice Processing Summary
# Total number of Invoices: {total_invoices}
# Successfully Ingested into system: {success_count}
# Failed to ingest: {failed_count}
# 
# 
# Failed includes API errors, Unclassified documents, and other processing failures
# 
# This report was generated automatically. Please contact support if you have any questions.
# 
# Best regards,
# Pando Invoice Processing System
#         """
# 
#         # Attach parts into message container
#         part1 = MIMEText(plain_text, "plain")
#         part2 = MIMEText(html_content, "html")
#         msg.attach(part1)
#         msg.attach(part2)
# 
#         # Attach Excel file
#         excel_attachment = MIMEBase('application', 'octet-stream')
#         excel_attachment.set_payload(excel_content)
#         encoders.encode_base64(excel_attachment)
#         excel_attachment.add_header('Content-Disposition', 'attachment', filename=filename)
#         msg.attach(excel_attachment)
# 
#         # Send the email via SMTP
#         context = ssl.create_default_context()
#         with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
#             server.starttls(context=context)
#             server.login(smtp_username, smtp_password)
#             server.send_message(msg)
#             logger.info(f"Carrier email sent successfully to {to_email}!")
#             return True
#     except Exception as e:
#         logger.error(f"Error sending carrier email to {to_email}: {str(e)}")
#         return False


def send_sender_daily_summary_email(sender_email, subject, excel_content, filename, smtp_username, smtp_password, sender_invoices, s3_links, from_email, twenty_four_hours_ago, now, retry_count=3):
    """Send daily consolidated summary email to individual sender with their invoice report"""
    try:
        # Validate email address format FIRST (before creating message)
        if not sender_email or "@" not in sender_email:
            logger.error(f"Invalid sender email address: {sender_email}")
            return False
        
        # Validate that we have invoices to send
        if not sender_invoices or len(sender_invoices) == 0:
            logger.warning(f"No invoices to send for {sender_email}")
            return False
        
        # Validate Excel content
        if not excel_content:
            logger.error(f"Excel content is None or empty for {sender_email}")
            return False
        
        # Create a multipart message and set headers
        msg = MIMEMultipart("alternative")
        
        # Set subject - use the subject passed from lambda_handler (already formatted with date range)
        # Remove "Re:" prefix if present
        clean_subject = subject.replace("Re: ", "").replace("re: ", "").strip()
        msg["Subject"] = clean_subject
            
        msg["From"] = from_email
        msg["To"] = sender_email
        
        # Add important email headers for better deliverability
        msg["Date"] = formatdate(localtime=True)
        # Generate VERY unique Message-ID with timestamp and sender email hash to avoid deduplication
        unique_id = hashlib.md5(f"{sender_email}{time.time()}{random.random()}".encode()).hexdigest()[:12]
        timestamp = int(time.time() * 1000)  # Millisecond precision
        msg["Message-ID"] = f"<individual-{unique_id}-{timestamp}@{from_email.split('@')[1] if '@' in from_email else 'pibypando.ai'}>"
        msg["X-Mailer"] = "Pando Invoice Processing System - Individual Sender Report"
        msg["X-Priority"] = "1"  # High priority (1 = High, 3 = Normal, 5 = Low)
        msg["Importance"] = "High"  # Outlook/Exchange importance header
        msg["X-Email-Type"] = "Individual-Daily-Summary"  # Custom header to distinguish from consolidated email
        msg["X-Report-Type"] = "Personal"  # Additional distinguishing header
        msg["X-Invoice-Count"] = str(len(sender_invoices))  # Number of invoices in this report
        
        # Calculate statistics for this sender
        status_counts = {"Success": 0, "Failed - API error": 0, "Failed": 0, "Failed-Unclassified Document": 0, "Rejected by system": 0, "Other": 0}
        for invoice in sender_invoices:
            combined_status = combine_status_values(
                invoice.get("api_status", ""),
                invoice.get("status", ""),
                invoice.get("field_status", ""),
                invoice.get("missing_critical_field", 0)
            )
            if combined_status in status_counts:
                status_counts[combined_status] += 1
            else:
                status_counts["Other"] += 1
        
        total_invoices = len(sender_invoices)
        success_count = status_counts["Success"]
        rejected_count = status_counts["Rejected by system"]
        failed_count = status_counts["Failed - API error"] + status_counts["Failed"] + status_counts["Failed-Unclassified Document"]
        
        # Create sender-specific HTML content (without HTML table - Excel file already contains all details)
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    padding: 20px 0;
                    margin-bottom: 20px;
                    border-bottom: 1px solid #ccc;
                }}
                .summary {{
                    padding: 15px 0;
                    margin-bottom: 20px;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ccc;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Your Daily Invoice Summary</h2>
                <p style="margin: 5px 0 0 0;">Generated on {now.strftime('%Y-%m-%d %H:%M:%S IST')}</p>
            </div>
            
            <div class="summary">
                <p>Dear Valued Customer,</p>
                <p>Kindly find your consolidated invoice submission report for today ({twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}).</p>
                <p>This report contains <strong>{total_invoices}</strong> invoice(s) that you submitted today.</p>
                <p>The Excel file attached provides the complete details of each invoice along with its processing status. Please review the Excel attachment for all invoice information.</p>
                <p>In case the status of any invoice is rejected, please resubmit the invoice after making necessary corrections as suggested in the Excel file.</p>
            </div>
            
            <div class="footer">
                <p><strong>Please send revised invoices after making necessary corrections to:</strong></p>
                <ul>
                    <li>invoices.jnjnorthamerica@pibypando.ai (J&J)</li>
                </ul>
                <p>This report was generated automatically. Please contact support if you have any questions.</p>
                <p>Best regards,<br>Pando Invoice Processing System</p>
            </div>
        </body>
        </html>
        """

        # Create plain text version
        plain_text = f"""
Daily Invoice Summary
Generated on {now.strftime('%Y-%m-%d %H:%M:%S IST')}

Dear Valued Customer,

Kindly find your consolidated invoice submission report for today ({twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}).

This report contains {total_invoices} invoice(s) that you submitted today.

The Excel file attached provides the complete details of each invoice along with its processing status. Please review the Excel attachment for all invoice information.

In case the status of any invoice is rejected, please resubmit the invoice after making necessary corrections as suggested in the Excel file.

Please send revised invoices after making necessary corrections to:
- invoices.jnjnorthamerica@pibypando.ai (J&J)

This report was generated automatically. Please contact support if you have any questions.

Best regards,
Pando Invoice Processing System
        """

        # Attach parts into message container
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        # Attach Excel file
        excel_attachment = MIMEBase('application', 'octet-stream')
        excel_attachment.set_payload(excel_content)
        encoders.encode_base64(excel_attachment)
        excel_attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(excel_attachment)
        
        logger.info(f"📎 Excel attachment added: {filename} ({len(excel_content)} bytes)")

        # Check attachment size (some email servers have size limits)
        attachment_size_mb = len(excel_content) / (1024 * 1024)
        if attachment_size_mb > 25:  # 25MB is a common email limit
            logger.warning(f"⚠️  Email attachment is large ({attachment_size_mb:.2f} MB) for {sender_email}. This might cause delivery issues.")
        
        logger.info("=" * 80)
        logger.info(f"📨 INDIVIDUAL EMAIL MESSAGE CONSTRUCTED FOR: {sender_email}")
        logger.info("=" * 80)
        logger.info(f"   From: {msg['From']}")
        logger.info(f"   To: {msg['To']}")
        logger.info(f"   Subject: {msg['Subject']}")
        logger.info(f"   Message-ID: {msg['Message-ID']}")
        logger.info(f"   Priority: {msg['X-Priority']} (High)")
        logger.info(f"   Importance: {msg['Importance']}")
        logger.info(f"   X-Email-Type: {msg['X-Email-Type']}")
        logger.info(f"   X-Report-Type: {msg['X-Report-Type']}")
        logger.info(f"   X-Invoice-Count: {msg['X-Invoice-Count']}")
        logger.info(f"   Attachment: {filename} ({attachment_size_mb:.2f} MB)")
        logger.info("=" * 80)
        
        # Add Reply-To header for better email deliverability
        msg["Reply-To"] = from_email
        
        # Send the email via SMTP with retry logic
        context = ssl.create_default_context()
        last_exception = None
        
        for attempt in range(1, retry_count + 1):
            try:
                logger.info(f"Attempting to send email to {sender_email} (attempt {attempt}/{retry_count})")
                
                with smtplib.SMTP(SMTP_SERVER, SMTP_PORT, timeout=30) as server:
                    server.set_debuglevel(0)  # Set to 1 for verbose SMTP debugging
                    server.starttls(context=context)
                    server.login(smtp_username, smtp_password)
                    
                    # send_message returns a dict of failed recipients (empty dict = success)
                    logger.info("=" * 80)
                    logger.info(f"🚀 SENDING EMAIL VIA SMTP TO: {sender_email}")
                    logger.info(f"   SMTP Server: {SMTP_SERVER}:{SMTP_PORT}")
                    logger.info(f"   Attempt: {attempt}/{retry_count}")
                    logger.info("=" * 80)
                    
                    failed_recipients = server.send_message(msg, to_addrs=[sender_email])
                    
                    if failed_recipients:
                        logger.error("=" * 80)
                        logger.error(f"❌ EMAIL DELIVERY FAILED for {sender_email} on attempt {attempt}")
                        logger.error(f"   Failed recipients: {failed_recipients}")
                        logger.error("=" * 80)
                        if attempt < retry_count:
                            time.sleep(2 ** attempt)  # Exponential backoff: 2s, 4s, 8s
                            continue
                        return False
                    
                    # Log email details for debugging
                    logger.info("=" * 80)
                    logger.info(f"✅ ✅ ✅ EMAIL SUCCESSFULLY SENT TO {sender_email}! ✅ ✅ ✅")
                    logger.info("=" * 80)
                    logger.info(f"📧 COMPLETE EMAIL DETAILS:")
                    logger.info(f"   From: {from_email}")
                    logger.info(f"   To: {sender_email}")
                    logger.info(f"   Subject: {msg['Subject']}")
                    logger.info(f"   Message-ID: {msg['Message-ID']}")
                    logger.info(f"   Priority: HIGH")
                    logger.info(f"   Attachment: {filename} ({attachment_size_mb:.2f} MB)")
                    logger.info(f"   Invoice Count: {len(sender_invoices)}")
                    logger.info("=" * 80)
                    logger.info(f"✉️  SMTP SERVER ({SMTP_SERVER}) ACCEPTED MESSAGE FOR DELIVERY")
                    logger.info(f"⚠️  CRITICAL NOTE: Email was successfully sent by AWS Lambda")
                    logger.info(f"⚠️  and accepted by Microsoft SMTP server for delivery.")
                    logger.info(f"⚠️  If you don't receive it, it's being filtered by Gmail/Outlook")
                    logger.info(f"⚠️  CHECK YOUR SPAM/JUNK FOLDER FOR:")
                    logger.info(f"⚠️     Subject: {msg['Subject']}")
                    logger.info(f"⚠️     From: {from_email}")
                    logger.info("=" * 80)
                    return True
                    
            except smtplib.SMTPRecipientsRefused as e:
                logger.error(f"SMTP recipients refused for {sender_email} on attempt {attempt}: {str(e)}")
                logger.error(f"Rejected recipients details: {e.recipients}")
                # Don't retry if recipient is refused - it's a permanent error
                return False
            except smtplib.SMTPDataError as e:
                logger.error(f"SMTP data error for {sender_email} on attempt {attempt}: {str(e)}")
                logger.error(f"SMTP error code: {e.smtp_code}, message: {e.smtp_error}")
                last_exception = e
                if attempt < retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False
            except smtplib.SMTPServerDisconnected as e:
                logger.warning(f"SMTP server disconnected for {sender_email} on attempt {attempt}: {str(e)}")
                last_exception = e
                if attempt < retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False
            except smtplib.SMTPAuthenticationError as e:
                logger.error(f"SMTP authentication error for {sender_email}: {str(e)}")
                # Don't retry authentication errors - they're permanent
                return False
            except smtplib.SMTPException as e:
                logger.warning(f"SMTP exception for {sender_email} on attempt {attempt}: {str(e)}")
                last_exception = e
                if attempt < retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False
            except Exception as e:
                logger.warning(f"Unexpected error sending email to {sender_email} on attempt {attempt}: {str(e)}")
                last_exception = e
                if attempt < retry_count:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return False
        
        # If we exhausted all retries
        logger.error(f"Failed to send email to {sender_email} after {retry_count} attempts. Last error: {str(last_exception)}")
        return False
        
    except Exception as e:
        logger.error(f"Error sending daily summary email to {sender_email}: {str(e)}", exc_info=True)
        return False


def send_client_email(to_emails, subject, body, excel_content, filename, smtp_username, smtp_password, total_invoices, success_count, rejected_count, failed_count, from_email, twenty_four_hours_ago, now, client_name):
    """Send email to multiple clients (Meta, J&J) and Pando with consolidated content"""
    try:
        # Create a multipart message and set headers
        msg = MIMEMultipart("alternative")
        
        # Set subject with "Re:" prefix if not already present
        if not subject.lower().startswith("re:"):
            msg["Subject"] = f"Re: {subject}"
        else:
            msg["Subject"] = subject
            
        msg["From"] = from_email
        
        # Handle multiple recipients
        if isinstance(to_emails, list):
            msg["To"] = ", ".join(to_emails)
            recipient_list = to_emails
        else:
            msg["To"] = to_emails
            recipient_list = [to_emails]
        
        # Add important email headers for better deliverability
        msg["Date"] = formatdate(localtime=True)
        msg["Message-ID"] = make_msgid(domain=from_email.split("@")[1] if "@" in from_email else "pibypando.ai")
        msg["X-Mailer"] = "Pando Invoice Processing System"
        msg["X-Priority"] = "3"  # Normal priority
        msg["Reply-To"] = from_email
        msg["X-Email-Type"] = "Consolidated-Report"  # Custom header to distinguish from individual emails
        
        # Determine the correct email address based on client
        if client_name.lower() == "meta":
            correction_email = "invoices.meta@pibypando.ai (Meta)"
        elif client_name.lower() == "j&j":
            correction_email = "invoices.jnjnorthamerica@pibypando.ai (J&J)"
        else:
            correction_email = "invoices.meta@pibypando.ai (Meta) and invoices.jnjnorthamerica@pibypando.ai (J&J)"
        
        # Create client-specific HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    padding: 20px 0;
                    margin-bottom: 20px;
                    border-bottom: 1px solid #ccc;
                }}
                .summary {{
                    padding: 15px 0;
                    margin-bottom: 20px;
                }}
                .footer {{
                    margin-top: 30px;
                    padding-top: 20px;
                    border-top: 1px solid #ccc;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h2 style="margin: 0;">Invoice Ingestion Report</h2>
                <p style="margin: 5px 0 0 0;">Generated on {now.strftime('%Y-%m-%d %H:%M:%S IST')}</p>
            </div>
            
            <div class="summary">
                <p>Kindly find the consolidated invoice ingestion report attached for the past 24 hours ({twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}).</p>
                <p>This report contains <strong>{total_invoices}</strong> invoices which were received from all carriers.</p>
                <p>The excel file attached provides the details of each invoice along with its status. In case the status of the invoice is rejected, carriers are requested to resubmit the invoice after making necessary corrections as suggested in the excel file.</p>
                <p><strong>Carriers should send (email) revised invoices after corrections to:</strong></p>
                <ul>
                    <li>{correction_email}</li>
                </ul>
            </div>
            
            <div style="margin: 20px 0;">
                <h3>Invoice Processing Summary</h3>
                <p><strong>Total number of Invoices:</strong> {total_invoices}</p>
                <p><strong>Successfully Ingested into system:</strong> {success_count}</p>
                <p><strong>Rejected by system:</strong> {rejected_count}</p>
                <p><strong>Failed to ingest:</strong> {failed_count}</p>
                <p><em>Rejected includes missing mandatory fields for processing invoices.</em></p>
                <p><em>Failed includes API errors, Unclassified documents, and other processing failures</em></p>
            </div>
            
            <div class="footer">
                <p>This report was generated automatically. Please contact support if you have any questions.</p>
                <p>Best regards,<br>Pando Invoice Processing System</p>
            </div>
        </body>
        </html>
        """

        # Create plain text version
        plain_text = f"""
Invoice Ingestion Report
Generated on {now.strftime('%Y-%m-%d %H:%M:%S IST')}

Kindly find the consolidated invoice ingestion report attached for the past 24 hours ({twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M')} to {now.strftime('%Y-%m-%d %H:%M')}).

This report contains {total_invoices} invoices which were received from all carriers.

The excel file attached provides the details of each invoice along with its status. In case the status of the invoice is rejected, carriers are requested to resubmit the invoice after making necessary corrections as suggested in the excel file.

Carriers should send (email) revised invoices after corrections to: {correction_email}

Invoice Processing Summary
Total number of Invoices: {total_invoices}
Successfully Ingested into system: {success_count}
Rejected by system: {rejected_count}
Failed to ingest: {failed_count}

Rejected includes missing mandatory fields for processing invoices.

Failed includes API errors, Unclassified documents, and other processing failures

This report was generated automatically. Please contact support if you have any questions.

Best regards,
Pando Invoice Processing System
        """

        # Attach parts into message container
        part1 = MIMEText(plain_text, "plain")
        part2 = MIMEText(html_content, "html")
        msg.attach(part1)
        msg.attach(part2)

        # Attach Excel file
        excel_attachment = MIMEBase('application', 'octet-stream')
        excel_attachment.set_payload(excel_content)
        encoders.encode_base64(excel_attachment)
        excel_attachment.add_header('Content-Disposition', 'attachment', filename=filename)
        msg.attach(excel_attachment)

        # Send the email via SMTP
        context = ssl.create_default_context()
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls(context=context)
            server.login(smtp_username, smtp_password)
            server.send_message(msg, to_addrs=recipient_list)
            logger.info(f"Client email sent successfully to {', '.join(recipient_list)}!")
            return True
    except Exception as e:
        logger.error(f"Error sending client email to {recipient_list}: {str(e)}")
        return False


def lambda_handler(event, context):
    """
    Lambda handler to extract invoice details from DynamoDB for the previous 24 hours 
    and send different email templates to carriers vs clients/Pando

    Parameters:
    - event: Can contain filters like email_id or attachment_id
    - context: Lambda context

    Returns:
    - S3 locations of the generated CSV files and email status
    """
    try:
        # Get SMTP credentials from Secrets Manager
        smtp_username, smtp_password = get_smtp_credentials(SMTP_SECRET_NAME)
        
        # Validate email configuration
        if not all([SMTP_SERVER, smtp_username, smtp_password, FROM_EMAIL_JNJ]):
            logger.warning("Email configuration incomplete. Emails will not be sent.")
            email_enabled = False
        else:
            email_enabled = True
            logger.info("Email configuration validated. Emails will be sent.")
        
        # Get J&J table
        jnj_table = dynamodb.Table(JNJ_TABLE_NAME)

        # Set date range to previous 24 hours (IST - Indian Standard Time, UTC+5:30)
        # IST timezone offset: UTC+5:30
        ist_offset = timezone(timedelta(hours=5, minutes=30))
        
        # Get current UTC time (Lambda runs in UTC)
        now_utc = datetime.now(timezone.utc)
        # Convert UTC to IST for display and calculation
        now_ist = now_utc.astimezone(ist_offset)
        
        # Calculate 24 hours ago in IST
        twenty_four_hours_ago_ist = (now_ist - timedelta(hours=24)).replace(microsecond=0)
        
        # Convert IST times back to UTC for DynamoDB query (since created_at_iso is stored in UTC)
        # This ensures we query exactly 24 hours in IST, regardless of when Lambda actually runs
        now_ist_utc = now_ist.astimezone(timezone.utc)
        twenty_four_hours_ago_utc = twenty_four_hours_ago_ist.astimezone(timezone.utc)
        
        # Use UTC times for DynamoDB query (converted from IST to ensure exact 24-hour IST window)
        now = now_ist_utc
        twenty_four_hours_ago = twenty_four_hours_ago_utc
        date_start = twenty_four_hours_ago.isoformat()
        date_end = now.isoformat()

        # Extract other filters from event
        email_id = event.get("email_id")
        attachment_id = event.get("attachment_id")
        limit = event.get("limit", 1000)  # Default limit to 1000 records

        logger.info(f"Querying invoices for previous 24 hours (IST): {twenty_four_hours_ago_ist.strftime('%Y-%m-%d %H:%M IST')} to {now_ist.strftime('%Y-%m-%d %H:%M IST')}")
        logger.info(f"Querying invoices for previous 24 hours (UTC): {twenty_four_hours_ago.strftime('%Y-%m-%d %H:%M UTC')} to {now.strftime('%Y-%m-%d %H:%M UTC')}")

        # Build query parameters for J&J table only
        if email_id and attachment_id:
            # Direct query by primary key
            response = jnj_table.get_item(
                Key={"pk": f"EMAIL#{email_id}", "sk": f"ATTACHMENT#{attachment_id}"}
            )
            items = [response.get("Item")] if "Item" in response else []

            # Filter by created_at_iso date range and exclude unclassified attachments
            items = [
                item
                for item in items
                if item.get("created_at_iso", "") >= date_start
                and item.get("created_at_iso", "") <= date_end
                and not (item.get("classification_failed") and int(item.get("classification_failed", 0)) == 1)
                and item.get("status", "").lower() != "unclassified"
                and item.get("field_status", "").lower() != "unclassified"
            ]

        elif email_id:
            # Query by email_id with filter to exclude unclassified attachments
            response = jnj_table.query(
                KeyConditionExpression=Key("pk").eq(f"EMAIL#{email_id}")
                & Key("sk").begins_with("ATTACHMENT#"),
                FilterExpression=Attr("created_at_iso").between(date_start, date_end)
                & Attr("classification_failed").ne(1)  # Exclude classification_failed = 1
                & Attr("status").ne("unclassified")    # Exclude status = "unclassified"
                & Attr("field_status").ne("unclassified")  # Exclude field_status = "unclassified"
            )
            items = response.get("Items", [])

        else:
            # For scan operation, exclude unclassified attachments at the database level
            scan_kwargs = {
                "FilterExpression": Attr("created_at_iso").between(date_start, date_end)
                & Attr("sk").begins_with("ATTACHMENT#")
                & Attr("classification_failed").ne(1)  # Exclude classification_failed = 1
                & Attr("status").ne("unclassified")    # Exclude status = "unclassified"
                & Attr("field_status").ne("unclassified"),  # Exclude field_status = "unclassified"
                "Limit": limit,
            }

            response = jnj_table.scan(**scan_kwargs)
            items = response.get("Items", [])

            # Handle pagination for large result sets
            while "LastEvaluatedKey" in response and len(items) < limit:
                scan_kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]
                response = jnj_table.scan(**scan_kwargs)
                items.extend(response.get("Items", []))
                if len(items) >= limit:
                    items = items[:limit]
                    break

        # Create a cache for email metadata to avoid repeated lookups
        email_metadata_cache = {}

        # Extract required fields from each item
        results = []
        
        # Process J&J table
        try:
            jnj_scan_kwargs = {
                "FilterExpression": Attr("created_at_iso").between(date_start, date_end)
                & Attr("sk").begins_with("ATTACHMENT#")
                & Attr("classification_failed").ne(1)
                & Attr("status").ne("unclassified")
                & Attr("field_status").ne("unclassified"),
                "Limit": limit,
            }
            
            logger.info(f"Scanning J&J table with date range: {date_start} to {date_end}")
            jnj_response = jnj_table.scan(**jnj_scan_kwargs)
            jnj_items = jnj_response.get("Items", [])
            logger.info(f"Initial scan returned {len(jnj_items)} items from J&J table")
            
            # Handle pagination
            while "LastEvaluatedKey" in jnj_response and len(jnj_items) < limit:
                jnj_scan_kwargs["ExclusiveStartKey"] = jnj_response["LastEvaluatedKey"]
                jnj_response = jnj_table.scan(**jnj_scan_kwargs)
                jnj_items.extend(jnj_response.get("Items", []))
                if len(jnj_items) >= limit:
                    jnj_items = jnj_items[:limit]
                    break
            
            logger.info(f"Total J&J items after pagination: {len(jnj_items)}")
            
            # Add client identifier to J&J items
            for item in jnj_items:
                item["client"] = "J&J"
            
        except Exception as e:
            logger.error(f"Error scanning J&J table: {str(e)}")
            jnj_items = []
        
        # Process J&J items
        for item in jnj_items:
            # Skip non-attachment items
            if not item.get("sk", "").startswith("ATTACHMENT#"):
                continue

            # Extract basic fields directly from DynamoDB
            invoice_number = item.get("invoice_number", "")
            invoice_date = item.get("invoice_date", "")
            payment_due_date = item.get("payment_due_date", "")

            # Extract email details
            email_parts = item.get("pk", "").split("#")
            email_id = email_parts[1] if len(email_parts) > 1 else ""

            # Get email metadata from cache or fetch from DynamoDB
            if email_id not in email_metadata_cache:
                email_metadata = get_email_metadata(jnj_table, email_id)
                email_metadata_cache[email_id] = email_metadata
            else:
                email_metadata = email_metadata_cache[email_id]

            # Extract sender email from metadata
            sender_email = extract_sender_email(email_metadata)
            
            # Get to_email from DynamoDB item, fallback to sender_email if not set
            to_email_from_db = item.get("to_email", "")
            logger.info(f"Processing invoice {invoice_number}: to_email from DB='{to_email_from_db}', sender_email from metadata='{sender_email}'")
            
            if not to_email_from_db or not to_email_from_db.strip():
                # If to_email is not set in DynamoDB, use sender_email as fallback
                to_email_from_db = sender_email or ""
                logger.info(f"Invoice {invoice_number}: Using fallback email='{to_email_from_db}'")

            # Extract date from created_at_iso
            date = extract_date(item)

            # Extract attachment details
            attachment_parts = item.get("sk", "").split("#")
            attachment_id = attachment_parts[1] if len(attachment_parts) > 1 else ""

            # Extract currency and total invoice value from extracted_fields
            extracted_fields = item.get("extracted_fields", [])

            # Convert DynamoDB format to Python dict if needed
            clean_fields = []
            if isinstance(extracted_fields, list):
                for field in extracted_fields:
                    if isinstance(field, dict) and "M" in field:
                        # Convert DynamoDB map to Python dict
                        clean_field = {}
                        for k, v in field["M"].items():
                            if "S" in v:
                                clean_field[k] = v["S"]
                            elif "N" in v:
                                clean_field[k] = Decimal(v["N"])
                        clean_fields.append(clean_field)
                    else:
                        clean_fields.append(field)

            # Extract currency and total invoice value
            currency = extract_field_value(clean_fields, "currency")
            total_invoice_value = extract_field_value(clean_fields, "total_invoice_value")

            # Extract shipment number
            shipment_number = extract_shipment_number(clean_fields)
            
            # Log the direct DynamoDB values
            logger.info(f"Invoice {invoice_number}: DynamoDB invoice_date: {invoice_date}, payment_due_date: {payment_due_date}")

            # Extract API status and message
            api_response = item.get("api_response", {})
            api_status, api_message = get_api_status_and_message(api_response)

            # Extract field status from the attachment record
            field_status = None
            # Check for field status in various possible locations
            if item.get("field_status"):
                field_status = item.get("field_status")
            elif item.get("classification_failed") and int(item.get("classification_failed", 0)) == 1:
                field_status = "unclassified"
            elif item.get("status", "").lower() == "unclassified":
                field_status = "unclassified"

            # Extract missing fields
            missing_fields = extract_missing_fields(item)
            
            # Extract external and internal field errors
            external_field_errors = extract_external_field_errors(item)
            internal_field_errors = extract_internal_field_errors(item)
            
            # Extract rejection_reason from error object
            rejection_reason = extract_rejection_reason(item)

            # Create result object
            result = {
                "email_id": email_id,
                "attachment_id": attachment_id,
                "invoice_number": invoice_number,
                "shipment_number": shipment_number,
                "sender_email": sender_email or "unknown@example.com",  # Default value if sender email is missing
                "currency": currency,
                "total_invoice_value": total_invoice_value,
                "invoice_date": invoice_date,
                "payment_due_date": payment_due_date,
                "api_status": api_status,
                "api_message": api_message,
                "status": item.get("status"),
                "field_status": field_status,  # Added field status
                "filename": item.get("filename", ""),
                "date": date,  # Added date field
                "s3_path": item.get("s3_path", ""),  # Added S3 path for original PDF
                "carrier_name": item.get("carrier_name", ""),  # Added carrier name
                "processing_type": item.get("processing_type", ""),  # Added processing type
                "confidence_score": item.get("confidence_score", ""),  # Added confidence score
                "created_at_iso": item.get("created_at_iso", ""),  # Added created date
                "completed_at_iso": item.get("completed_at_iso", ""),  # Added completed date
                "client": item.get("client", ""),  # Added client identifier
                "to_email": to_email_from_db,  # Added to_email field (with sender_email fallback if not in DB)
                "classification_failed": item.get("classification_failed", 0),  # Added classification status
                "extraction_failed": item.get("extraction_failed", 0),  # Added extraction status
                "format_failed": item.get("format_failed", 0),  # Added format status
                "textract_failed": item.get("textract_failed", 0),  # Added textract status
                "missing_critical_field": item.get("missing_critical_field", 0),  # Added missing critical field
                "missing_fields": missing_fields,  # Added missing fields array
                "external_field_errors": external_field_errors,  # Added external field errors (missing mandatory fields)
                "internal_field_errors": internal_field_errors,  # Added internal field errors (format validation failures)
                "rejection_reason": rejection_reason,  # Added rejection reason from error object
            }

            # Check if this would result in "Failed-Unclassified Document" status
            combined_status = combine_status_values(api_status, item.get("status"), field_status)
            
            # Skip rows that would have "Failed-Unclassified Document" status
            if combined_status == "Failed-Unclassified Document":
                logger.info(f"Skipping unclassified document: {item.get('sk', '')}")
                continue

            results.append(result)

        logger.info(f"Found {len(results)} invoice records for the previous 24 hours")
        
        # Log summary of found invoices
        if results:
            logger.info("=" * 60)
            logger.info("INVOICES FOUND IN THIS RUN:")
            for idx, result in enumerate(results[:10], 1):  # Log first 10 invoices
                logger.info(f"  {idx}. Invoice: {result.get('invoice_number', 'N/A')}, "
                           f"Email ID: {result.get('email_id', 'N/A')}, "
                           f"Created: {result.get('created_at_iso', 'N/A')}")
            if len(results) > 10:
                logger.info(f"  ... and {len(results) - 10} more invoices")
            logger.info("=" * 60)
        else:
            logger.warning("⚠️ No invoices found in this run!")

        # Group invoices by J&J only
        jnj_invoices = []
        
        for invoice in results:
            # All invoices are J&J in this file
            if invoice.get("client") == "J&J":
                jnj_invoices.append(invoice)

        # Create S3 links mapping for all results
        s3_links = {}
        for result in results:
            s3_path = result.get("s3_path", "")
            if s3_path:
                try:
                    if s3_path.startswith("s3://"):
                        path_parts = s3_path[5:].split("/", 1)
                        if len(path_parts) == 2:
                            bucket = path_parts[0]
                            key = path_parts[1]
                            pdf_presigned_url = s3.generate_presigned_url(
                                "get_object",
                                Params={"Bucket": bucket, "Key": key},
                                ExpiresIn=3600,  # 1 hour
                            )
                            s3_links[result.get("email_id", "")] = pdf_presigned_url
                        else:
                            s3_links[result.get("email_id", "")] = s3_path
                    else:
                        s3_links[result.get("email_id", "")] = s3_path
                except Exception as e:
                    logger.warning(f"Could not generate presigned URL for {s3_path}: {str(e)}")
                    s3_links[result.get("email_id", "")] = s3_path
            else:
                s3_links[result.get("email_id", "")] = "No S3 path available"

        # Calculate overall statistics
        total_invoices = len(results)
        status_counts = {
            "Success": 0,
            "Failed - API error": 0,
            "Failed": 0,
            "Failed-Unclassified Document": 0,
            "Rejected by system": 0,
            "Other": 0
        }
        
        for invoice in results:
            combined_status = combine_status_values(
                invoice.get("api_status", ""),
                invoice.get("status", ""),
                invoice.get("field_status", ""),
                invoice.get("missing_critical_field", 0)
            )
            if combined_status in status_counts:
                status_counts[combined_status] += 1
            else:
                status_counts["Other"] += 1

        success_count = status_counts["Success"]
        rejected_count = status_counts["Rejected by system"]
        failed_count = status_counts["Failed - API error"] + status_counts["Failed"] + status_counts["Failed-Unclassified Document"]

        # Create date range string for filenames (using IST times)
        date_range_str = f"{twenty_four_hours_ago_ist.strftime('%Y%m%d_%H%M')}_to_{now_ist.strftime('%Y%m%d_%H%M')}"
        
        # Generate consolidated CSV with all records
        all_csv_content = generate_csv(results, s3_links)
        all_filename = f"invoice_report_all_records_{date_range_str}.csv"
        all_s3_key = f"{OUTPUT_PREFIX}{all_filename}"
        
        # Upload consolidated CSV to S3
        s3.put_object(Bucket=OUTPUT_BUCKET, Key=all_s3_key, Body=all_csv_content, ContentType="text/csv")
        all_presigned_url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": OUTPUT_BUCKET, "Key": all_s3_key},
            ExpiresIn=3600,  # 1 hour
        )

        # Generate consolidated HTML table for all invoice data
        html_table = generate_html_table(results, s3_links)

        # Email sending logic - Send only J&J consolidated email
        email_results = {
            "jnj_email_sent": 0,
            "jnj_email_failed": 0
        }
        
        # Initialize sender_groups for grouping invoices by sender
        sender_groups = defaultdict(list)
        
        # Group invoices by sender email (to_email field) - always do this for reporting
        logger.info(f"Grouping {len(results)} invoices by sender email for individual summary emails")
        logger.info("=" * 60)
        logger.info("DETAILED INVOICE GROUPING ANALYSIS")
        logger.info("=" * 60)
        
        emails_without_valid_address = 0
        for idx, invoice in enumerate(results, 1):
            to_email = invoice.get("to_email", "").strip() if invoice.get("to_email") else ""
            sender_email_fallback = invoice.get("sender_email", "").strip() if invoice.get("sender_email") else ""
            
            logger.info(f"[Invoice {idx}/{len(results)}] {invoice.get('invoice_number', 'N/A')}:")
            logger.info(f"  - to_email from invoice: '{to_email}'")
            logger.info(f"  - sender_email from invoice: '{sender_email_fallback}'")
            
            # Skip placeholder/unknown emails
            if sender_email_fallback and sender_email_fallback.lower() in ["unknown@example.com", "unknown"]:
                logger.info(f"  - Clearing sender_email (placeholder detected)")
                sender_email_fallback = ""
            
            # Use to_email as primary, fallback to sender_email
            final_sender_email = None
            if to_email and "@" in to_email:
                final_sender_email = to_email
                logger.info(f"  ✅ GROUPED using to_email: {to_email}")
            elif sender_email_fallback and "@" in sender_email_fallback:
                final_sender_email = sender_email_fallback
                logger.info(f"  ✅ GROUPED using sender_email: {sender_email_fallback}")
            else:
                emails_without_valid_address += 1
                logger.warning(f"  ❌ NOT GROUPED - No valid email address found!")
            
            if final_sender_email:
                sender_groups[final_sender_email].append(invoice)
        
        logger.info("=" * 60)
        
        logger.info(f"Grouped invoices into {len(sender_groups)} unique sender groups")
        if emails_without_valid_address > 0:
            logger.warning(f"{emails_without_valid_address} invoices could not be grouped due to missing/invalid email addresses")
        
        # Log summary of sender groups
        if sender_groups:
            logger.info(f"Sender groups summary:")
            for email, invoices in list(sender_groups.items())[:10]:  # Log first 10 groups
                logger.info(f"  {email}: {len(invoices)} invoice(s)")
            if len(sender_groups) > 10:
                logger.info(f"  ... and {len(sender_groups) - 10} more sender groups")

        if email_enabled:
            # Send J&J consolidated email
            if jnj_invoices:
                jnj_excel_content = generate_excel(jnj_invoices, s3_links, "J&J Invoice Report")
                jnj_filename = f"jnj_invoice_report_{date_range_str}.xlsx"
                
                # Calculate J&J statistics
                jnj_status_counts = {"Success": 0, "Failed - API error": 0, "Failed": 0, "Failed-Unclassified Document": 0, "Rejected by system": 0, "Other": 0}
                for invoice in jnj_invoices:
                    combined_status = combine_status_values(invoice.get("api_status", ""), invoice.get("status", ""), invoice.get("field_status", ""), invoice.get("missing_critical_field", 0))
                    if combined_status in jnj_status_counts:
                        jnj_status_counts[combined_status] += 1
                    else:
                        jnj_status_counts["Other"] += 1
                
                jnj_success_count = jnj_status_counts["Success"]
                jnj_rejected_count = jnj_status_counts["Rejected by system"]
                jnj_failed_count = jnj_status_counts["Failed - API error"] + jnj_status_counts["Failed"] + jnj_status_counts["Failed-Unclassified Document"]
                
                jnj_subject = f"Invoice Ingestion Report - J&J - {twenty_four_hours_ago_ist.strftime('%Y-%m-%d %H:%M IST')} to {now_ist.strftime('%Y-%m-%d %H:%M IST')}"
                jnj_body = f"J&J consolidated invoice report"
                
                jnj_email_sent = send_client_email(
                    JNJ_RECIPIENTS,
                    jnj_subject,
                    jnj_body,
                    jnj_excel_content,
                    jnj_filename,
                    smtp_username,
                    smtp_password,
                    len(jnj_invoices),
                    jnj_success_count,
                    jnj_rejected_count,
                    jnj_failed_count,
                    FROM_EMAIL_JNJ,
                    twenty_four_hours_ago_ist,
                    now_ist,
                    "J&J"
                )
                
                if jnj_email_sent:
                    email_results["jnj_email_sent"] = 1
                    logger.info(f"J&J consolidated email sent to {', '.join(JNJ_RECIPIENTS)}")
                else:
                    email_results["jnj_email_failed"] = 1
                    logger.warning(f"Failed to send J&J consolidated email to {', '.join(JNJ_RECIPIENTS)}")
                
                # Add a delay between consolidated and individual emails to avoid email client deduplication
                # 5 minute delay to ensure both emails are sent reliably
                delay_seconds = 300  # 5 minutes
                logger.info(f"Waiting {delay_seconds} seconds (5 minutes) before sending individual emails to avoid email client deduplication and ensure reliable delivery...")
                time.sleep(delay_seconds)
            
            # Send daily summary emails to individual senders
            logger.info("=" * 60)
            logger.info("Sending daily summary emails to individual senders")
            logger.info("=" * 60)
            
            logger.info(f"Found {len(sender_groups)} unique senders to send daily summary emails")
            if len(sender_groups) == 0:
                logger.warning(f"⚠️ No sender groups found - {len(results)} invoices processed but no valid email addresses found")
                logger.warning(f"Debug: Checking invoice email fields...")
                for invoice in results[:5]:  # Log first 5 invoices for debugging
                    logger.warning(f"  Invoice {invoice.get('invoice_number', 'N/A')}: to_email='{invoice.get('to_email', '')}', sender_email='{invoice.get('sender_email', '')}'")
            
            # Track sender email results
            sender_email_results = {
                "sent": 0,
                "failed": 0,
                "skipped": 0
            }
            
            # Send daily summary email to each sender
            sender_list = list(sender_groups.items())
            logger.info(f"Processing {len(sender_list)} sender groups for individual emails")
            logger.info("=" * 80)
            logger.info("SENDER EMAIL VALIDATION & PAYLOAD SUMMARY")
            logger.info("=" * 80)
            for idx, (sender_email, sender_invoices) in enumerate(sender_list, 1):
                sample_invoice_numbers = [inv.get("invoice_number", "N/A") for inv in sender_invoices[:5]]
                logger.info(
                    f"[{idx}/{len(sender_list)}] Sender: {sender_email or 'N/A'} | "
                    f"Invoice count: {len(sender_invoices)} | "
                    f"Sample invoices: {sample_invoice_numbers}"
                )
            logger.info("=" * 80)
            
            for idx, (sender_email, sender_invoices) in enumerate(sender_list):
                try:
                    # Skip if sender email is invalid
                    if not sender_email or sender_email.strip() == "":
                        sender_email_results["skipped"] += 1
                        logger.warning(f"[{idx+1}/{len(sender_list)}] Skipping sender with empty email address")
                        continue
                    
                    # Note: Internal recipients (J&J team members) will receive BOTH:
                    # 1. Individual sender summary email (their own invoices)
                    # 2. J&J consolidated email (all invoices)
                    # So we do NOT skip them from individual sender emails
                    
                    # Skip placeholder/unknown emails
                    if sender_email.lower() in ["unknown@example.com", "unknown", ""]:
                        sender_email_results["skipped"] += 1
                        logger.info(f"[{idx+1}/{len(sender_list)}] Skipping placeholder email: {sender_email}")
                        continue
                    
                    # Validate email format before proceeding
                    if "@" not in sender_email:
                        sender_email_results["skipped"] += 1
                        logger.warning(f"[{idx+1}/{len(sender_list)}] Skipping invalid email format: {sender_email}")
                        continue
                    
                    logger.info(f"[{idx+1}/{len(sender_list)}] Preparing daily summary email for sender: {sender_email} ({len(sender_invoices)} invoices)")
                    
                    # Validate that we have invoices
                    if not sender_invoices or len(sender_invoices) == 0:
                        sender_email_results["skipped"] += 1
                        logger.warning(f"[{idx+1}/{len(sender_list)}] Skipping {sender_email} - no invoices to send")
                        continue
                    
                    # Generate Excel file for this sender's invoices
                    logger.info(f"Generating Excel file for {sender_email}...")
                    sender_excel_content = generate_excel(sender_invoices, s3_links, f"Daily Invoice Summary - {sender_email}")
                    
                    # Validate Excel content was generated successfully
                    if not sender_excel_content:
                        sender_email_results["failed"] += 1
                        logger.error(f"[{idx+1}/{len(sender_list)}] Failed to generate Excel file for {sender_email}")
                        continue
                    
                    logger.info(f"Excel file generated successfully for {sender_email} ({len(sender_excel_content)} bytes)")
                    sender_filename = f"daily_invoice_summary_{sender_email.replace('@', '_at_').replace('.', '_')}_{date_range_str}.xlsx"
                    
                    # Create subject for sender email - matching overall email format
                    sender_subject = f"Your Personal Invoice Report ({len(sender_invoices)} Invoices) - {twenty_four_hours_ago_ist.strftime('%Y-%m-%d %H:%M IST')} to {now_ist.strftime('%Y-%m-%d %H:%M IST')}"
                    logger.info(
                        f"[{idx+1}/{len(sender_list)}] Email payload snapshot -> "
                        f"To: {sender_email}, Subject: {sender_subject}, "
                        f"Invoices: {[inv.get('invoice_number', 'N/A') for inv in sender_invoices]}"
                    )
                    
                    # Send daily summary email to sender
                    logger.info(f"Attempting to send email to {sender_email}...")
                    sender_email_sent = send_sender_daily_summary_email(
                        sender_email,
                        sender_subject,
                        sender_excel_content,
                        sender_filename,
                        smtp_username,
                        smtp_password,
                        sender_invoices,
                        s3_links,
                        FROM_EMAIL_JNJ,
                        twenty_four_hours_ago_ist,
                        now_ist
                    )
                    
                    if sender_email_sent:
                        sender_email_results["sent"] += 1
                        logger.info(f"✅ Daily summary email sent successfully to {sender_email}")
                    else:
                        sender_email_results["failed"] += 1
                        logger.warning(f"❌ Failed to send daily summary email to {sender_email}")
                    
                    # Add a small delay between emails to avoid rate limiting (except for the last email)
                    # This helps prevent emails from being filtered as spam
                    if idx < len(sender_list) - 1:
                        delay_seconds = 2  # 2 second delay between emails
                        logger.info(f"Waiting {delay_seconds} seconds before sending next email to avoid rate limiting...")
                        time.sleep(delay_seconds)
                        
                except Exception as e:
                    sender_email_results["failed"] += 1
                    logger.error(f"Error sending daily summary email to {sender_email}: {str(e)}", exc_info=True)
            
            logger.info(f"Daily summary email results: {sender_email_results['sent']} sent, {sender_email_results['failed']} failed, {sender_email_results['skipped']} skipped")
            
            email_results["sender_summary"] = sender_email_results

        # Prepare response data
        uploaded_files = [{
            "sender_email": "ALL_SENDERS",
            "record_count": len(results),
            "s3_location": f"s3://{OUTPUT_BUCKET}/{all_s3_key}",
            "download_url": all_presigned_url,
            "email_sent": email_results["jnj_email_sent"] > 0
        }]

        logger.info(f"Consolidated CSV report saved to s3://{OUTPUT_BUCKET}/{all_s3_key}")
        logger.info(f"Email summary: J&J email sent: {email_results.get('jnj_email_sent', 0)}")
        if "sender_summary" in email_results:
            sender_summary = email_results["sender_summary"]
            logger.info(f"Sender daily summaries: {sender_summary.get('sent', 0)} sent, {sender_summary.get('failed', 0)} failed, {sender_summary.get('skipped', 0)} skipped")

        return {
            "statusCode": 200,
            "body": json.dumps(
                {
                    "message": f"Generated {len(uploaded_files)} CSV reports successfully",
                    "total_count": len(results),
                    "date_range": f"{twenty_four_hours_ago_ist.strftime('%Y-%m-%d %H:%M IST')} to {now_ist.strftime('%Y-%m-%d %H:%M IST')}",
                    "files": uploaded_files,
                    "email_summary": email_results,
                    "jnj_invoice_count": len(jnj_invoices),
                    "unique_senders_count": len(sender_groups) if email_enabled else 0
                },
                cls=DecimalEncoder
            ),
        }

    except Exception as e:
        logger.error(f"Error generating invoice CSV report: {str(e)}", exc_info=True)
        return {"statusCode": 500, "body": json.dumps({"error": str(e)})}
