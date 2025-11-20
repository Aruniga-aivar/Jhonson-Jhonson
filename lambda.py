import json
import logging
import uuid
import boto3
import os
import re
from datetime import datetime
from email import policy
from email.parser import BytesParser
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.message import EmailMessage
import requests
from utils import (
    calling_bedrock_llm,
    extract_email_from_s3,
    get_secret,
    send_email,
    send_status_email,
    generate_job_id,
    create_dynamodb_entry,
    update_job_status,
    get_job_status,
    create_attachment_entry,
    update_attachment_status
)
import tempfile
import importlib.util


PROMPT_TEMPLATE_BUCKET = os.environ.get('PROMPT_TEMPLATE_BUCKET')
PROMPT_TEMPLATE_KEY = os.environ.get('PROMPT_TEMPLATE_KEY')

# Prompt Template Location
bucket = f"{PROMPT_TEMPLATE_BUCKET}"
key = f"{PROMPT_TEMPLATE_KEY}"

def load_prompt_template_from_s3(bucket_name, key):
    s3 = boto3.client('s3')

    with tempfile.TemporaryDirectory() as tmpdir:
        local_path = os.path.join(tmpdir, "prompt_template.py")
        s3.download_file(bucket_name, key, local_path)

        spec = importlib.util.spec_from_file_location("prompt_template", local_path)
        prompt_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompt_module)

        return prompt_module.classification_prompt

classification_prompt = load_prompt_template_from_s3(bucket, key)


# Constants from environment variables
SES_S3_BUCKET = os.environ['SES_S3_BUCKET']
DESTINATION_BUCKET = os.environ['DESTINATION_BUCKET']
SMTP_SERVER = os.environ['SMTP_SERVER']
SMTP_PORT = int(os.environ['SMTP_PORT'])
FROM_EMAIL = os.environ['FROM_EMAIL']
INFERENCE_PROFILE_ARN = os.environ['INFERENCE_PROFILE_ARN']
DYNAMODB_TABLE = os.environ['DYNAMODB_TABLE'].strip()
INVOICE_PROCESSOR_ARN = os.environ.get('INVOICE_PROCESSOR_ARN')  # Keep for backward compatibility
SQS_QUEUE_URL = os.environ.get('SQS_QUEUE_URL')  # SQS queue URL for invoice processing
SMTP_SECRET_NAME = os.environ['SMTP_SECRET_NAME']

# AWS clients
s3 = boto3.client('s3')
bedrock_client = boto3.client('bedrock-runtime')
lambda_client = boto3.client('lambda')  # Keep for backward compatibility
sqs_client = boto3.client('sqs')
dynamodb_client = boto3.client('dynamodb')

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Get SMTP credentials
SMTP_CONFIG = get_secret(SMTP_SECRET_NAME)
SMTP_USERNAME = SMTP_CONFIG["SMTP_USERNAME"]
SMTP_PASSWORD = SMTP_CONFIG["SMTP_PASSWORD"]

# Authorized email addresses for invoice processing
AUTHORIZED_EMAIL_ADDRESSES = [
    'cynthia.santiago@craneww.com',
    'Lisa.Glover@craneww.com',
    'tommy.lu@craneww.com',
    'Nathan.Monica@magnointl.com',
    'vilasak.thebpanya@magnointl.com',
    'Cheryl.Santos@magnointl.com',
    'Gaetano.Parisi@am.kwe.com',
    'Jessica.Cuazitl@am.kwe.com',
    'Damian.Duda@am.kwe.com',
    'fredrick.hayde@am.kwe.com',
    'jay.sonani@am.kwe.com',
    'joanna.palentinos@am.kwe.com',
    'Mirelis.gonzelez@am.kwe.com',
    'Justin.Sisouvankham@am.kwe.com',
    'Hadiyah.Khan@am.kwe.com',
    'John.Foote@am.kwe.com',
    'Janelle.Daniels@am.kwe.com',
    'sadhanand.moorthy@pando.ai',
    'shashank.tiwari@pando.ai',
    'AHollbac@ITS.JNJ.com',
    'PPearson@ITS.JNJ.com', 
    "jeeva@pando.ai",
    "CString3@ITS.JNJ.com"
]

def extract_email_address(email_string):
    """
    Extract just the email address from email header format.
    Handles formats like:
    - "Name" <email@domain.com>
    - email@domain.com
    - <email@domain.com>
    
    Args:
        email_string: Email string that may contain display name and/or angle brackets
        
    Returns:
        str: Just the email address (e.g., email@domain.com)
    """
    if not email_string:
        return ""
    
    email_string = email_string.strip()
    
    # Try to extract email from angle brackets first (most common format)
    # Pattern to match email address in angle brackets: <email@domain.com>
    bracket_match = re.search(r'<([^>]+)>', email_string)
    if bracket_match:
        return bracket_match.group(1).strip()
    
    # Pattern to match email address directly (if no brackets)
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, email_string)
    if email_match:
        return email_match.group(0).strip()
    
    # If no pattern matches, return the original string (fallback)
    return email_string

def is_authorized_sender(email_address):
    """
    Check if the email address is in the authorized list.
    Uses exact match (case-insensitive) for email validation.
    
    Args:
        email_address: The email address to check (case-insensitive exact match)
        
    Returns:
        bool: True if authorized, False otherwise
    """
    if not email_address:
        return False
    
    # Extract just the email address from the string (in case it has display name/formatting)
    clean_email = extract_email_address(email_address)
    
    # Normalize email to lowercase for comparison
    email_lower = clean_email.lower().strip()
    
    # Convert authorized emails to lowercase for case-insensitive comparison
    authorized_emails_lower = [email.lower().strip() for email in AUTHORIZED_EMAIL_ADDRESSES]
    
    # Check if sender email matches any authorized email
    # Use exact match instead of substring match for better security
    is_authorized = email_lower in authorized_emails_lower
    
    return is_authorized

def send_rejection_email(
    smtp_server, smtp_port, smtp_username, smtp_password,
    from_email, to_email, subject, email_id, message_id,
    original_sender=None, original_date=None, original_subject=None, original_body=None
):
    """
    Send a rejection email to unauthorized sender.
    This email does NOT include the "attachments picked up for processing" message.
    """
    import smtplib
    import ssl
    
    try:
        logger.info(f"Sending rejection email to unauthorized sender: {to_email}")
        
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
                <h2 style="color: #dc3545; margin-top: 0;">You are not authorized to send invoices</h2>
                <p><strong>Status: REJECTED</strong></p>
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
            logger.info(f"SMTP login successful, sending rejection email to: {to_email}")
            server.send_message(msg)
            logger.info("Rejection email sent successfully!")
            
        return True
    except Exception as e:
        logger.error(f"Error sending rejection email: {str(e)}", exc_info=True)
        raise

def parse_s3_path(s3_uri):
    if not s3_uri.startswith("s3://"):
        raise ValueError(f"Invalid S3 URI: {s3_uri}")
    without_scheme = s3_uri[5:]
    if "/" not in without_scheme:
        raise ValueError(f"Invalid S3 URI (missing key): {s3_uri}")
    bucket_name, key = without_scheme.split('/', 1)
    return bucket_name, key

def copy_attachments_to_email_folder(s3_client, attachments, email_id, destination_bucket, logger):
    """
    Copy all email attachments to the destination bucket under email_attachments/{email_id}/
    
    Args:
        s3_client: Boto3 S3 client
        attachments: List of attachment dictionaries with 's3_path' and 'filename'
        email_id: Unique email identifier
        destination_bucket: Destination S3 bucket name
        logger: Logger instance
    
    Returns:
        List of new S3 paths for the copied attachments
    """
    copied_attachments = []
    
    for attachment in attachments:
        try:
            src_bucket, src_key = parse_s3_path(attachment['s3_path'])
            filename = attachment['filename']
            
            # Create destination key: email_attachments/{email_id}/{filename}
            safe_email_id = email_id.replace('/', '_')
            dest_key = f"email_attachments/{safe_email_id}/{filename}"
            
            # Copy the file to the destination
            copy_source = {'Bucket': src_bucket, 'Key': src_key}
            s3_client.copy_object(
                CopySource=copy_source,
                Bucket=destination_bucket,
                Key=dest_key
            )
            
            new_s3_path = f"s3://{destination_bucket}/{dest_key}"
            copied_attachments.append(new_s3_path)
            
            logger.info(f"Copied attachment '{filename}' to {dest_key}")
            
        except Exception as e:
            logger.error(f"Error copying attachment '{attachment.get('filename', 'unknown')}': {str(e)}")
            # Continue with other attachments even if one fails
    
    return copied_attachments


def send_invoice_to_queue(invoice_processor_input, logger):
    """
    Send invoice processing request to SQS queue, Batch, or Lambda (fallback).
    
    Priority:
    1. SQS Queue (if SQS_QUEUE_URL configured) - triggers Batch via batch_trigger_lambda
    2. Direct Batch submission (if BATCH_JOB_QUEUE configured)
    3. Direct Lambda invocation (if INVOICE_PROCESSOR_ARN configured) - fallback
    
    Args:
        invoice_processor_input: Dictionary with invoice processing parameters
        logger: Logger instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Check if Batch is configured (direct submission)
        BATCH_JOB_QUEUE = os.environ.get('BATCH_JOB_QUEUE')
        BATCH_JOB_DEFINITION = os.environ.get('BATCH_JOB_DEFINITION')
        
        # Prefer SQS if queue URL is configured (triggers Batch via batch_trigger_lambda)
        if SQS_QUEUE_URL:
            logger.info(f"Sending invoice processing request to SQS queue: {SQS_QUEUE_URL}")
            logger.info("   (SQS will trigger Batch job via batch_trigger_lambda)")
            
            # Send message to SQS
            response = sqs_client.send_message(
                QueueUrl=SQS_QUEUE_URL,
                MessageBody=json.dumps(invoice_processor_input)
            )
            
            logger.info(f"✅ Successfully sent message to SQS queue")
            logger.info(f"   MessageId: {response.get('MessageId')}")
            logger.info(f"   Invoice processor input: {json.dumps(invoice_processor_input)}")
            return True
        
        # Option 2: Direct Batch submission (if configured)
        elif BATCH_JOB_QUEUE and BATCH_JOB_DEFINITION:
            logger.info(f"Sending invoice processing request directly to AWS Batch")
            logger.info(f"   Job Queue: {BATCH_JOB_QUEUE}")
            logger.info(f"   Job Definition: {BATCH_JOB_DEFINITION}")
            
            try:
                import uuid
                batch_client = boto3.client('batch')
                
                job_name = f"invoice-processor-{uuid.uuid4().hex[:8]}"
                job_params_json = json.dumps(invoice_processor_input)
                
                submit_job_request = {
                    'jobName': job_name,
                    'jobQueue': BATCH_JOB_QUEUE,
                    'jobDefinition': BATCH_JOB_DEFINITION,
                    'parameters': {
                        'job_parameters': job_params_json
                    }
                }
                
                # Add IAM role if specified
                BATCH_JOB_ROLE_ARN = os.environ.get('BATCH_JOB_ROLE_ARN')
                if BATCH_JOB_ROLE_ARN:
                    submit_job_request['jobRoleArn'] = BATCH_JOB_ROLE_ARN
                
                response = batch_client.submit_job(**submit_job_request)
                
                logger.info(f"✅ Successfully submitted Batch job")
                logger.info(f"   Job ID: {response['jobId']}")
                logger.info(f"   Job Name: {response['jobName']}")
                return True
                
            except Exception as batch_error:
                logger.error(f"Failed to submit Batch job: {str(batch_error)}")
                # Fall through to Lambda fallback
        
        # Option 3: Fallback to direct Lambda invocation
        if INVOICE_PROCESSOR_ARN:
            logger.warning("SQS_QUEUE_URL and Batch not configured, falling back to direct Lambda invocation")
            lambda_client.invoke(
                FunctionName=INVOICE_PROCESSOR_ARN,
                InvocationType='Event',
                Payload=json.dumps(invoice_processor_input)
            )
            logger.info(f"Asynchronously invoked invoice processor via Lambda: {INVOICE_PROCESSOR_ARN}")
            logger.info(f"Invoice processor input: {json.dumps(invoice_processor_input)}")
            return True
        else:
            logger.error("Neither SQS_QUEUE_URL, Batch, nor INVOICE_PROCESSOR_ARN is configured")
            return False
            
    except Exception as e:
        logger.error(f"Failed to send invoice processing request: {str(e)}", exc_info=True)
        return False

def merge_pdfs_and_upload(s3_client, pdf_s3_paths, destination_bucket, output_key, logger):
    try:
        import fitz  # PyMuPDF
    except Exception as import_error:
        logger.error("PyMuPDF (fitz) not available. Please include it via a Lambda layer. Error: %s", str(import_error))
        raise

    with tempfile.TemporaryDirectory() as tmpdir:
        local_paths = []
        for idx, s3_uri in enumerate(pdf_s3_paths):
            src_bucket, src_key = parse_s3_path(s3_uri)
            local_path = os.path.join(tmpdir, f"input_{idx}.pdf")
            s3_client.download_file(src_bucket, src_key, local_path)
            local_paths.append(local_path)

        merged_local_path = os.path.join(tmpdir, "merged.pdf")

        # Merge using PyMuPDF
        merged_doc = fitz.open()
        try:
            for path in local_paths:
                with fitz.open(path) as src:
                    merged_doc.insert_pdf(src)
            merged_doc.save(merged_local_path)
        finally:
            try:
                merged_doc.close()
            except Exception:
                pass

        s3_client.upload_file(merged_local_path, destination_bucket, output_key, ExtraArgs={"ContentType": "application/pdf"})
        return f"s3://{destination_bucket}/{output_key}"

def lambda_handler(event, context):
    logger.info("Received event: %s", json.dumps(event))

    # Parse SNS S3 event
    key = event['Records'][0]['s3']['object']['key']
    logger.info("S3 object key: %s", key)

    logger.info("Dynamo Db : %s", DYNAMODB_TABLE)
    # Check if this email has already been processed
    try:
        # Use the S3 key as a unique identifier
        existing_job = dynamodb_client.query(
            TableName=DYNAMODB_TABLE,
            KeyConditionExpression='pk = :pk AND sk = :sk',
            ExpressionAttributeValues={
                ':pk': {'S': f"EMAIL#{key}"},
                ':sk': {'S': 'METADATA'}
            }
        )
        logger.info(existing_job)
        
        if existing_job.get('Items'):
            logger.info(f"Email {key} has already been processed. Skipping.")
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Email already processed',
                    'key': key
                })
            }
    except Exception as e:
        logger.error(f"Error checking for existing job: {str(e)}")
        # Continue processing even if check fails

    # Generate unique job ID
    email_id = key
    created_at = datetime.utcnow().isoformat()  # This is now just used for metadata

    email_data = extract_email_from_s3(s3, SES_S3_BUCKET, key, DESTINATION_BUCKET)
    logger.info("Parsed email content from S3")

    original_body = email_data['body']
    source = email_data['from']
    attachments = email_data['attachments']
    attachment_analysis = email_data['attachment_analysis']
    logger.info("Email from: %s | Subject: %s | Attachments: %d| query: %s", 
                source, email_data['subject'], len(attachments), email_data['query'])

    # Check if sender is authorized
    if not is_authorized_sender(source):
        logger.warning(f"Unauthorized sender detected: {source}")
        
        # Create DynamoDB entry with REJECTED status
        try:
            create_dynamodb_entry(
                dynamodb_client,
                DYNAMODB_TABLE,
                email_id,
                'REJECTED',
                source,
                created_at,
                metadata={
                    'subject': email_data['subject'],
                    'sender': source,
                    'recipient': email_data.get('to', ''),
                    'received_time': email_data.get('date', created_at),
                    'rejection_reason': 'Unauthorized sender'
                },
                attachment_analysis={
                    'total_count': len(attachments),
                    'pdf_count': attachment_analysis['pdf_count'],
                    'excel_count': attachment_analysis['excel_count'],
                    'other_count': len(attachment_analysis['attachment_types']) - (attachment_analysis['pdf_count'] + attachment_analysis['excel_count'])
                }
            )
        except Exception as e:
            logger.error(f"Error creating DynamoDB entry for rejected email: {str(e)}")
        
        # Send rejection email to unauthorized sender (without "attachments picked up" message)
        try:
            send_rejection_email(
                SMTP_SERVER, SMTP_PORT,
                SMTP_USERNAME, SMTP_PASSWORD,
                FROM_EMAIL, source,
                email_data['subject'],
                email_id,
                email_data['message_id'],
                original_sender=source,
                original_date=email_data.get('date', created_at),
                original_subject=email_data['subject'],
                original_body=original_body
            )
            logger.info(f"Rejection email sent to unauthorized sender: {source}")
        except Exception as e:
            logger.error(f"Error sending rejection email: {str(e)}")
        
        return {
            'statusCode': 200,
            'body': json.dumps({
                'message': 'Email rejected - unauthorized sender',
                'email_id': email_id,
                'sender': source,
                'reason': 'Sender not in authorized list'
            })
        }

    # Note: Attachment copying only happens for authorized senders
    # Rejected emails return early above, so attachments are not copied for processing
    # Copy all attachments to email_attachments folder in destination bucket
    if attachments:
        logger.info("Copying all attachments to email_attachments folder...")
        copied_attachments = copy_attachments_to_email_folder(
            s3, 
            attachments, 
            email_id, 
            DESTINATION_BUCKET, 
            logger
        )
        logger.info(f"Copied {len(copied_attachments)} attachment(s) to email_attachments folder")

    # Create initial DynamoDB entry
    create_dynamodb_entry(
        dynamodb_client,
        DYNAMODB_TABLE,
        email_id,
        'RECEIVED',
        source,
        created_at,  # This is now just used for metadata
        metadata={
            'subject': email_data['subject'],
            'sender': source,
            'recipient': email_data.get('to', ''),
            'received_time': email_data.get('date', created_at)
        },
        attachment_analysis={
            'total_count': len(attachments),
            'pdf_count': attachment_analysis['pdf_count'],
            'excel_count': attachment_analysis['excel_count'],
            'other_count': len(attachment_analysis['attachment_types']) - (attachment_analysis['pdf_count'] + attachment_analysis['excel_count'])
        }
    )

    # Check if PDF files are present
    has_pdf = any(att['filename'].lower().endswith('.pdf') for att in attachments)
    
    logger.info(f"Attachment analysis - has_pdf: {has_pdf}")
    logger.info(f"Attachments: {[att['filename'] for att in attachments]}")

    # Initialize status tracking variables
    status_message_id = None
    status_email_body = None
    status_email_subject = None
    status_email_sender = None
    status_email_date = None

    # Early return for no attachments
    if not attachments:
        logger.info("No attachments found - sending UNCLASSIFIED status")
        try:
            update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'UNCLASSIFIED')
            status_message_id, status_email_body, status_email_subject, status_email_sender, status_email_date = send_status_email(
                SMTP_SERVER, SMTP_PORT,
                SMTP_USERNAME, SMTP_PASSWORD,
                FROM_EMAIL, source,
                email_data['subject'],
                email_data['attachments'],
                email_id,
                'UNCLASSIFIED',
                email_data['message_id'],
                original_sender=source,
                original_date=email_data.get('date', created_at),
                original_subject=email_data['subject'],
                original_body=original_body
            )
            return {
                'statusCode': 200,
                'body': json.dumps({
                    'message': 'Email marked as unclassified due to no attachments',
                    'email_id': email_id
                })
            }
        except Exception as e:
            logger.error(f"Error sending UNCLASSIFIED status email: {str(e)}")
            return {
                'statusCode': 500,
                'body': json.dumps({
                    'message': 'Failed to send unclassified status email',
                    'error': str(e),
                    'email_id': email_id
                })
            }

    # NOTE: Do NOT send RECEIVED status email here
    # The "received" email will be sent later in the invoice processor (lambda_function.py)
    # after carrier identification and authorization check passes
    # This ensures we only send "received" email when:
    # 1. User is valid (already checked above)
    # 2. Carrier matches sender email domain (checked in J_J.py)
    logger.info("Skipping RECEIVED status email - will be sent after carrier authorization check in invoice processor")
    
    # Store email details for later use in invoice processor
    # Note: status_email_sender should be the recipient (to) from the original email, not the sender
    # This is the email address that received the original email (e.g., jnjna.demo@pibypando.ai)
    status_message_id = None
    status_email_body = original_body
    status_email_subject = email_data['subject']
    
    # Extract just the email address from the 'to' field (may contain display name and angle brackets)
    to_email_raw = email_data.get('to', source)
    status_email_sender = extract_email_address(to_email_raw)  # Extract just email address from format like "Name" <email@domain.com>
    status_email_date = email_data.get('date', created_at)
    
    # Log the email details for debugging
    logger.info(f"Email details - From: {source}, To (raw): {to_email_raw}, To (extracted): {status_email_sender}")

    # Step 1: Determine invoice status
    attachment_analysis = email_data['attachment_analysis']
    logger.info("Processing attachments for classification...")

    # Step 2: Construct attachment status for classifier prompt
    if not attachments:
        attachment_status = 'No attachments found in the email'
    else:
        parts = []
        if attachment_analysis['has_pdf']:
            parts.append(f"PDF file(s) detected: {attachment_analysis['pdf_count']}")
        if len(attachment_analysis['attachment_types']) > attachment_analysis['pdf_count']:
            parts.append("Other file types detected")
        attachment_status = ' ; '.join(parts)
    logger.info("Attachment status: %s", attachment_status)

    # Step 3: Prepare classification prompt
    formatted_prompt = (classification_prompt
                        .replace('{subject}', email_data['subject'])
                        .replace('{body}', original_body)
                        .replace('{invoice_status}', attachment_status)
                        .replace('{attachment_analysis}', json.dumps(attachment_analysis)))
    logger.info("Formatted classification prompt")

    # Step 4: Run classification
    try:
        result = calling_bedrock_llm(
            bedrock_client, INFERENCE_PROFILE_ARN,
            prompt=formatted_prompt, max_attempts=3, retry_delay=1
        )
        logger.info("LLM classification result: %s", result)

        # Override classification if attachment analysis provides strong indicators
        if not result or not result.get('classification'):
            if attachment_analysis['has_pdf']:
                result = {'classification': 'Invoice'}
                logger.info("Using attachment-based classification: Invoice")
            else:
                result = {'classification': 'Unclear'}
                logger.info("Using attachment-based classification: Unclear")

        # Update status to CLASSIFIED
        update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'CLASSIFIED', result.get('classification'))
    except Exception as e:
        logger.error(f"Error during classification: {str(e)}")
        # Fallback to attachment-based classification
        if attachment_analysis['has_pdf']:
            result = {'classification': 'Invoice'}
            logger.info("Using attachment-based classification after error: Invoice")
        else:
            result = {'classification': 'Unclear'}
            logger.info("Using attachment-based classification after error: Unclear")
        # Update status to CLASSIFIED with fallback classification
        update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'CLASSIFIED', result.get('classification'))
        
        # Send notification about fallback classification
        send_status_email(
            SMTP_SERVER, SMTP_PORT,
            SMTP_USERNAME, SMTP_PASSWORD,
            FROM_EMAIL, source,
            email_data['subject'],
            email_data['attachments'],
            email_id,
            'CLASSIFIED - Using attachment-based classification due to LLM error',
            email_data['message_id']
        )

    # Process Invoice
    if result and result.get('classification') == 'Invoice':
        logger.info("Classification: Invoice")
        if not attachment_analysis['has_pdf']:
            update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'FAILED')
            send_status_email(
                SMTP_SERVER, SMTP_PORT,
                SMTP_USERNAME, SMTP_PASSWORD,
                FROM_EMAIL, source,
                email_data['subject'],
                email_data['attachments'],
                email_id,
                'FAILED - No PDF attachments found',
                email_data['message_id']
            )
        else:
            # If multiple PDF attachments, merge and process as a single PDF
            pdf_attachments = [att for att in attachments if att['filename'].lower().endswith('.pdf')]
            if len(pdf_attachments) > 1:
                try:
                    attachment_id = str(uuid.uuid4())
                    safe_email_id = email_id.replace('/', '_')
                    merged_key = f"invoice/input/merged/{safe_email_id}-{attachment_id}.pdf"
                    merged_uri = merge_pdfs_and_upload(
                        s3,
                        [att['s3_path'] for att in pdf_attachments],
                        DESTINATION_BUCKET,
                        merged_key,
                        logger
                    )

                    # Create a single attachment entry for the merged PDF
                    create_attachment_entry(
                        dynamodb_client,
                        DYNAMODB_TABLE,
                        email_id,
                        attachment_id,
                        'merged.pdf',
                        'PDF',
                        merged_uri,
                        'Invoice'
                    )

                    invoice_processor_input = {
                        'input_bucket': DESTINATION_BUCKET,
                        'input_key': merged_key,
                        'output_bucket': DESTINATION_BUCKET,
                        'output_prefix': 'invoice/output/',
                        'email_details': {
                            'to': source,
                            'subject': email_data['subject'],
                            'message_id': email_data['message_id'],
                            'original_body': original_body,
                            'filename': 'merged.pdf',
                            'status_message_id': status_message_id,
                            'status_email_body': status_email_body,
                            'status_email_subject': status_email_subject,
                            'status_email_sender': status_email_sender,
                            'status_email_date': status_email_date
                        },
                        'job_id': email_id,
                        'attachment_id': attachment_id,
                        'email_attachments': f"s3://{DESTINATION_BUCKET}/email_attachments/{safe_email_id}"
                    }

                    update_attachment_status(
                        dynamodb_client,
                        DYNAMODB_TABLE,
                        email_id,
                        attachment_id,
                        'PROCESSING'
                    )

                    try:
                        success = send_invoice_to_queue(invoice_processor_input, logger)
                        if not success:
                            raise Exception("Failed to send invoice processing request to queue/Lambda")
                        logger.info("Successfully sent merged.pdf to invoice processor queue")
                    except Exception as e:
                        logger.error("Failed to send invoice processing request for merged.pdf: %s", str(e))
                        update_attachment_status(
                            dynamodb_client,
                            DYNAMODB_TABLE,
                            email_id,
                            attachment_id,
                            'FAILED',
                            error={
                                'message': str(e),
                                'error_code': 'PROCESSOR_INVOCATION_FAILED'
                            }
                        )
                        update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'FAILED')
                        send_status_email(
                            SMTP_SERVER, SMTP_PORT,
                            SMTP_USERNAME, SMTP_PASSWORD,
                            FROM_EMAIL, source,
                            email_data['subject'],
                            email_data['attachments'],
                            email_id,
                            'FAILED - Error processing merged invoice',
                            email_data['message_id']
                        )
                        raise
                except Exception as e:
                    logger.error("Failed to process merged PDF: %s", str(e))
                    update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'FAILED')
                    send_status_email(
                        SMTP_SERVER, SMTP_PORT,
                        SMTP_USERNAME, SMTP_PASSWORD,
                        FROM_EMAIL, source,
                        email_data['subject'],
                        email_data['attachments'],
                        email_id,
                        'FAILED - Error merging invoice PDFs',
                        email_data['message_id']
                    )
            else:
                # Single PDF (or none) - keep existing per-attachment behavior
                # Create safe_email_id for this scope
                safe_email_id = email_id.replace('/', '_')
                
                for att in attachments:
                    if not att['filename'].lower().endswith('.pdf'):
                        continue
                    try:
                        attachment_id = str(uuid.uuid4())
                        create_attachment_entry(
                            dynamodb_client,
                            DYNAMODB_TABLE,
                            email_id,
                            attachment_id,
                            att['filename'],
                            'PDF',
                            att['s3_path'],
                            'Invoice'
                        )

                        s3_path = att['s3_path']
                        input_key = s3_path.replace(f"s3://{DESTINATION_BUCKET}/", "")

                        invoice_processor_input = {
                            'input_bucket': DESTINATION_BUCKET,
                            'input_key': input_key,
                            'output_bucket': DESTINATION_BUCKET,
                            'output_prefix': 'invoice/output/',
                            'email_details': {
                                'to': source,
                                'subject': email_data['subject'],
                                'message_id': email_data['message_id'],
                                'original_body': original_body,
                                'filename': att['filename'],
                                'status_message_id': status_message_id,
                                'status_email_body': status_email_body,
                                'status_email_subject': status_email_subject,
                                'status_email_sender': status_email_sender,
                                'status_email_date': status_email_date
                            },
                            'job_id': email_id,
                            'attachment_id': attachment_id,
                            'email_attachments': f"s3://{DESTINATION_BUCKET}/email_attachments/{safe_email_id}"
                        }

                        update_attachment_status(
                            dynamodb_client,
                            DYNAMODB_TABLE,
                            email_id,
                            attachment_id,
                            'PROCESSING'
                        )

                        try:
                            success = send_invoice_to_queue(invoice_processor_input, logger)
                            if not success:
                                raise Exception("Failed to send invoice processing request to queue/Lambda")
                            logger.info("Successfully sent %s to invoice processor queue", att['filename'])
                        except Exception as e:
                            logger.error("Failed to send invoice processing request for %s: %s", att['filename'], str(e))
                            update_attachment_status(
                                dynamodb_client,
                                DYNAMODB_TABLE,
                                email_id,
                                attachment_id,
                                'FAILED',
                                error={
                                    'message': str(e),
                                    'error_code': 'PROCESSOR_INVOCATION_FAILED'
                                }
                            )
                            update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'FAILED')
                            send_status_email(
                                SMTP_SERVER, SMTP_PORT,
                                SMTP_USERNAME, SMTP_PASSWORD,
                                FROM_EMAIL, source,
                                email_data['subject'],
                                email_data['attachments'],
                                email_id,
                                'FAILED - Error processing invoice',
                                email_data['message_id']
                            )
                            raise
                    except Exception as e:
                        logger.error("Failed to invoke invoice processor for %s: %s", att['filename'], str(e))
                        update_attachment_status(
                            dynamodb_client,
                            DYNAMODB_TABLE,
                            email_id,
                            attachment_id,
                            'FAILED',
                            error={
                                'message': str(e),
                                'error_code': 'PROCESSOR_INVOCATION_FAILED'
                            }
                        )
                        update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'FAILED')
                        send_status_email(
                            SMTP_SERVER, SMTP_PORT,
                            SMTP_USERNAME, SMTP_PASSWORD,
                            FROM_EMAIL, source,
                            email_data['subject'],
                            email_data['attachments'],
                            email_id,
                            'FAILED - Error processing invoice',
                            email_data['message_id']
                        )
    # Handle Unclear Classification
    else:
        logger.info("Classification: Unclear or Not Invoice")
        update_job_status(dynamodb_client, DYNAMODB_TABLE, email_id, 'UNCLASSIFIED')
        send_status_email(
            SMTP_SERVER, SMTP_PORT,
            SMTP_USERNAME, SMTP_PASSWORD,
            FROM_EMAIL, source,
            email_data['subject'],
            email_data['attachments'],
            email_id,
            'UNCLASSIFIED - This email does not appear to contain an invoice PDF',
            email_data['message_id']
        )

    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': 'Email processed successfully',
            'email_id': email_id,
            'classification': result.get('classification') if result else 'Unclear'
        })
    }