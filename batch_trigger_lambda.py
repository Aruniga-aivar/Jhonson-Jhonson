"""
AWS Lambda function to trigger Batch jobs from SQS messages.

This Lambda function:
1. Reads messages from SQS queue
2. Submits AWS Batch jobs for each message
3. Handles retries and error cases

Environment Variables Required:
- BATCH_JOB_QUEUE: AWS Batch job queue name
- BATCH_JOB_DEFINITION: AWS Batch job definition name
- BATCH_JOB_ROLE_ARN: IAM role ARN for Batch jobs (optional, uses default if not set)
"""

import json
import logging
import os
import boto3
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# AWS clients
batch_client = boto3.client('batch')
sqs_client = boto3.client('sqs')

# Environment variables
BATCH_JOB_QUEUE = os.environ.get('BATCH_JOB_QUEUE')
BATCH_JOB_DEFINITION = os.environ.get('BATCH_JOB_DEFINITION')
BATCH_JOB_ROLE_ARN = os.environ.get('BATCH_JOB_ROLE_ARN')  # Optional

# Environment variables to pass to Batch job (from Lambda environment)
# These are the same variables used in lambda_function.py and J_J.py
PASS_THROUGH_ENV_VARS = [
    'AWS_REGION',
    'REGION',  # Alternative region variable
    'DYNAMODB_TABLE',
    'SMTP_SERVER',
    'SMTP_PORT',
    'FROM_EMAIL',
    'SMTP_SECRET_NAME',
    'API_ENDPOINT',
    'INTERNAL_TOKEN',
    'AUTHORIZATION_TOKEN',
    'BEDROCK_MODEL_ID',  # If set in Lambda environment
    'OUTPUT_BUCKET',  # S3 bucket for output files
    'S3_BUCKET',  # S3 bucket for templates and other resources
    'S3_BUCKET_KEY',  # S3 key for prompt template file
]

def submit_batch_job(job_params: Dict[str, Any], job_name_prefix: str = 'invoice-processor', sqs_record: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Submit a Batch job with the given parameters and environment variables.
    
    Args:
        job_params: Dictionary with invoice processing parameters
        job_name_prefix: Prefix for the Batch job name
        sqs_record: Optional SQS record for passing SQS event details
        
    Returns:
        Dictionary with Batch job submission response
    """
    try:
        # Create a unique job name
        import uuid
        job_name = f"{job_name_prefix}-{uuid.uuid4().hex[:8]}"
        
        # Prepare job parameters as JSON string for Batch (for backward compatibility)
        job_params_json = json.dumps(job_params)
        
        # Build environment variables list for containerOverrides
        env_vars = []
        
        # 1. Add all pass-through environment variables from Lambda environment
        for env_var in PASS_THROUGH_ENV_VARS:
            env_value = os.environ.get(env_var)
            if env_value is not None:
                env_vars.append({
                    'name': env_var,
                    'value': env_value
                })
                logger.debug(f"Adding environment variable: {env_var}")
        
        # 2. Add job parameters as environment variables
        env_vars.extend([
            {'name': 'input_bucket', 'value': job_params.get('input_bucket', '')},
            {'name': 'input_key', 'value': job_params.get('input_key', '')},
            {'name': 'output_bucket', 'value': job_params.get('output_bucket', job_params.get('input_bucket', ''))},
            {'name': 'output_prefix', 'value': job_params.get('output_prefix', 'invoice/output/')},
            {'name': 'job_id', 'value': str(job_params.get('job_id', ''))},
            {'name': 'attachment_id', 'value': str(job_params.get('attachment_id', ''))},
        ])
        
        # 3. Add optional job parameters as environment variables
        if job_params.get('carrier_name'):
            env_vars.append({'name': 'carrier_name', 'value': str(job_params['carrier_name'])})
        
        if job_params.get('s3_bucket'):
            env_vars.append({'name': 's3_bucket', 'value': str(job_params['s3_bucket'])})
        
        if job_params.get('excel_file_path'):
            env_vars.append({'name': 'excel_file_path', 'value': str(job_params['excel_file_path'])})
        
        if job_params.get('use_textract') is not None:
            env_vars.append({'name': 'use_textract', 'value': str(job_params['use_textract'])})
        
        if job_params.get('use_s3_clustering') is not None:
            env_vars.append({'name': 'use_s3_clustering', 'value': str(job_params['use_s3_clustering'])})
        
        if job_params.get('email_attachments'):
            env_vars.append({'name': 'email_attachments', 'value': str(job_params['email_attachments'])})
        
        # 4. Add email_details as JSON string if provided
        if job_params.get('email_details'):
            env_vars.append({
                'name': 'email_details',
                'value': json.dumps(job_params['email_details'])
            })
        
        # 5. Add SQS event details if provided
        if sqs_record:
            if sqs_record.get('messageId'):
                env_vars.append({'name': 'SQS_MESSAGE_ID', 'value': sqs_record['messageId']})
            if sqs_record.get('receiptHandle'):
                env_vars.append({'name': 'SQS_RECEIPT_HANDLE', 'value': sqs_record['receiptHandle']})
            if sqs_record.get('eventSourceARN'):
                env_vars.append({'name': 'SQS_EVENT_SOURCE_ARN', 'value': sqs_record['eventSourceARN']})
        
        # Build submit_job request with containerOverrides
        submit_job_request = {
            'jobName': job_name,
            'jobQueue': BATCH_JOB_QUEUE,
            'jobDefinition': BATCH_JOB_DEFINITION,
            'parameters': {
                'job_parameters': job_params_json  # Keep for backward compatibility
            },
            'containerOverrides': {
                'environment': env_vars
            }
        }
        
        # Add IAM role if specified
        if BATCH_JOB_ROLE_ARN:
            submit_job_request['jobRoleArn'] = BATCH_JOB_ROLE_ARN
        
        logger.info(f"Submitting Batch job: {job_name}")
        logger.info(f"Job parameters: {job_params_json[:500]}...")  # Log first 500 chars
        logger.info(f"Setting {len(env_vars)} environment variables in container")
        
        # Submit the job
        response = batch_client.submit_job(**submit_job_request)
        
        job_id = response['jobId']
        job_name_returned = response['jobName']
        
        logger.info(f"✅ Successfully submitted Batch job")
        logger.info(f"   Job ID: {job_id}")
        logger.info(f"   Job Name: {job_name_returned}")
        
        return {
            'success': True,
            'job_id': job_id,
            'job_name': job_name_returned,
            'response': response
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to submit Batch job: {str(e)}", exc_info=True)
        return {
            'success': False,
            'error': str(e)
        }

def process_sqs_message(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Process a single SQS message and submit a Batch job.
    
    Args:
        record: SQS record from Lambda event
        
    Returns:
        Dictionary with processing result
    """
    try:
        # Extract message body
        message_body = record.get('body', '{}')
        message_id = record.get('messageId', 'unknown')
        receipt_handle = record.get('receiptHandle')
        
        logger.info(f"Processing SQS message: {message_id}")
        
        # Parse message body (should be JSON with invoice processing parameters)
        try:
            job_params = json.loads(message_body)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse SQS message body as JSON: {e}")
            logger.error(f"Message body: {message_body[:500]}")
            return {
                'success': False,
                'message_id': message_id,
                'error': f'Invalid JSON in message body: {str(e)}'
            }
        
        # Validate required parameters
        required_params = ['input_key', 'input_bucket', 'job_id', 'attachment_id']
        missing_params = [param for param in required_params if param not in job_params]
        if missing_params:
            error_msg = f"Missing required parameters: {missing_params}"
            logger.error(f"{error_msg} in message {message_id}")
            return {
                'success': False,
                'message_id': message_id,
                'error': error_msg
            }
        
        # Submit Batch job with SQS record details
        result = submit_batch_job(job_params, job_name_prefix='invoice-processor', sqs_record=record)
        
        if result['success']:
            logger.info(f"✅ Successfully submitted Batch job for message {message_id}")
            return {
                'success': True,
                'message_id': message_id,
                'job_id': result['job_id'],
                'job_name': result['job_name']
            }
        else:
            logger.error(f"❌ Failed to submit Batch job for message {message_id}: {result.get('error')}")
            return {
                'success': False,
                'message_id': message_id,
                'error': result.get('error', 'Unknown error')
            }
            
    except Exception as e:
        logger.error(f"❌ Error processing SQS message: {str(e)}", exc_info=True)
        return {
            'success': False,
            'message_id': record.get('messageId', 'unknown'),
            'error': str(e)
        }

def lambda_handler(event, context):
    """
    AWS Lambda handler for triggering Batch jobs from SQS messages.
    
    Event format (SQS):
    {
        "Records": [
            {
                "messageId": "...",
                "body": "{...}",  # JSON string with invoice processing parameters
                "receiptHandle": "...",
                ...
            }
        ]
    }
    """
    logger.info("=" * 60)
    logger.info("BATCH TRIGGER LAMBDA - STARTING")
    logger.info("=" * 60)
    
    # Validate environment variables
    if not BATCH_JOB_QUEUE:
        error_msg = "BATCH_JOB_QUEUE environment variable is required"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    if not BATCH_JOB_DEFINITION:
        error_msg = "BATCH_JOB_DEFINITION environment variable is required"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    logger.info(f"Batch Job Queue: {BATCH_JOB_QUEUE}")
    logger.info(f"Batch Job Definition: {BATCH_JOB_DEFINITION}")
    
    # Process SQS records
    if 'Records' not in event:
        error_msg = "No 'Records' found in event"
        logger.error(error_msg)
        return {
            'statusCode': 400,
            'body': json.dumps({'error': error_msg})
        }
    
    records = event['Records']
    logger.info(f"Processing {len(records)} SQS message(s)")
    
    # Process all records (SQS trigger handles batching automatically)
    results = []
    successful = 0
    failed = 0
    
    for i, record in enumerate(records):
        logger.info(f"Processing record {i+1}/{len(records)}")
        result = process_sqs_message(record)
        results.append(result)
        
        if result['success']:
            successful += 1
        else:
            failed += 1
    
    # Summary
    logger.info("=" * 60)
    logger.info("BATCH TRIGGER LAMBDA - COMPLETE")
    logger.info("=" * 60)
    logger.info(f"✅ Successful: {successful}/{len(records)}")
    logger.info(f"❌ Failed: {failed}/{len(records)}")
    
    return {
        'statusCode': 200,
        'body': json.dumps({
            'message': f'Processed {len(records)} message(s)',
            'successful': successful,
            'failed': failed,
            'total': len(records),
            'results': results
        })
    }

