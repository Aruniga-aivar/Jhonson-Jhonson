#!/usr/bin/env python3
"""
AWS Batch Job Script for Invoice Processing
This script processes invoices in AWS Batch, allowing for long-running jobs
that exceed Lambda's 15-minute timeout limit.

Usage:
    python3 batch_processor.py <job_parameters_json>
    
Or as AWS Batch job:
    The job parameters are passed via environment variables or job parameters
"""

import json
import logging
import os
import sys
import boto3
from typing import Dict, Any
from lambda_function import process_single_invoice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# AWS clients
batch_client = boto3.client('batch')
cloudwatch_logs = boto3.client('logs')

def get_job_parameters_from_environment() -> Dict[str, Any]:
    """
    Get job parameters from environment variables (set by AWS Batch).
    AWS Batch can pass parameters via environment variables or job parameters.
    """
    # Check if job parameters are in environment variables
    job_params_json = os.environ.get('JOB_PARAMETERS')
    if job_params_json:
        try:
            return json.loads(job_params_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JOB_PARAMETERS from environment: {e}")
            raise
    
    # Check AWS_BATCH_JOB_ARRAY_INDEX for array jobs
    array_index = os.environ.get('AWS_BATCH_JOB_ARRAY_INDEX')
    job_id = os.environ.get('AWS_BATCH_JOB_ID')
    
    logger.info(f"Batch Job ID: {job_id}")
    logger.info(f"Array Index: {array_index}")
    
    # If no job parameters in environment, return empty dict
    # The parameters should be passed via command line arguments instead
    return {}

def get_job_parameters_from_batch_parameters() -> Dict[str, Any]:
    """
    Get job parameters from AWS Batch job parameters.
    AWS Batch passes parameters via the 'parameters' field in job definition.
    We access them via environment variables that Batch sets.
    """
    # AWS Batch sets job parameters as environment variables
    # The parameter name from job definition becomes the env var name
    job_params_json = os.environ.get('job_parameters')  # Lowercase, matches parameter name
    if job_params_json:
        try:
            return json.loads(job_params_json)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse job_parameters from Batch parameters: {e}")
            raise
    
    return {}

def get_job_parameters_from_args() -> Dict[str, Any]:
    """
    Get job parameters from command line arguments.
    AWS Batch can pass parameters via command line arguments.
    """
    if len(sys.argv) < 2:
        logger.warning("No command line arguments provided")
        return {}
    
    try:
        # First argument should be JSON string with job parameters
        job_params_json = sys.argv[1]
        return json.loads(job_params_json)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse job parameters from command line: {e}")
        logger.error(f"Arguments: {sys.argv}")
        raise
    except IndexError:
        logger.warning("No job parameters provided in command line arguments")
        return {}

def get_job_parameters_from_sqs_message(message_body: str) -> Dict[str, Any]:
    """
    Get job parameters from SQS message body.
    This is used when Batch job is triggered by SQS message.
    """
    try:
        return json.loads(message_body)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse SQS message body: {e}")
        raise

def get_job_parameters_from_individual_env_vars() -> Dict[str, Any]:
    """
    Get job parameters from individual environment variables.
    This is used when parameters are passed via containerOverrides environment variables.
    """
    job_params = {}
    
    # Required parameters
    input_key = os.environ.get('input_key')
    input_bucket = os.environ.get('input_bucket')
    job_id = os.environ.get('job_id')
    attachment_id = os.environ.get('attachment_id')
    
    # Check if we have at least the required parameters
    if input_key and input_bucket and job_id and attachment_id:
        job_params['input_key'] = input_key
        job_params['input_bucket'] = input_bucket
        job_params['job_id'] = job_id
        job_params['attachment_id'] = attachment_id
        
        # Optional parameters
        if os.environ.get('output_bucket'):
            job_params['output_bucket'] = os.environ.get('output_bucket')
        
        if os.environ.get('output_prefix'):
            job_params['output_prefix'] = os.environ.get('output_prefix')
        
        if os.environ.get('carrier_name'):
            job_params['carrier_name'] = os.environ.get('carrier_name')
        
        if os.environ.get('s3_bucket'):
            job_params['s3_bucket'] = os.environ.get('s3_bucket')
        
        if os.environ.get('excel_file_path'):
            job_params['excel_file_path'] = os.environ.get('excel_file_path')
        
        if os.environ.get('use_textract'):
            job_params['use_textract'] = os.environ.get('use_textract').lower() in ('true', '1', 'yes')
        
        if os.environ.get('use_s3_clustering'):
            job_params['use_s3_clustering'] = os.environ.get('use_s3_clustering').lower() in ('true', '1', 'yes')
        
        if os.environ.get('email_attachments'):
            job_params['email_attachments'] = os.environ.get('email_attachments')
        
        # Parse email_details if provided as JSON string
        email_details_str = os.environ.get('email_details')
        if email_details_str:
            try:
                job_params['email_details'] = json.loads(email_details_str)
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse email_details from environment: {email_details_str}")
        
        return job_params
    
    return {}

def update_batch_job_status(job_id: str, status: str, reason: str = None):
    """
    Update Batch job status (for monitoring purposes).
    Note: AWS Batch automatically tracks job status, but we can log it.
    """
    logger.info(f"Batch Job Status Update - Job ID: {job_id}, Status: {status}")
    if reason:
        logger.info(f"Reason: {reason}")

def main():
    """
    Main entry point for AWS Batch job.
    """
    logger.info("=" * 60)
    logger.info("AWS BATCH INVOICE PROCESSOR - STARTING")
    logger.info("=" * 60)
    
    # Get Batch job metadata
    job_id = os.environ.get('AWS_BATCH_JOB_ID', 'unknown')
    job_name = os.environ.get('AWS_BATCH_JOB_NAME', 'unknown')
    job_queue = os.environ.get('AWS_BATCH_JOB_QUEUE', 'unknown')
    job_attempt = os.environ.get('AWS_BATCH_JOB_ATTEMPT', '1')
    
    logger.info(f"Batch Job ID: {job_id}")
    logger.info(f"Batch Job Name: {job_name}")
    logger.info(f"Batch Job Queue: {job_queue}")
    logger.info(f"Batch Job Attempt: {job_attempt}")
    
    try:
        # Try to get job parameters from multiple sources (in priority order)
        job_params = {}
        
        # 1. Try AWS Batch job parameters (from job definition parameters)
        job_params = get_job_parameters_from_batch_parameters()
        if job_params:
            logger.info("✅ Loaded job parameters from AWS Batch job parameters")
        
        # 2. Try environment variables (JOB_PARAMETERS)
        if not job_params:
            job_params = get_job_parameters_from_environment()
            if job_params:
                logger.info("✅ Loaded job parameters from environment variables")
        
        # 3. Try command line arguments
        if not job_params:
            job_params = get_job_parameters_from_args()
            if job_params:
                logger.info("✅ Loaded job parameters from command line arguments")
        
        # 4. Try individual environment variables (from containerOverrides)
        if not job_params:
            job_params = get_job_parameters_from_individual_env_vars()
            if job_params:
                logger.info("✅ Loaded job parameters from individual environment variables")
        
        # 5. If still no parameters, check if we're processing an SQS message
        if not job_params:
            # Check if SQS message body is in environment (set by trigger Lambda)
            sqs_message_body = os.environ.get('SQS_MESSAGE_BODY')
            if sqs_message_body:
                job_params = get_job_parameters_from_sqs_message(sqs_message_body)
                logger.info("✅ Loaded job parameters from SQS message body")
        
        if not job_params:
            error_msg = "No job parameters found in environment variables, command line arguments, or SQS message"
            logger.error(error_msg)
            logger.error("Available environment variables:")
            for key, value in os.environ.items():
                if 'BATCH' in key or 'JOB' in key or 'SQS' in key:
                    logger.error(f"  {key}: {value}")
            raise ValueError(error_msg)
        
        logger.info(f"Job parameters loaded: {json.dumps(job_params, indent=2)}")
        
        # Validate required parameters
        required_params = ['input_key', 'input_bucket', 'job_id', 'attachment_id']
        missing_params = [param for param in required_params if param not in job_params]
        if missing_params:
            error_msg = f"Missing required job parameters: {missing_params}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        # Process the invoice using the existing process_single_invoice function
        logger.info("=" * 60)
        logger.info("PROCESSING INVOICE")
        logger.info("=" * 60)
        
        # The process_single_invoice function expects an event dict similar to Lambda event
        # We'll create a compatible event structure
        invoice_event = {
            'input_key': job_params['input_key'],
            'input_bucket': job_params['input_bucket'],
            'output_bucket': job_params.get('output_bucket', job_params['input_bucket']),
            'output_prefix': job_params.get('output_prefix', 'invoice/output/'),
            'carrier_name': job_params.get('carrier_name'),
            'use_textract': job_params.get('use_textract', True),
            'use_s3_clustering': job_params.get('use_s3_clustering', True),
            's3_bucket': job_params.get('s3_bucket', job_params['input_bucket']),
            'excel_file_path': job_params.get('excel_file_path', 'templates/Excel_mapping/value_mapping.xlsx'),
            'email_details': job_params.get('email_details', {}),
            'job_id': job_params['job_id'],
            'attachment_id': job_params['attachment_id'],
            'email_attachments': job_params.get('email_attachments')  # S3 path to email attachments
        }
        
        # Process the invoice
        result = process_single_invoice(invoice_event, None)
        
        # Log the result
        logger.info("=" * 60)
        logger.info("INVOICE PROCESSING COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Result: {json.dumps(result, indent=2, default=str)}")
        
        # Check if processing was successful
        if result.get('statusCode') == 200:
            logger.info("✅ Invoice processing completed successfully")
            update_batch_job_status(job_id, 'SUCCEEDED')
            sys.exit(0)
        else:
            error_body = result.get('body', 'Unknown error')
            logger.error(f"❌ Invoice processing failed: {error_body}")
            update_batch_job_status(job_id, 'FAILED', error_body)
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ Fatal error in Batch job: {str(e)}", exc_info=True)
        update_batch_job_status(job_id, 'FAILED', str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()

