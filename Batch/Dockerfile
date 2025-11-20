# Dockerfile for AWS Batch Invoice Processor
FROM public.ecr.aws/lambda/python:3.11

# Install system dependencies
RUN yum update -y && \
    yum install -y \
    gcc \
    gcc-c++ \
    make \
    python3-devel \
    && ln -sf /usr/bin/g++ /usr/bin/c++ && \
    yum clean all

# Set working directory
WORKDIR /opt

# Copy application code
COPY batch_processor.py /opt/batch_processor.py
COPY lambda_function.py /opt/lambda_function.py
COPY J_J.py /opt/J_J.py

# Copy requirements file
COPY requirements.txt /opt/requirements.txt

# Install Python dependencies
# Install numpy first with pre-built wheel (numpy < 2.0 to avoid GCC 9.3+ requirement)
# Then install pandas 2.0.x which is compatible with numpy 1.x
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel && \
    pip3 install --no-cache-dir --only-binary :all: "numpy==1.26.4" -t /opt/python/ && \
    pip3 install --no-cache-dir --only-binary :all: "pandas>=2.0.0,<2.1.0" -t /opt/python/ && \
    pip3 install --no-cache-dir \
    boto3>=1.28.0 \
    botocore>=1.31.0 \
    jsonschema>=4.17.0 \
    Pillow>=10.0.0 \
    openpyxl>=3.1.0 \
    requests>=2.31.0 \
    -t /opt/python/

# Install PyMuPDF - try to use pre-built wheel only
# Newer versions require C++20 which GCC 7.3.1 doesn't support
# If wheels aren't available, continue build anyway (code handles missing PyMuPDF gracefully)
RUN pip3 install --no-cache-dir --only-binary :all: PyMuPDF==1.23.8 -t /opt/python/ 2>&1 || \
    (pip3 install --no-cache-dir --only-binary :all: PyMuPDF==1.23.0 -t /opt/python/ 2>&1 || \
     (echo "WARNING: PyMuPDF pre-built wheel not available for this platform." && \
      echo "PDF clustering features will be disabled. Code will still run." && \
      true)) && \
    echo "PyMuPDF installation step completed."

# Add Python dependencies to path
ENV PYTHONPATH=/opt/python

# Set entrypoint
ENTRYPOINT ["python3", "batch_processor.py"]

# Default command (can be overridden by Batch job definition)
CMD []

