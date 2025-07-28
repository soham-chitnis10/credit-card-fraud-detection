terraform {
    required_version = ">= 1.0"
    backend "s3" {
        bucket = "mlops-course-2025"
        key    = "mlops-project.tfstate"
        region = "us-east-2"
        encrypt = true
    }
}

provider "aws" {
    region = var.aws_region
}

data aws_caller_identity "current_identity" {}

locals {
  account_id = data.aws_caller_identity.current_identity.account_id
  ecr_registry = "${local.account_id}.dkr.ecr.${var.aws_region}.amazonaws.com"
}

# trans_event stream
module "source_kinesis_stream" {
    source  = "./modules/kinesis"
    stream_name = "${var.source_stream_name}-${var.project_id}"
    tags     = var.project_id
}

# trans_event_prediction stream
module "output_kinesis_stream" {
    source  = "./modules/kinesis"
    stream_name = "${var.output_stream_name}-${var.project_id}"
    tags     = var.project_id
}


module "lambda_function" {
    source = "./modules/lambda"
    lambda_function_name = "${var.lambda_function_name}-${var.project_id}"
    image_uri            = "${local.ecr_registry}/${var.ecr_repo_name}:stream-credit-card-fraud-detection"
    source_stream_name   = "${var.source_stream_name}-${var.project_id}"
    source_stream_arn    = module.source_kinesis_stream.stream_arn
    output_stream_name   = "${var.output_stream_name}-${var.project_id}"
    output_stream_arn    = module.output_kinesis_stream.stream_arn
}
