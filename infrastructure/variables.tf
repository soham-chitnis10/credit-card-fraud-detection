variable "aws_region" {
  description = "The AWS region where the resources will be created."
  type        = string
  default     = "us-east-2"

}

variable "project_id" {
  description = "The unique identifier for the project."
  type        = string
  default     = "credit-card-fraud-detection"

}

variable "source_stream_name" {
  description = "The base name for the source Kinesis stream."
  type        = string

}

variable "output_stream_name" {
  description = "The base name for the output Kinesis stream."
  type        = string

}
variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string
  default     = "kinesis-lambda-function"

}

variable "ecr_repo_name" {
  description = "Name of the ECR repository for the Lambda function image"
  type        = string

}
