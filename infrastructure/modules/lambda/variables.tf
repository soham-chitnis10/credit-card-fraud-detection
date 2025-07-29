variable "lambda_function_name" {
  description = "Name of the Lambda function"
  type        = string

}

variable "image_uri" {
  description = "URI of the Lambda function image"
  type        = string

}

variable "source_stream_name" {
  type        = string
  description = "Source Kinesis Data Streams stream name"
}

variable "source_stream_arn" {
  type        = string
  description = "Source Kinesis Data Streams stream name"
}

variable "output_stream_name" {
  description = "Name of output stream where all the events will be passed"
}

variable "output_stream_arn" {
  description = "ARN of output stream where all the events will be passed"
}

variable "mlflow_tracking_uri" {
  description = "MLflow tracking URI"
  type        = string
}
