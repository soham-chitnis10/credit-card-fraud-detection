variable "stream_name" {
  description = "The name of the Kinesis stream."
  type        = string

}
variable "shard_count" {
  description = "The number of shards in the Kinesis stream."
  type        = number
  default     = 1

}
variable "retention_period" {
  description = "The retention period of the Kinesis stream in hours."
  type        = number
  default     = 24

}

variable "shard_level_metrics" {
  description = "The shard-level metrics for the Kinesis stream."
  type        = list(string)
  default     = [
    "IncomingBytes",
    "IncomingRecords",
    "OutgoingBytes",
    "OutgoingRecords",
    "WriteProvisionedThroughputExceeded",
    "ReadProvisionedThroughputExceeded",
    "IteratorAgeMilliseconds"
  ]

}

variable "tags" {
  description = "Tags to apply to the Kinesis stream."
  type        = string
  default     = "credit-card-fraud-detection"

}

output "stream_arn" {
    description = "The ARN of the Kinesis stream."
    value       = aws_kinesis_stream.stream.arn

}
