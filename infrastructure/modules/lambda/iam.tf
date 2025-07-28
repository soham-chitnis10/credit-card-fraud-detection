resource "aws_iam_role" "lambda_for_kinesis" {
    name = "lambda_for_kinesis"
    assume_role_policy = jsonencode({
        Version = "2012-10-17"
        Statement = [
            {
                Action = "sts:AssumeRole"
                Effect = "Allow"
                Principal = {
                    Service = ["lambda.amazonaws.com", "kinesis.amazonaws.com"]

                }
            }
        ]
    })
}

resource "aws_iam_policy" "allow_kinesis_processing" {
  name        = "allow_kinesis_processing_${var.lambda_function_name}"
  path        = "/"
  description = "IAM policy for logging from a lambda"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "kinesis:ListShards",
        "kinesis:ListStreams",
        "kinesis:*"
      ],
      "Resource": "arn:aws:kinesis:*:*:*",
      "Effect": "Allow"
    },
    {
      "Action": [
        "stream:GetRecord",
        "stream:GetShardIterator",
        "stream:DescribeStream",
        "stream:*"
      ],
      "Resource": "arn:aws:stream:*:*:*",
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "kinesis_processing" {
  role       = aws_iam_role.lambda_for_kinesis.name
  policy_arn = aws_iam_policy.allow_kinesis_processing.arn
}

resource "aws_iam_role_policy" "inline_lambda_policy" {
  name       = "LambdaInlinePolicy"
  role       = aws_iam_role.lambda_for_kinesis.id
  depends_on = [aws_iam_role.lambda_for_kinesis]
  policy     = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "kinesis:PutRecords",
        "kinesis:PutRecord"
      ],
      "Resource": "${var.output_stream_arn}"
    }
  ]
}
EOF
}

# IAM for CW

resource "aws_lambda_permission" "allow_cloudwatch_to_trigger_lambda_function" {
  statement_id  = "AllowExecutionFromCloudWatch"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.kinesis_lambda.function_name
  principal     = "events.amazonaws.com"
  source_arn    = var.source_stream_arn
}

resource "aws_iam_policy" "allow_logging" {
  name        = "allow_logging_${var.lambda_function_name}"
  path        = "/"
  description = "IAM policy for logging from a lambda"

  policy = <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:*:*:*",
      "Effect": "Allow"
    }
  ]
}
EOF
}

resource "aws_iam_role_policy_attachment" "lambda_logs" {
  role       = aws_iam_role.lambda_for_kinesis.name
  policy_arn = aws_iam_policy.allow_logging.arn
}
