variable "aws_region" {
  description = "The AWS region to deploy to"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "The name of the project"
  type        = string
  default     = "soccer-prediction"
}

variable "environment" {
  description = "The environment to deploy to (production, staging, development)"
  type        = string
  default     = "development"
}

variable "db_instance_class" {
  description = "The instance class for the RDS database"
  type        = string
  default     = "db.t3.micro"
}

variable "db_username" {
  description = "The username for the RDS database"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "The password for the RDS database"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "The name of the database"
  type        = string
  default     = "soccer_prediction"
}

variable "certificate_arn" {
  description = "The ARN of the SSL certificate for HTTPS"
  type        = string
  default     = ""
}

variable "backend_cpu" {
  description = "The number of CPU units for the backend task"
  type        = string
  default     = "256"
}

variable "backend_memory" {
  description = "The amount of memory for the backend task in MB"
  type        = string
  default     = "512"
}

variable "backend_task_count" {
  description = "The number of backend tasks to run"
  type        = number
  default     = 2
} 