variable "project_id" {
  description = "The ID of the GCP project"
  type        = string
}

variable "region" {
  description = "The GCP region to deploy to"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone to deploy to"
  type        = string
  default     = "us-central1-a"
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

variable "db_tier" {
  description = "The tier of the SQL database"
  type        = string
  default     = "db-f1-micro"
}

variable "db_username" {
  description = "The username for the database"
  type        = string
  sensitive   = true
}

variable "db_password" {
  description = "The password for the database"
  type        = string
  sensitive   = true
}

variable "db_name" {
  description = "The name of the database"
  type        = string
  default     = "soccer_prediction"
}

variable "gke_num_nodes" {
  description = "The number of nodes in the GKE cluster"
  type        = number
  default     = 2
}

variable "gke_machine_type" {
  description = "The machine type for the GKE nodes"
  type        = string
  default     = "e2-medium"
}