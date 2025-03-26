variable "location" {
  description = "The Azure region to deploy to"
  type        = string
  default     = "eastus"
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

variable "db_sku" {
  description = "The SKU of the PostgreSQL Flexible Server"
  type        = string
  default     = "B_Standard_B1ms"
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

variable "kubernetes_version" {
  description = "The version of Kubernetes to use"
  type        = string
  default     = "1.27.7"
}

variable "aks_node_count" {
  description = "The number of nodes in the AKS cluster"
  type        = number
  default     = 2
}

variable "aks_vm_size" {
  description = "The VM size for the AKS nodes"
  type        = string
  default     = "Standard_DS2_v2"
} 